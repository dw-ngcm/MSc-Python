import math
import numpy
import pyaudio
import time

# Dictionary to translate between API names and the pyaudio enumeration values
paApiIds = { 'ASIO': pyaudio.paASIO,
             'CoreAudio': pyaudio.paCoreAudio,
             'DirectSound': pyaudio.paDirectSound,
             'MME': pyaudio.paMME,
             'SoundManager': pyaudio.paSoundManager,
             'OSS': pyaudio.paOSS,
             'ALSA': pyaudio.paALSA,
             'AL': pyaudio.paAL,
             'BeOs': pyaudio.paBeOS,
             'WDMKS': pyaudio.paWDMKS,
             'JACK': pyaudio.paJACK,
             'WASAPI': pyaudio.paWASAPI,
             'NoDevice': pyaudio.paNoDevice,           # Pseudo device
             'inDevelopment': pyaudio.paInDevelopment  # Pseudo device
           }

def duplexAudio( outputSignal, samplingFrequency,
                 blockLength,
                 recordChannels=None,
                 recordLength = None, audioApi=None,
                 inputDevice=None,
                 outputDevice=None,
                 compensateLatency=False,
                 extraLatencySamples=0):

    def callback(in_data, frame_count, time_info, status):
        try:
            idx = signalIndex[0] # Retrieve the value of the signal index held in the enclosing scope
            # print("Callback: signal index: %d, signal length: %d" % (signalIndex[0], outputSignal.shape[0]))
            if frame_count != blockLength:
                # Informational message: The code should be able to cope with arbitrary frame counts
                # (unless we exceed the total length of input/output signals in the current iteration)
                print( "Unexpected block size: %d instead of %d" % (frame_count, blockLength ))
            if idx >= outputSignal.shape[0]:
                # print("Complete: signal index: %d, signal length: %d" % (idx, outputSignal.shape[0]))
                return (None, pyaudio.paComplete )
            else:
                inSigRaw = numpy.fromstring( in_data, dtype=numpy.float32)
                inSig = numpy.reshape( inSigRaw, [-1, paChannels] )
                recordSignal[idx:idx+frame_count,:] = inSig
                signalChunk = outputSignal[idx:(idx+frame_count),:]
                signalRaw = numpy.reshape( signalChunk, [-1,1] )
                out_data = signalRaw.astype(numpy.float32).tostring()
                signalIndex[0] = idx + frame_count
                return (out_data, pyaudio.paContinue)
        except BaseException as ex:
            print( "Error during callback: %s!" % ex.message )
            return (None, pyaudio.paAbort )

    p = pyaudio.PyAudio()

    if audioApi == None:
        hostApiInfo = p.get_default_host_api_info()
    else:
        # Both calls might throw an exception
        # try:
        hostApiType = paApiIds[audioApi]
        hostApiInfo = p.get_host_api_info_by_type( hostApiType )



    if outputSignal.ndim == 1:
        # Transform a signal vector as returned by scipy.io.wavfile.read()
        # in case of a mono file into a 2D matrix in order to hace a consistent
        # handling in this function.
        outputSignal = numpy.reshape( outputSignal, [outputSignal.shape[0],1])
    numOutputChannels = outputSignal.shape[1]
    if recordChannels == None:
        numInputChannels = numOutputChannels
    else:
        numInputChannels = recordChannels
    # In the current version, the number if playback and capture channels must be identical
    # in duplex mode. We emulate that by padding either the input or output signal.
    paChannels = max( numInputChannels, numOutputChannels )

    if recordLength == None:
        recordLengthSamples = outputSignal.shape[0]
    else:
        recordLengthSamples = int(math.ceil( float(recordLength) * samplingFrequency ))

    if inputDevice == None:
        inputDevice = hostApiInfo['defaultInputDevice']
    if outputDevice == None:
        outputDevice = hostApiInfo['defaultOutputDevice']

    if not p.is_format_supported( rate=samplingFrequency,
                                  input_device=inputDevice,
                                  output_device=outputDevice,
                                  input_channels=paChannels,
                                  output_channels=paChannels,
                                  input_format=pyaudio.paFloat32,
                                  output_format=pyaudio.paFloat32 ):
        raise RuntimeError( "The parameters are not supported by the audio API" )

    stream = p.open(format=pyaudio.paFloat32,
                    input_device_index=inputDevice,
                    output_device_index=outputDevice,
                    frames_per_buffer=blockLength,
                    channels=paChannels,
                    rate=samplingFrequency,
                    input=True, output=True,
                    start=False, # wait for the stream.start() call
                    stream_callback=callback )

    if compensateLatency == True:
        totalIoLatencySec = stream.get_input_latency() + stream.get_output_latency()
        compensateLatencySamples = numpy.round( totalIoLatencySec*samplingFrequency ) + extraLatencySamples
    else:
        compensateLatencySamples = extraLatencySamples

    # Integer 'round up to the next integral number of blocks'
    numBlocks =  int(numpy.ceil(float(recordLengthSamples+compensateLatencySamples)/blockLength))
    signalLength = numBlocks * blockLength;

    if outputSignal.shape[0] < signalLength:
        # zero-pad the signal
        if outputSignal.ndim == 1:
            outputSignal = numpy.concatenate( (outputSignal,
                                          numpy.zeros( [ signalLength -     outputSignal.shape[0] ]) ))
        else:
            outputSignal = numpy.concatenate( (outputSignal,
                                          numpy.zeros( [ signalLength - outputSignal.shape[0],
                                                         numOutputChannels ]) ))
    elif outputSignal.shape[0] > signalLength:
        # Truncate to the recorded length
        outputSignal = outputSignal[0:signalLength,:]

    # Zero-pad the number of output channels if it is smaller than the number of
    # recorded channels
    # TODO: This has not been checked with monaural input (signal vector) yet.
    if numOutputChannels < paChannels:
        outputSignal = numpy.concatenate( (outputSignal,
                                           numpy.zeros( [outputSignal.shape[0],
                                                         paChannels-numOutputChannels]
                                          )),
                                          axis = 1)

    recordSignal = numpy.zeros( [signalLength, paChannels] )
    # Caveat: In order to make the current signal index accessible in the nested callback function,
    # we need to put it in an array, because we cannot reference a variable itself.
    signalIndex = [0];

    # print( "Starting stream.")
    stream.start_stream()

    while stream.is_active():
        # print( "Streaming..." )
        time.sleep( 0.1 )

    stream.stop_stream()
    # print( "Stream stopped.")

    p.terminate()

    # If the recorded length is longer than the requested one (if it is not an
    # integral multiple of the block length )
    #if signalLength > recordLengthSamples:
    recordSignal = recordSignal[compensateLatencySamples:compensateLatencySamples+recordLengthSamples,:]

    if (recordLength is None) or (numInputChannels is 0):
        return

    # If the recorded channels have been padded to the number of playback channels,
    # discard the unwanted ones.
    if paChannels > numInputChannels:
        recordSignal = recordSignal[:,0:numInputChannels]

    # Remove unwanted singleton dimensions
    return numpy.squeeze(recordSignal)
