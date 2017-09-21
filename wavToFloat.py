# -*- coding: utf-8 -*-
"""
Function to convert audio data returned by scipy.io.wavfile into floating-point arrays. In case of integer-type wav files, this includes a rescaling to [-1,1).
Created on Sun Apr 10 07:54:15 2016

Copyright ISVR 2016 - All tights reserved

@author: Andreas Franck (a.franck@soton.ac.uk)
"""

import numpy

def wavToFloat( x ):
    if x.dtype == numpy.int16:
        return 1.0/float( 1 << 15 ) * x
    elif x.dtype == numpy.int32:
        return 1.0/float( 1 << 31 ) * x
    else:
        return x # Don't change the sequence
    # wavio does not support 24-bit integer WAVs?
    # No handling required for floating-point WAVs
    x = wavToFloat( x )