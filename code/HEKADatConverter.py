#!/usr/bin/env python
# coding: utf-8
# by Junzhanj

from HekaHelpers import HekaBundleInfo
import numpy as np
from pathlib import Path 
import struct
import numpy as np

def writeABF1(sweepData, filename, sampleRateHz, units='pA'):
    """
    Create an ABF1 file from scratch and write it to disk.
    Files created with this function are compatible with MiniAnalysis.
    Data is expected to be a 2D numpy array (each row is a sweep).
    Credit to pyABF
    """

    assert isinstance(sweepData, np.ndarray)

    # constants for ABF1 files
    BLOCKSIZE = 512
    HEADER_BLOCKS = 4

    # determine dimensions of data
    sweepCount = sweepData.shape[0]
    sweepPointCount = sweepData.shape[1]
    dataPointCount = sweepPointCount*sweepCount

    # predict how large our file must be and create a byte array of that size
    bytesPerPoint = 2
    dataBlocks = int(dataPointCount * bytesPerPoint / BLOCKSIZE) + 1
    data = bytearray((dataBlocks + HEADER_BLOCKS) * BLOCKSIZE)

    # populate only the useful header data values
    struct.pack_into('4s', data, 0, b'ABF ')  # fFileSignature
    struct.pack_into('f', data, 4, 1.3)  # fFileVersionNumber
    struct.pack_into('h', data, 8, 5)  # nOperationMode (5 is episodic)
    struct.pack_into('i', data, 10, dataPointCount)  # lActualAcqLength
    struct.pack_into('i', data, 16, sweepCount)  # lActualEpisodes
    struct.pack_into('i', data, 40, HEADER_BLOCKS)  # lDataSectionPtr
    struct.pack_into('h', data, 100, 0)  # nDataFormat is 1 for float32
    struct.pack_into('h', data, 120, 1)  # nADCNumChannels
    struct.pack_into('f', data, 122, 1e6 / sampleRateHz)  # fADCSampleInterval
    struct.pack_into('i', data, 138, sweepPointCount)  # lNumSamplesPerEpisode

    # These ADC adjustments are used for integer conversion. It's a good idea
    # to populate these with non-zero values even when using float32 notation
    # to avoid divide-by-zero errors when loading ABFs.

    fSignalGain = 1  # always 1
    fADCProgrammableGain = 1  # always 1
    lADCResolution = 2**15  # 16-bit signed = +/- 32768

    # determine the peak data deviation from zero
    maxVal = np.max(np.abs(sweepData))

    # set the scaling factor to be the biggest allowable to accommodate the data
    fInstrumentScaleFactor = 100
    for i in range(10):
        fInstrumentScaleFactor /= 10
        fADCRange = 10
        valueScale = lADCResolution / fADCRange * fInstrumentScaleFactor
        maxDeviationFromZero = 32767 / valueScale
        if (maxDeviationFromZero >= maxVal):
            break

    # prepare units as a space-padded 8-byte string
    unitString = units
    while len(unitString) < 8:
        unitString = unitString + " "

    # store the scale data in the header
    struct.pack_into('i', data, 252, lADCResolution)
    struct.pack_into('f', data, 244, fADCRange)
    for i in range(16):
        struct.pack_into('f', data, 922+i*4, fInstrumentScaleFactor)
        struct.pack_into('f', data, 1050+i*4, fSignalGain)
        struct.pack_into('f', data, 730+i*4, fADCProgrammableGain)
        struct.pack_into('8s', data, 602+i*8, unitString.encode())

    # fill data portion with scaled data from signal
    dataByteOffset = BLOCKSIZE * HEADER_BLOCKS
    for sweepNumber, sweepSignal in enumerate(sweepData):
        sweepByteOffset = sweepNumber * sweepPointCount * bytesPerPoint
        for valueNumber, value in enumerate(sweepSignal):
            valueByteOffset = valueNumber * bytesPerPoint
            bytePosition = dataByteOffset + sweepByteOffset + valueByteOffset
            struct.pack_into('h', data, bytePosition, int(value*valueScale))

    # save the byte array to disk
    with open(filename, 'wb') as f:
        f.write(data)
    return


def convert_dat_to_ABF(input_dir=None, file=None, output_dir=None):
    bundleTester = HekaBundleInfo(input_dir + file)
    countGroups = bundleTester.countGroups()
    print('Current data has',countGroups,'Group(s):')
    for i in range(countGroups):
        countSeries = bundleTester.countSeries([i])
        group_label = bundleTester.getGroupRecord([i]).Label
        print('group', i+1,group_label,'has',countSeries,'Series:')
        for j in range(countSeries):
            countSweeps = bundleTester.countSweeps([i,j])
            SeriesLabel = bundleTester.getSeriesLabel([i,j])
            countTraces = bundleTester.countTraces([i,j,0])
            Seriesname = SeriesLabel.split()[-1]
            print('        Series',j+1,':',SeriesLabel,'with',countSweeps,'sweep(s) and',countTraces,'trace(s)/sweep')
            if SeriesLabel.startswith("*"):
                SeriesLabel = SeriesLabel[1:]
            for t in range(countTraces):
                sweep_length = bundleTester.getNumberOfSamplesPerSweep([i,j,0,t])          
                matrix = np.zeros(shape = (sweep_length,countSweeps))
                stimInfo = []
                for k in range(countSweeps):
                    traceIndex = [i,j,k,t]
                    matrix[:,k][0:len(bundleTester.getSingleTraceData(traceIndex))] = bundleTester.getSingleTraceData(traceIndex)
                    time, stim, stimInfo1 = bundleTester.getStim(traceIndex)
                    stimInfo.append(stimInfo1)
                unit = bundleTester.getTraceRecord([i,j,0,t]).YUnit
                trace_label = bundleTester.getTraceRecord([i,j,0,t]).Label
                if unit == 'V':
                    unit = 'mV'
                    scale = 1e3
                if unit == 'A':
                    unit = 'pA'
                    scale = 1e12
                matrix = matrix.T*scale
                freq = round(1/stimInfo[0][0]['sampleInteval'])
                
                if countGroups == 1:
                    if countTraces == 1:
                        writeABF1(matrix,output_dir+Path(file).stem+'_Serie_'+str(j+1)+'_'
                                  +SeriesLabel+'.abf', freq, units=unit)
                    else:
                        writeABF1(matrix,output_dir+Path(file).stem+'_Serie_'+str(j+1)+'_'
                                  +SeriesLabel+'_'+trace_label+'.abf', freq, units=unit)                    
                else:
                    if countTraces == 1:
                        writeABF1(matrix,output_dir+Path(file).stem+'_'+group_label+'_'
                                  +'_Serie_'+str(j+1)+'_'+SeriesLabel+'.abf', freq, units=unit)
                    else:
                        writeABF1(matrix,output_dir+Path(file).stem+'_'+group_label+'_'
                                  +'_Serie_'+str(j+1)+'_'+SeriesLabel+'_'+trace_label+'.abf', freq, units=unit)



