# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 17:34:23 2019

@author: chalbeisen

This program is to download audios. The required arguments are set by the 
powershell script "Run-audio_download.ps1".
"""

from LookingToListen_Audio_clean import Audio
import argparse
import sys

'''
------------------------------------------------------------------------------
desc:      get parameters from script "Run-audio_download.ps1"
param:    
    argv:    arguments from script "Run-audio_download.ps1" 
return:    parsed arguments
------------------------------------------------------------------------------
'''
    
def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str,
        help='Define directory of audios', default='audio/audio_speech')
    parser.add_argument('--start', type=int,
        help='Define start index of audio download', default=0)
    parser.add_argument('--stop', type=int,
        help='Define start index of audio download', default=123800)
    return parser.parse_args(argv)

'''
------------------------------------------------------------------------------
desc:      run audio download
param:    
    argv:    arguments from script "Run-audio_download.ps1" 
return:    -
------------------------------------------------------------------------------
'''
    
def main(argv):  
    
    audio = Audio()
    audio.download_speech(argv.dir, argv.start, argv.stop)

if __name__== "__main__":
    main(parse_arguments(sys.argv[1:]))