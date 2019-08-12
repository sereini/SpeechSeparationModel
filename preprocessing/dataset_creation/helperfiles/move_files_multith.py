# -*- coding: utf-8 -*-
"""
Created on Sat Jun 29 11:56:30 2019

@author: chalbeisen
This program is used to copy files from a defined source to a defined destination
and remove the files afterwards from source
"""
import os
import argparse
import subprocess
from LookingToListen_Audio_clean import Audio
import sys
import shutil
import RunCMD
import threading
import time
from copy_files import CopyFiles

#global variables
src_dirs =  dest_dirs = audio = data_dir = copyFile =  None
subps = []
'''
------------------------------------------------------------------------------
desc:      copy all files (wav and stft of audio from av_speech dataset, mixed audios, embeddings and tfRecords) from source
param:     -
return:    -
------------------------------------------------------------------------------
'''
def copy_files():
    thread_audio_wav = threading.Thread(target = copyFile.copy_audio_speech_wav, args = ())
    thread_audio_npy = threading.Thread(target = copyFile.copy_audio_speech_npy, args = ())
    thread_mixed = threading.Thread(target = copyFile.copy_audio_speech_mixed, args = ())
    thread_visual = threading.Thread(target = copyFile.copy_visual, args = ())
    thread_records = threading.Thread(target = copyFile.copy_records, args = ())
    
    thread_audio_wav.start()
    thread_audio_npy.start()
    thread_mixed.start()
    thread_visual.start()
    thread_records.start()
    
    thread_audio_wav.join()
    thread_audio_npy.join()
    thread_mixed.join()
    thread_visual.join()
    thread_records.join()

'''
------------------------------------------------------------------------------
desc:      remove files from source 
param:     -
return:    -
------------------------------------------------------------------------------
'''
def remove_files():
    thread_audio_speech = threading.Thread(target = copyFile.remove_audio_speech, args = ())
    thread_mixed = threading.Thread(target = copyFile.remove_audio_speech_mixed, args = ())
    thread_visual = threading.Thread(target = copyFile.remove_visual, args = ())
    thread_records = threading.Thread(target = copyFile.remove_records, args = ())
    
    thread_audio_speech.start()
    thread_mixed.start()
    thread_visual.start()
    thread_records.start()
    
    thread_audio_speech.join()
    thread_mixed.join()
    thread_visual.join()
    thread_records.join()

'''
------------------------------------------------------------------------------
desc:      create required directories
param:    
    dirs:    directories to create
return:    -
------------------------------------------------------------------------------
'''            
def create_directories(dirs):
    for (key, value) in dirs.items():
        if not os.path.exists(value):
            os.makedirs(value)

'''
------------------------------------------------------------------------------
desc:      get parameters from script "Run-copy_files_multith.ps1"
param:    
    argv:    arguments from script "Run-copy_files_multith.ps1" 
return:    parsed arguments
------------------------------------------------------------------------------
'''                 
def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_dir', type=str,
        help='Define source directory of matched files', default='D:/cnn_data2/visual/')
    parser.add_argument('--dest_dir', type=str,
        help='Define destination directory of matched files', default='D:/audio_speech/')
    
    return parser.parse_args(argv)
 
'''
------------------------------------------------------------------------------
desc:      copy all files from source, if copying is completed remove files
           measures required time for copying and removing files
param:     
    argv:    arguments from script "Run-copy_files_multith.ps1" 
return:    -
------------------------------------------------------------------------------
'''            
def main(argv):  
    start = time.time()
    global src_dirs, dest_dirs, data_dir, audio, subps, copyFile
    dirs = {'audio_speech':'audio/audio_speech/','audio_speech_wav':'audio/audio_speech/wav_files/','audio_noise/':'audio/audio_noise','audio_noise_wav':'audio/audio_noise/wav_files/','audio_speech_mixed':'audio/audio_speech_mixed/', 'audio_speech_noise':'audio/audio_speech_noise/', 'visual':'visual/', 'records':'records/'}
    src_dir = argv.src_dir
    dest_dir = argv.dest_dir
    src_dirs = {key: src_dir+value for (key, value) in dirs.items()}
    dest_dirs = {key: dest_dir+value for (key, value) in dirs.items()}
    create_directories(dest_dirs)
    
    audio = Audio()
    
    copyFile = CopyFiles(src_dirs,dest_dirs)
    copy_files()
    copyFile.wait_for_subprocesses()
    print("DONE COPYING FILES")
    end = time.time()
    time_elapsed = int((end-start)/60)
    print("time elapsed for copying files: "+str(time_elapsed)+" min")
    remove_files()

    
if __name__== "__main__":
    main(parse_arguments(sys.argv[1:]))