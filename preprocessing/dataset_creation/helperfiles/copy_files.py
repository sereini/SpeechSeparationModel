# -*- coding: utf-8 -*-
"""
Created on Sat Jun 29 11:56:30 2019

@author: chalbeisen
This program is used for copying specific files (audios, visuals)
and removing specific files (audios, visuals)
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
import re
import pyfastcopy

#global variable
subps = []

class CopyFiles:
    '''
    ------------------------------------------------------------------------------
    desc:      init CopyFiles class and set global parameters, 
               comments contain another way of copying with robocopy
    param:    -
    return:    -
    ------------------------------------------------------------------------------
    '''
    def __init__(self, src_dirs, dest_dirs):
        self.src_dirs = src_dirs
        self.dest_dirs = dest_dirs 
        
    def copyfile(self, src, dest):
        try:
            shutil.copy(src,dest)
        except Exception as e:
            print("Exception: ",e)
            print("could not copy")
        # src = src.replace('/', '\\')
        # dest = dest.replace('/', '\\')
        # cmd = ['xcopy', src, dest, "/K/O/X"] 
        # cmd = ['robocopy', src, dest] 
        # RunCMD.RunCmd(cmd,200,5,False).Run()
        
    '''
    ------------------------------------------------------------------------------
    desc:      copy files from defined source to defined destination,
               comments contain another way of copying with robocopy
    param:    
        src:      source filename
        dest:      destination directory
    return:    -
    ------------------------------------------------------------------------------
    '''   
    def copydir(self, src, dest):
        for file in os.listdir(src):
            if not os.path.exists(dest+file):
                print("copying ",src+file)
                try:
                    shutil.copyfile(src+file,dest+file)
                except (OSError, IOError):
                    print("test")
            else:
                print("file "+file+" already exists")
#        src = src.replace('/', '\\')
#        src = src[:-1]
#        dest = dest.replace('/', '\\')
#        dest = dest[:-1]
#        cmd = 'robocopy '+'"'+src+'"'+' '+'"'+dest+'"'+' /E /XC /XN /XO'
#        subps.append(subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE))
                
    '''
    ------------------------------------------------------------------------------
    desc:      wait for subprocess in subps array to complete
    param:     -
    return:    -
    ------------------------------------------------------------------------------
    '''  
    def wait_for_subprocesses(self):
        exit_codes = [p.communicate() for p in subps]
        print(exit_codes)
 
    '''
    ------------------------------------------------------------------------------
    desc:      copy audios in ".wav"-format to destination
    param:     -
    return:    -
    ------------------------------------------------------------------------------
    '''  
    def copy_audio_speech_wav(self):
        print("..copying wav files")
        for file in os.listdir(self.src_dirs['audio_speech_wav']):
            print(self.src_dirs['audio_speech_wav']+file)
            if not os.path.exists(self.dest_dirs['audio_speech_wav']+file):
                self.copyfile(self.src_dirs['audio_speech_wav']+file,self.dest_dirs['audio_speech_wav'])
                print(file)
        print("copying wav files done")  

    '''
    ------------------------------------------------------------------------------
    desc:      copy stft of audios to destination
    param:     -
    return:    -
    ------------------------------------------------------------------------------
    '''             
    def copy_audio_speech_npy(self):
        print("..copying npy files")
        for file in os.listdir(self.src_dirs['audio_speech']):
            if not os.path.exists(self.dest_dirs['audio_speech']+file):
                self.copyfile(self.src_dirs['audio_speech']+file,self.dest_dirs['audio_speech'])
                print(file)
        print("copying npy files done") 
        
    '''
    ------------------------------------------------------------------------------
    desc:      copy stft of mixed audios to destination
    param:     -
    return:    -
    ------------------------------------------------------------------------------
    '''     
    def copy_audio_speech_mixed(self):
        print("..copying wav files")
        for file in os.listdir(self.src_dirs['audio_speech_mixed']):
            if not os.path.exists(self.dest_dirs['audio_speech_mixed']+file):
                self.copyfile(self.src_dirs['audio_speech_mixed']+file,self.dest_dirs['audio_speech_mixed'])
                print(file)
        print("copying wav files done") 
        
    '''
    ------------------------------------------------------------------------------
    desc:      copy embeddings to destination
    param:     -
    return:    -
    ------------------------------------------------------------------------------
    '''                    
    def copy_visual(self):
        print("..copying visual files")
        for file in os.listdir(self.src_dirs['visual']):
            if not os.path.exists(self.dest_dirs['visual']+file):
                self.copyfile(self.src_dirs['visual']+file,self.dest_dirs['visual'])
                print(file)
        print("copying visual files done") 

    '''
    ------------------------------------------------------------------------------
    desc:      copy tfRecords to destination
    param:     -
    return:    -
    ------------------------------------------------------------------------------
    '''        
    def copy_records(self):
        print("..copying record files")
        for file in os.listdir(self.src_dirs['records']):
            if not os.path.exists(self.dest_dirs['records']+file):
                self.copyfile(self.src_dirs['records']+file,self.dest_dirs['records'])
                print(file)
        print("copying record files done") 
        
    '''
    ------------------------------------------------------------------------------
    desc:      remove audios in ".wav"-format and stft of audio from source
    param:     -
    return:    -
    ------------------------------------------------------------------------------
    '''        
    def remove_audio_speech(self):   
        print("remove wav files")
        for file in os.listdir(self.src_dirs['audio_speech_wav']):
            if not os.path.exists(self.dest_dirs['audio_speech_wav']+file):
                print("file "+ self.dest_dirs['audio_speech_wav']+file+" doesn't exist")
                self.copyfile(self.src_dirs['audio_speech_wav']+file,self.dest_dirs['audio_speech_wav'])
                if os.path.exists(self.dest_dirs['audio_speech_wav']+file):
                    os.remove(self.src_dirs['audio_speech_wav']+file)
            else:
                os.remove(self.src_dirs['audio_speech_wav']+file)
        print("remove wav files done")  
         
        print("remove npy files")
        for file in os.listdir(self.src_dirs['audio_speech']):
            if not os.path.exists(self.dest_dirs['audio_speech']+file):
                print("file "+ self.dest_dirs['audio_speech']+file+" doesn't exist")
                self.copyfile(self.src_dirs['audio_speech']+file,self.dest_dirs['audio_speech'])
                if os.path.exists(self.dest_dirs['audio_speech']+file):
                    os.remove(self.src_dirs['audio_speech']+file)
            else:
                os.remove(self.src_dirs['audio_speech']+file)
        print("remove npy files done")

    '''
    ------------------------------------------------------------------------------
    desc:      remove stft of mixed audios from source
    param:     -
    return:    -
    ------------------------------------------------------------------------------
    '''     
    def remove_audio_speech_mixed(self):
        print("remove mixed files")
        for file in os.listdir(self.src_dirs['audio_speech_mixed']):
            if not os.path.exists(self.dest_dirs['audio_speech_mixed']+file):
                print("file "+ self.dest_dirs['audio_speech_mixed']+file+" doesn't exist")
                self.copyfile(self.src_dirs['audio_speech_mixed']+file,self.dest_dirs['audio_speech_mixed'])
                if os.path.exists(self.dest_dirs['audio_speech_mixed']+file):
                    os.remove(self.src_dirs['audio_speech_mixed']+file)
            else:
                os.remove(self.src_dirs['audio_speech_mixed']+file)
        print("remove mixed files done")

    '''
    ------------------------------------------------------------------------------
    desc:      remove embeddings from source
    param:     -
    return:    -
    ------------------------------------------------------------------------------
    '''     
    def remove_visual(self):
        print("remove visual files")
        for file in os.listdir(self.src_dirs['visual']):
            if not os.path.exists(self.dest_dirs['visual']+file):
                print("file "+ self.dest_dirs['visual']+file+" doesn't exist")
                self.copyfile(self.src_dirs['visual']+file,self.dest_dirs['visual'])
                if os.path.exists(self.dest_dirs['visual']+file):
                    os.remove(self.src_dirs['visual']+file)
            else:
                os.remove(self.src_dirs['visual']+file)
        print("remove visual files done")

    '''
    ------------------------------------------------------------------------------
    desc:      remove tfRecords from source
    param:     -
    return:    -
    ------------------------------------------------------------------------------
    '''    
    def remove_records(self):
        print("remove record files")
        for file in os.listdir(self.src_dirs['records']):
            if not os.path.exists(self.dest_dirs['records']+file):
                print("file "+ self.dest_dirs['records']+file+" doesn't exist")
                self.copyfile(self.src_dirs['records']+file,self.dest_dirs['records'])
                if os.path.exists(self.dest_dirs['records']+file):
                    os.remove(self.src_dirs['records']+file)
            else:
                os.remove(self.src_dirs['records']+file)
        print("remove record files done")
                
