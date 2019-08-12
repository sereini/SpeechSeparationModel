# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 14:24:59 2019

@author: chalbeisen
This program is used for saving filenames of .wav files which are already downloaded into a numpy file
"""
import numpy as np
import os 
import re

#global variables

#audio_src defines directory where .wav files are saved
#audio_dest defines where numpy file should be saved
dirs = {'audio_src':['D:/Bachelorarbeit/audio/audio_speech/500000/','D:/Bachelorarbeit/audio/audio_speech/600000/'],'audio_dest':'D:/Bachelorarbeit/audio_download_500000_600000/'}#'D:/audio_speech/'}
audios = []

#get audios from different directories
for i in range(0, len(dirs['audio_src'])):
    for file in os.listdir(dirs['audio_src'][i]):
        if file.endswith(".wav"):
            audios.append(file)
audios.sort()


'''
------------------------------------------------------------------------------
desc:      save filenames of .wav files which are already downloaded into a numpy file
param:    -
return:    -
------------------------------------------------------------------------------
'''

def ids_to_np():
    idx = []
    for i in range(0, len(audios)):
        aud1 = os.path.splitext(audios[i])[0]
        idx1 = int(re.split(r'(^\d+)', aud1)[1])
        print(idx1)
        idx.append(idx1)
    idx.sort()
    np.save(dirs['audio_dest']+'ids_finished.npy', idx)
            
def main():
    ids_to_np()

if __name__== "__main__":
    main()