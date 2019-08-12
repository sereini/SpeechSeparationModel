# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 14:24:59 2019

@author: chalbeisen

This program is used for saving filenames of .tfrecord files which are already downloaded into a numpy file
"""
import numpy as np
import os 

#global variable
#audio_src defines directory where .tfrecord files are saved
#audio_dest defines where numpy file should be saved
dirs = {'records_src':['//pcklz101/F$/BA_LipRead/records/'],'records_dest':'D:/helperfiles/'}

'''
------------------------------------------------------------------------------
desc:      save filenames of .tfrecord files which are already downloaded into a numpy file
param:    -
return:    -
------------------------------------------------------------------------------
'''
def ids_records_to_np():
    records = []
    for i in range(0, len(dirs['records_src'])):
        for file in os.listdir(dirs['records_src'][i]):
            if file.endswith(".tfrecord"):
                print(file)
                records.append(file)
    np.save(dirs['records_dest']+'ids_records_finished.npy', records)
            
def main():
    ids_records_to_np()

if __name__== "__main__":
    main()