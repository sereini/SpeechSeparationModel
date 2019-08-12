# -*- coding: utf-8 -*-
"""
Checks if all folders from the youtube videos have 75 frames. 
You have to delete the printed folders.

@author: sscherrer
"""

import os
import re
import numpy as np
import shutil

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [atoi(c) for c in re.split(r'(\d+)', text)]

# define by yourself
path = 'D:\images_AVSpeech'
corrupt_folders = np.array([])

print("Check: ", path)

for p in os.listdir(path):
    if len(os.listdir(os.path.join(path, p))) < 75:
        print(p)
        corrupt_folders = np.append(corrupt_folders, p)
        try:
            shutil.rmtree(path + '/' + p )
        except Exception as e:
            print(e)
            print("Remove folder failed: ", p)
        else:
            print("Folder removed: ", p)
        print("-------------------------")
    elif len(os.listdir(os.path.join(path, p))) > 75:
        print(p)
        for d in os.listdir(os.path.join(path, p)):
            if "Copy" in d:
                try:
                    os.remove(path + '/' + p + '/' + d)
                except Exception as e:
                    print(e)
                    print("Remove file failed: ", d)
                else:
                    print("File removed: ", d)
        print("-------------------------")
    
    else:
        image_list = os.listdir(os.path.join(path, p))
        image_list.sort(key=natural_keys)
        for d in image_list:
            fsize = os.path.getsize(os.path.join(os.path.join(path, p),d))
            if fsize < 5:
                print(p)
                corrupt_folders = np.append(corrupt_folders, p)
                #np.save(os.path.join(os.getcwd(),'corrupt_folder.npy'), corrupt_folders)
                print("Failure: Filesize, please check!")
                print("-------------------------")
                break
