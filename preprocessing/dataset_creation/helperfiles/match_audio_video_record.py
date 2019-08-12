# -*- coding: utf-8 -*-
"""
Created on Thu May  2 07:50:48 2019

@author: chalbeisen

This program is used to match generated embeddings with audio, to mix the matched audio files and 
put data into tfRecords 
"""
import os
import numpy as np
import contextlib
import wave
from LookingToListen_Audio_clean import Audio
import re
import pandas as pd 
import sys
import argparse
import tensorflow as tf
import scipy.signal
import threading
import shutil
from shutil import copy
from pathlib import Path
from copy_files import CopyFiles

#global variables
src_dirs =  dest_dirs = audio = data_dir = threads = copyFile = lock = None
'''
------------------------------------------------------------------------------
desc:      check duration of audio file
param:    
    aud_filename:    filename of audio to check duration
return:    True if duration = 3s, False otherwise 
------------------------------------------------------------------------------
'''
def check_duration(aud_filename):
    try:
        with contextlib.closing(wave.open(aud_filename,'r')) as f:
            frames = f.getnframes()
            rate = f.getframerate()
            duration = frames / float(rate)
            #print("duration: ", duration)
            f.close()
            
            if duration != 3:
                #print('duration != 3s')
                return False
            else:
                return True
    except:
        return False
    
'''
------------------------------------------------------------------------------
desc:      check duration of audio file
param:    
    aud_filename:    filename of audio to check filesize
return:    True if size > 1000 bytes, False otherwise 
------------------------------------------------------------------------------
'''
def check_filesize(aud_filename) :
        size = os.stat(aud_filename).st_size
        if size < 1000:
            #print('size < 1000')
            return False
        else:
            return True

'''
------------------------------------------------------------------------------
desc:      redownload audio in av_speech dataset
param:    
    i:    index of audio in av_speech dataset
    audio:    instance of Audio object 
    aud_filename:    filename of audio 
return:    True if size > 1000 bytes, False otherwise 
------------------------------------------------------------------------------
'''
def redownload(i, audio, aud_filename) :
	if os.path.exists(aud_filename):
		os.remove(aud_filename)
	audio.download_speech(src_dirs['audio'], i, i+1, wait=True)
	
	if os.path.exists(aud_filename) and check_duration(aud_filename) == True and check_filesize(aud_filename) == True:
		return True
	else:
		print('unsuccessful download')
		return False

'''
------------------------------------------------------------------------------
desc:      save existing records into an array to check later if record already exists
param:     -
return:    -
------------------------------------------------------------------------------
'''
def ids_records_to_np():
    records = []
    for file in os.listdir(dest_dirs_copy['records']):
        if file.endswith(".tfrecord"):
            print(file)
            records.append(file)
    path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'ids_records_finished.npy')
    np.save(path, records)
    
'''
------------------------------------------------------------------------------
desc:      match generated embeddings with audio files which contain the ids of av_speech dataset between start and stop
param:     
    start:      start id from av_speech dataset
    stop:      stop id from av_speech dataset      
return:    -
------------------------------------------------------------------------------
'''       
def match_audio_video(start, stop):
    print("match audio")
    print(src_dirs['visual'])
    for root, subdirs, _ in os.walk(src_dirs['visual']):
        for subdir in subdirs:
            idx = int(re.split(r'(\d)',subdir)[1])*1000000+int(re.split(r'(\d{2,3})',subdir)[1])*1000
            print(idx)
            if(idx> start and idx<=stop):
                print("chosen index: ",idx)
                embeddings = os.listdir(root+subdir)  
                embeddings.sort()
                for i in range(0, len(embeddings)-1):
                    emb = os.path.splitext(embeddings[i])[0]
                    idx = int(re.split(r'(^\d+)', emb)[1])
                    aud_amnt = 100000
                    aud_delta = (int(idx / aud_amnt) + 1)*aud_amnt
                    audio_speech_fn = src_dirs['audio']+str(aud_delta)+'/'+emb+'.wav'
                    emb_fn = src_dirs['visual']+subdir+'/'+emb+'.npy'
                    #print(audio_speech_fn)
                    #print(emb_fn)
                   
                    if not os.path.exists(dest_dirs['audio_speech_wav']+emb+'.wav'):
                        if os.path.exists(audio_speech_fn) == False:
                            print("path "+emb)
                        elif check_duration(audio_speech_fn) == False:
                            print("duration "+emb)
                        elif check_filesize(audio_speech_fn) == False:
                            print("filesize "+emb)
                        else:
                           print('file '+emb+' copied') 
                           copy(audio_speech_fn,dest_dirs['audio_speech_wav'])
                           copy(emb_fn,dest_dirs['visual']+emb+'.npy')
                    else:
                        print('file '+str(i)+' already exists')

'''
------------------------------------------------------------------------------
desc:      check match of generated embeddings with audio files which contain the ids of av_speech dataset between start and stop
param:     
    start:      start id from av_speech dataset
    stop:      stop id from av_speech dataset      
return:    -
------------------------------------------------------------------------------
'''
def check_match(start, stop):
    embeddings = os.listdir(dest_dirs['visual'])  
    embeddings.sort()
    for i in range(start, stop):
       emb = os.path.splitext(embeddings[i])[0]
       idx = int(re.split(r'(^\d+)', emb)[1])
       if idx<start or idx>stop:
           continue
       audio_speech_fn = dest_dirs['audio_speech_wav']+emb+'.wav'
       emb_fn = dest_dirs['visual']+emb+'.npy'
       
       print("test match audio video: ", i)
       if not os.path.exists(audio_speech_fn) and os.path.exists(emb_fn):
           print("Audio nicht vorhanden: ",i)
           os.remove(emb_fn)
           
       if not os.path.exists(emb_fn) and os.path.exists(audio_speech_fn):
           print("Video nicht vorhanden: ",i)
           os.remove(audio_speech_fn)
           
'''
------------------------------------------------------------------------------
desc:      mix audio from index of av_speech dataset between start and stop, 
           mix always index i with index i+1
param:     
    start:      start id from av_speech dataset
    stop:      stop id from av_speech dataset      
return:    -
------------------------------------------------------------------------------
'''
def mix_audio(start, stop):
    audios = os.listdir(dest_dirs['audio_speech_wav'])
    if stop == len(audios):
        stop -=1
    for i in range(start, stop):
        aud1 = os.path.splitext(audios[i])[0]
        aud2 = os.path.splitext(audios[i+1])[0]
        #aud3 = os.path.splitext(audios[i+2])[0]
        idx1 = int(re.split(r'(^\d+)', aud1)[1])
        idx2 = int(re.split(r'(^\d+)', aud2)[1])
        print(idx1)
        print(idx2)
        audio.mix_audio(idx1, idx2, dest_dirs['audio_speech_wav'], dest_dirs['audio_speech_wav'], dest_dirs['audio_speech_mixed'], dest_dirs['audio_speech'], dest_dirs['audio_speech'], label="")


'''
------------------------------------------------------------------------------
desc:  convert a float / double value to a type compatible with tf.Example 
param:     
    value:      float / double value     
return:    float_list from a float / double
------------------------------------------------------------------------------
'''
def _float_feature(value):
    """Wrapper for inserting float features into Example proto."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

'''
------------------------------------------------------------------------------
desc:  convert a bool / enum / int / uint value to a type compatible with tf.Example 
param:     
    value:      bool / enum / int / uint value     
return:    int64_list from a bool / enum / int / uint
------------------------------------------------------------------------------
'''
def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

'''
------------------------------------------------------------------------------
desc:      write tfRecord. use data with the index from av_speech data set (between start and stop) which are contained in mixed audios,
           test if there is enough space on disc, otherwise move files to destination and create remaining records
param:     
    start:      start id from av_speech dataset
    stop:      stop id from av_speech dataset      
return:    
------------------------------------------------------------------------------
'''
def writeTFRecord(start, stop):
    p_audio = os.path.join(data_dir, dest_dirs['audio_speech_mixed']) #mixed audio
    f_audio = os.listdir(p_audio)
    for i in range(start, stop):  
        total, used, free = shutil.disk_usage("D:/")
        free = free / 1024
        if free > 4000:
            print(i)
            aud = os.path.splitext(f_audio[i])[0]
            print(aud)
            filename = dest_dirs['records'] + aud + ".tfrecord"
            
            if(os.path.exists(filename)):
                print("record " + aud + " already exists")
                continue 
            if Path('./helperfiles/ids_records_finished.npy').is_file():
                finished = np.load('./helperfiles/ids_records_finished.npy')
                fn = aud + ".tfrecord"
                if fn in finished:
                    print("file "+fn+" already exists")
                    continue
            else:
                print("match_audio_video_record.py doesn't exist")
            
            idx1 = int(re.split(r'(\d+)', aud)[1])
            idx2 = int(re.split(r'(\d+)', aud)[3])
            
            aud1 = dest_dirs['audio_speech']+str(idx1)+'_'+audio.yt_id[idx1]+'.npy'
            aud2 = dest_dirs['audio_speech']+str(idx2)+'_'+audio.yt_id[idx2]+'.npy'
            vid1 = dest_dirs['visual']+str(idx1)+'_'+audio.yt_id[idx1]+'.npy'
            vid2 = dest_dirs['visual']+str(idx2)+'_'+audio.yt_id[idx2]+'.npy'
            
            if not os.path.exists(aud1) or not os.path.exists(aud2) or not os.path.exists(vid1) or not os.path.exists(vid2):
                continue
     
            data_audio = np.array(np.load(os.path.join(p_audio, f_audio[i])))
            data_audio = np.stack((data_audio.real,data_audio.imag), -1)
            data_video = np.array((np.load(os.path.join(data_dir, vid1)), np.load(os.path.join(data_dir, vid2))))
            data_labels = np.array((np.load(os.path.join(data_dir, aud1)), np.load(os.path.join(data_dir, aud2))))
            data_labels = np.stack((data_labels.real,data_labels.imag), -1)
            
            
            
            writer = tf.python_io.TFRecordWriter(filename)
            a_s1 = data_audio.shape[0]
            a_s2 = data_audio.shape[1]
            a_s3 = data_audio.shape[2]
    
            v_s1 = data_video.shape[0]
            v_s2 = data_video.shape[1]
            v_s3 = data_video.shape[2]
            
            #shape from labels --> from video and audio
            a = data_audio.flatten()
            v = data_video.flatten()
            l = data_labels.flatten()
            
            example = tf.train.Example(features=tf.train.Features(feature={
                    'audio_s1': _int64_feature(a_s1),
                    'audio_s2': _int64_feature(a_s2),
                    'audio_s3': _int64_feature(a_s3),
                    'video_s1': _int64_feature(v_s1),
                    'video_s2': _int64_feature(v_s2),
                    'video_s3': _int64_feature(v_s3),
                    'label_s1': _int64_feature(v_s1),
                    'label_s2': _int64_feature(a_s1),
                    'label_s3': _int64_feature(a_s2),
                    'label_s4': _int64_feature(a_s3),
                    'audio': _float_feature(a.tolist()),
                    'video': _float_feature(v.tolist()),
                    'label': _float_feature(l.tolist())
                    }))
    
            writer.write(example.SerializeToString())
            writer.close() 
        else:
            lock.acquire()
            print("disc is full")
            copyFile.copy_records()
            copyFile.wait_for_subprocesses()
            print("done copying records")
            copyFile.remove_records()
            ids_records_to_np()
            lock.release()

'''
------------------------------------------------------------------------------
desc:      get parameters from script "Run-match_audio_video_record.ps1"
param:    
    argv:    arguments from script "Run-match_audio_video_record.ps1" 
return:    parsed arguments
------------------------------------------------------------------------------
'''       
def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--emb_dir', type=str,
        help='Define directory of embeddings', default='D:/cnn_data2/visual/')
    parser.add_argument('--aud_dir', type=str,
        help='Define directory of audios', default='D:/audio_speech/')
    parser.add_argument('--dest_dir', type=str,
        help='Define destination directory of files', default='D:/match_data/')
    parser.add_argument('--dest_dir_copy', type=str,
        help='Define destination directory for copying', default='D:/audio_speech/')
    parser.add_argument('--start', type=int,
        help='Define id where you want to start the match', default=0)
    parser.add_argument('--stop', type=int,
        help='Define id where you want to stop the match', default=100000)
    
    return parser.parse_args(argv)


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
desc:      start a multiple threads to execute the target function and wait for them to complete
param:    
    th_amnt:    amount of threads to execute target function
    f_amnt:    amount of files which need to be divided between threads
    target:    target function
return:    -
------------------------------------------------------------------------------
'''            
def start_threads(th_amnt, f_amnt , target):
    start = 0
    delta = int(f_amnt/th_amnt)
    threads = []
    
    for i in range(0,th_amnt):
        start = delta*i
        if i == (th_amnt-1):
            stop = f_amnt
        else:
            stop = start + delta
        print("target: ", target)
        print("start: ", start)
        print("stop: ", stop)
        thread = threading.Thread(target = target, args = (start, stop))
        threads.append(thread)
        
    for thread in threads:
        thread.start()
        
    for thread in threads:  
        thread.join()

'''
------------------------------------------------------------------------------
desc:      match audios with embeddings, check match, mix audios and create tfRecords
param:    
    argv:      arguments from script "Run-match_audio_video_record.ps1" 
return:    -
------------------------------------------------------------------------------
'''        
def main(argv):  
    global src_dirs, dest_dirs, data_dir, dest_dirs_copy, audio, threads, copyFile, lock
    src_dirs = {'audio':argv.aud_dir, 'visual':argv.emb_dir}
    dirs = {'audio_speech':'audio/audio_speech/','audio_speech_wav':'audio/audio_speech/wav_files/','audio_noise':'audio/audio_noise','audio_noise_wav':'audio/audio_noise/wav_files/','audio_speech_mixed':'audio/audio_speech_mixed/', 'audio_speech_noise':'audio/audio_speech_noise/', 'visual':'visual/', 'records':'records/'}
    data_dir = argv.dest_dir
    data_dir_copy = argv.dest_dir_copy
    dest_dirs = {key: data_dir+value for (key, value) in dirs.items()}
    dest_dirs_copy = {key: data_dir_copy+value for (key, value) in dirs.items()}
    copyFile = CopyFiles(dest_dirs, dest_dirs_copy)
    create_directories(dest_dirs)
    create_directories(dest_dirs_copy)
    
    audio = Audio()
    lock = threading.Lock()
    th_amnt = 4
   
    match_audio_video(argv.start,argv.stop)
   
    embeddings = os.listdir(dest_dirs['visual']) 
    start_threads(th_amnt, len(embeddings), check_match)
   
    audios = os.listdir(dest_dirs['audio_speech_wav'])
    start_threads(th_amnt, len(audios), mix_audio)
    
    p_audio = os.path.join(data_dir, dest_dirs['audio_speech_mixed']) #mixed audio
    f_audio = os.listdir(p_audio)
    start_threads(th_amnt, len(f_audio), writeTFRecord)
    
    print("done matching files")

if __name__== "__main__":
    main(parse_arguments(sys.argv[1:]))