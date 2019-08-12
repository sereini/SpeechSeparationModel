# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 12:18:44 2019

@author: sscherrer

This program is used to predict the separated audio tracks from a given TFRecord (evaluate data).
The pretrained model is trained to separate one speaker from noisy background.

"""

import tensorflow as tf
import numpy as np
import os, re
import time
#import scipy.io.wavfile as wav
import scipy.signal
import matplotlib.pyplot as plt
from matplotlib import cm, colors
import librosa
import librosa.display

# global parameters
fs = 16000
nperseg = 324
noverlap = 162
nfft = 512
window = "hann"


'''
------------------------------------------------------------------------------
desc:      load pretrained model
param: 
   model:  dir of pretrained avspeech model

return:    -
------------------------------------------------------------------------------
'''
def load_model(model, input_map=None):
    # check if the model is a model directory (containing a metagraph and a checkpoint file)
    # or if it is a protobuf file with a frozen graph
    model_exp = os.path.expanduser(model)
    if (os.path.isfile(model_exp)):
        print('Model filename: %s' % model_exp)
        with tf.gfile.FastGFile(model_exp,'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, input_map=input_map, name='')
    else:
        print('Model directory: %s' % model_exp)
        meta_file, ckpt_file = get_model_filenames(model_exp)
        
        print('Metagraph file: %s' % meta_file)
        print('Checkpoint file: %s' % ckpt_file)
      
        saver = tf.train.import_meta_graph(os.path.join(model_exp, meta_file), input_map=input_map)
        saver.restore(tf.get_default_session(), os.path.join(model_exp, ckpt_file))
        
        

'''
------------------------------------------------------------------------------
desc:          load model files (4 files needed)
param: 
   model_dir:  dir of pretrained avspeech model

return:        pretrained model
------------------------------------------------------------------------------
'''
def get_model_filenames(model_dir):
    files = os.listdir(model_dir)
    meta_files = [s for s in files if s.endswith('.meta')]
    if len(meta_files)==0:
        raise ValueError('No meta file found in the model directory (%s)' % model_dir)
    elif len(meta_files)>1:
        raise ValueError('There should not be more than one meta file in the model directory (%s)' % model_dir)
    meta_file = meta_files[0]
    ckpt = tf.train.get_checkpoint_state(model_dir)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_file = os.path.basename(ckpt.model_checkpoint_path)
        return meta_file, ckpt_file

    meta_files = [s for s in files if '.ckpt' in s]
    max_step = -1
    for f in files:
        step_str = re.match(r'(^model-[\w\- ]+.ckpt-(\d+))', f)
        if step_str is not None and len(step_str.groups())>=2:
            step = int(step_str.groups()[1])
            if step > max_step:
                max_step = step
                ckpt_file = step_str.groups()[0]
    return meta_file, ckpt_file


'''
------------------------------------------------------------------------------
desc:      plot two spectrograms among each other for better comparison between
           label and prediction of the model (one colorbar per subplot)
param: 
   Zxx1:   complex spectrogram of the prediction
   Zxx2:   complex spectrogram of the label (orgiginal)  
   title:  title of the subplots as a list
   path:   path to save the figure
   
return:    -
------------------------------------------------------------------------------
'''
def plot_spec(Zxx1, Zxx2, title, path):
    Zxx1_dB = librosa.amplitude_to_db(np.abs(Zxx1))
    Zxx2_dB = librosa.amplitude_to_db(np.abs(Zxx2))
    
    plt.figure()    
    plt.subplots_adjust(hspace=0.5)
    plt.subplot(211)
    plt.title(title[0])
    librosa.display.specshow(Zxx1_dB, sr=fs, hop_length=nperseg-noverlap, x_axis='time', y_axis='linear',cmap=cm.nipy_spectral)
    plt.colorbar(format='%2.0f db')
    plt.subplot(212)
    plt.title(title[1])
    librosa.display.specshow(Zxx2_dB, sr=fs, hop_length=nperseg-noverlap, x_axis='time', y_axis='linear',cmap=cm.nipy_spectral)
    plt.colorbar(format='%2.0f db')
    
    plt.savefig(path)
    plt.close()
    
    

'''
------------------------------------------------------------------------------
desc:      plot two spectrograms among each other for better comparison between
           label and prediction of the model (one colorbar for both subplots)
param: 
   Zxx1:   complex spectrogram of the prediction
   Zxx2:   complex spectrogram of the label (original)  
   title:  title of the subplots as a list
   path:   path to save the figure

return:    -
------------------------------------------------------------------------------
'''
def plot_spec2(Zxx1, Zxx2, title, path):
    Zxx1_dB = librosa.amplitude_to_db(np.abs(Zxx1))
    Zxx2_dB = librosa.amplitude_to_db(np.abs(Zxx2))
   
    dt, df = 3.01/(Zxx1_dB.shape[1]+1), 8050/(Zxx1_dB.shape[0]+1)
    
    f, t = np.mgrid[slice(0, 8050 , df),
                slice(0, 3.01, dt)]
    
    Zxx_min, Zxx_max = np.minimum(np.min(Zxx1_dB),np.min(Zxx2_dB)), np.maximum(np.max(Zxx1_dB),np.max(Zxx2_dB))
    
    cm_col = cm.nipy_spectral
    norm = colors.Normalize(vmin=Zxx_min, vmax=Zxx_max)
    cmap = cm.ScalarMappable(norm=norm, cmap=cm_col)
    cmap.set_array([])
    
    fig, (ax1, ax2) = plt.subplots(2,1)
    ax1.set_title(title[0])
    ax1.pcolormesh(t, f, Zxx1_dB, cmap=cm_col, vmin=Zxx_min, vmax=Zxx_max)
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Hz')
    
    ax2.set_title(title[1])
    ax2.pcolormesh(t, f, Zxx2_dB, cmap=cm_col, vmin=Zxx_min, vmax=Zxx_max)
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Hz')
    
    fig.colorbar(cmap, ax=[ax1,ax2], format='%2.0f db')
    fig.subplots_adjust(hspace=0.5, right=0.75)
    
    fig.savefig(path)
    plt.close()
    
    

'''
------------------------------------------------------------------------------
desc:      plot spectrogram of the mixed audiosignal
param: 
   Zxx1:   complex spectrogram of mixed signal
   title:  title of the figure
   path:   path to save the figure
   
return:    -
------------------------------------------------------------------------------
'''
def plot_spec_mix(Zxx, title, path):
    Zxx_dB = librosa.amplitude_to_db(np.abs(Zxx))
     
    # same size as subplots, for better comparison
    plt.figure(figsize=(6.2, 2.4))
    plt.title(title)
    librosa.display.specshow(Zxx_dB, sr=fs, hop_length=nperseg-noverlap, x_axis='time', y_axis='linear',cmap=cm.nipy_spectral)
    plt.colorbar(format='%2.0f db')
    plt.tight_layout()
    plt.subplots_adjust(right=0.9)
    plt.savefig(path)
    plt.close()
    
    
'''
------------------------------------------------------------------------------
desc:      plot two istft among each other for better comparison between
           label and prediction of the model
param: 
   t:      time array of the prediction
   x:      amplitude array of the prediction
   tl:     time array of the label
   xl:     amplitude array of the label
   title:  title of the subplots as a list
   path:   path to save the figure
   
return:    -
------------------------------------------------------------------------------
''' 
def plot_istft(t, x, tl, xl, title, path):
    plt.figure()
    plt.subplots_adjust(hspace=0.5)
    plt.subplot(211)        
    plt.plot(t,x)     
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.title(title[0])
    plt.tight_layout()
    plt.subplot(212)
    plt.plot(tl, xl)
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.title(title[1])
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    
    

'''
------------------------------------------------------------------------------
desc:      inverse power compression for inverse stft
param: 
   Zxx:    complex spectrogram (stft)

return:   
   Zxx:   stft repower compressed
------------------------------------------------------------------------------
'''
def powerTransform(Zxx):
    R = abs(Zxx)
    phi = np.angle(Zxx)
    Zxx = R**(1/0.3) * np.exp(1j*phi)
    return Zxx



'''
------------------------------------------------------------------------------
desc:      main function
param: 
   model_dir:  dir of the saved checkpoint (4 files)
                   - checkpoint
                   - model.ckpt-xx.data-00000-of-00001
                   - model.ckpt-xx.index
                   - model.ckpt-xx.meta
   data_dir:   dir of the TFRecords as a list
   
return:        -
------------------------------------------------------------------------------
'''
def main(model_dir, data_dir):
    
    # read TFRecords
    test_record = []
    for t in data_dir:
        for dirs,_,files in os.walk(t):
            for f in files:
                test_record.append(os.path.abspath(os.path.join(dirs, f)))
    for test_rec in test_record:
        record_iterator = tf.python_io.tf_record_iterator(path=test_rec)
        eval_train = test_rec.split("\\")[-2] + "/"
        test_rec = eval_train + (test_rec.split("\\")[-1]).split(".")[0]+"/"
        print("Data: ", test_rec)
        
        dir_save = os.path.join(model_dir, test_rec)
        
        dirs = ["img", "audio"]
        for d in dirs:
            if not os.path.exists(os.path.join(dir_save, d)):
                os.makedirs(os.path.join(dir_save, d))
        
        for string_record in record_iterator:
            example = tf.train.Example()
            example.ParseFromString(string_record)
            a_s1 = int(example.features.feature['audio_s1'].int64_list.value[0])
            a_s2 = int(example.features.feature['audio_s2'].int64_list.value[0])
            a_s3 = int(example.features.feature['audio_s3'].int64_list.value[0])
            
            v_s1 = int(example.features.feature['video_s1'].int64_list.value[0])
            v_s2 = int(example.features.feature['video_s2'].int64_list.value[0])
            v_s3 = int(example.features.feature['video_s3'].int64_list.value[0])
            
            l_s1 = int(example.features.feature['label_s1'].int64_list.value[0])
            l_s2 = int(example.features.feature['label_s2'].int64_list.value[0])
            l_s3 = int(example.features.feature['label_s3'].int64_list.value[0])
            l_s4 = int(example.features.feature['label_s4'].int64_list.value[0])
            
            a = (example.features.feature['audio'].float_list.value[:])
            v = (example.features.feature['video'].float_list.value[:])
            l = (example.features.feature['label'].float_list.value[:])
            
            # reshape back to their original shape from a 1D array read from tfrecords
            pred_audio = np.array(a).reshape((a_s1, a_s2, a_s3))
            pred_video = np.array(v).reshape((v_s1, v_s2, v_s3))
            pred_label = np.array(l).reshape((l_s1, l_s2, l_s3, l_s4))
        
        
        # start session
        with tf.Graph().as_default():
            with tf.Session() as sess:
    
                # load the model
                load_model(model_dir)
                
                # Get input and output tensors of pretrained model
                # inputs
                vid1 = tf.get_default_graph().get_tensor_by_name("vid1:0")
                aud_mix = tf.get_default_graph().get_tensor_by_name("aud_mix:0")
                # output
                prediction_masks = tf.get_default_graph().get_tensor_by_name("prediction_masks:0")
                
    
                # run forward pass to calculate masks
                start_time = time.time()
                feed_dict1 = {vid1:  np.reshape(pred_video[0,:,:],[-1,pred_video.shape[1],pred_video.shape[2]]),
                             aud_mix: np.reshape(pred_audio,[-1,pred_audio.shape[0],pred_audio.shape[1],pred_audio.shape[2]])}
                mask1 = sess.run(prediction_masks, feed_dict=feed_dict1)
                
                feed_dict2 = {vid1:  np.reshape(pred_video[1,:,:],[-1,pred_video.shape[1],pred_video.shape[2]]),
                             aud_mix: np.reshape(pred_audio,[-1,pred_audio.shape[0],pred_audio.shape[1],pred_audio.shape[2]])}
                mask2 = sess.run(prediction_masks, feed_dict=feed_dict2)
                    
                # calculate complex mask from splitted output (real and imaginary parts are stacked)
                mask1 = np.reshape(mask1,[mask1.shape[1],mask1.shape[2],mask1.shape[3]])
                mask1 = np.vectorize(complex)(mask1[:,:,0],mask1[:,:,1])
                
                mask2 = np.reshape(mask2,[mask2.shape[1],mask2.shape[2],mask2.shape[3]])
                mask2 = np.vectorize(complex)(mask2[:,:,0],mask2[:,:,1])
                             
                # complex spectrograms (real and imaginary parts are stacked)
                aud1 = np.reshape(np.vectorize(complex)(pred_label[0,:,:,0],pred_label[0,:,:,1]),[257,298])
                aud2 = np.reshape(np.vectorize(complex)(pred_label[1,:,:,0],pred_label[1,:,:,1]),[257,298])              
                aud_mix = np.reshape(np.vectorize(complex)(pred_audio[:,:,0],pred_audio[:,:,1]),[257,298])
                
                # calculate predictions
                pred1 = np.multiply(mask1, aud_mix)
                pred2 = np.multiply(mask2, aud_mix)
                
                # calculate original masks
                mask1_muster = np.divide(aud1,aud_mix)
                mask2_muster = np.divide(aud2,aud_mix)
                
                # plot spectrograms
                plot_spec(mask1, mask1_muster,['prediction mask1','original mask1'], dir_save + 'img/spec_mask1_col.png')
                plot_spec(mask2, mask2_muster,['prediction mask2','original mask2'], dir_save + 'img/spec_mask2_col.png')
                plot_spec2(mask1, mask1_muster,['prediction mask1','original mask1'], dir_save + 'img/spec_mask1.png')
                plot_spec2(mask2, mask2_muster,['prediction mask2','original mask2'], dir_save + 'img/spec_mask2.png')
               
                plot_spec2(pred1, aud1, ['prediction speaker1','original speaker1'], dir_save + 'img/spec_aud1.png')
                plot_spec2(pred2, aud2, ['prediction speaker2','original speaker2'], dir_save + 'img/spec_aud2.png')
                plot_spec_mix(aud_mix, 'spectro mixed', dir_save +'img/spec_mix.png')
    
                # calculate inverse stft
                t1, x1 = scipy.signal.istft(powerTransform(pred1), fs=fs, window=window, nperseg=nperseg, noverlap=noverlap, nfft=nfft)
                t2, x2 = scipy.signal.istft(powerTransform(pred2), fs=fs, window=window, nperseg=nperseg, noverlap=noverlap, nfft=nfft)
                tl1, xl1 = scipy.signal.istft(powerTransform(aud1), fs=fs, window=window, nperseg=nperseg, noverlap=noverlap, nfft=nfft)
                tl2, xl2 = scipy.signal.istft(powerTransform(aud2), fs=fs, window=window, nperseg=nperseg, noverlap=noverlap, nfft=nfft)
                taud_mix, xaud_mix = scipy.signal.istft(powerTransform(aud_mix), fs=fs, window=window, nperseg=nperseg, noverlap=noverlap, nfft=nfft)
                # plot istft
                plot_istft(t1, x1, tl1, xl1, ['prediction speaker1', 'original speaker1'], dir_save + 'img/aud1_istft.png')
                plot_istft(t2, x2, tl2, xl2, ['prediction speaker2', 'original speaker2'], dir_save + 'img/aud2_istft.png')
    
                # save istft as wav-file                                
                scipy.io.wavfile.write(dir_save + "audio/pred1.wav",fs,np.int16(x1))
                scipy.io.wavfile.write(dir_save + "audio/pred2.wav",fs,np.int16(x2))
                scipy.io.wavfile.write(dir_save + "audio/aud1.wav",fs,np.int16(xl1))
                scipy.io.wavfile.write(dir_save + "audio/aud2.wav",fs,np.int16(xl2))
                scipy.io.wavfile.write(dir_save + "audio/aud_mix.wav",fs,np.int16(xaud_mix))
                
    
                # calculate loss
                print("diff1: ",np.mean(np.square(np.abs(pred1-aud1))))
                print("diff1: ",np.mean(np.square(np.abs(pred2-aud2))))
                
                run_time = time.time() - start_time
                print('Run time: ', run_time)
                print("-------------------------------------------------------------------------")


model_dir = "./model/model_AVSpeech_oneSpeaker_007/"
data_dir = ["./data_train/","./data_eval/"]

main(model_dir, data_dir)