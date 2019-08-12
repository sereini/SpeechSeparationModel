"""
Created on Wed Mar 20 15:41:04 2019

@author: chalbeisen

This program is used to download audio, mix audios and to visualize the results
"""
import pandas as pd
import scipy.io.wavfile as wav
import scipy.signal 
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import librosa
import librosa.display
import wave
import pafy
import re
from pathlib import Path
import os
import RunCMD
import time
import scipy.io.wavfile
from matplotlib import cm
from matplotlib.ticker import FormatStrFormatter


class Audio:
    '''
    ------------------------------------------------------------------------------
    desc:      init Audio class and set global parameters
    param:    -
    return:    -
    ------------------------------------------------------------------------------
    '''
    def __init__(self):
        if Path('./helperfiles/avspeech_train.csv').is_file():
            self.av_train = pd.read_csv('./helperfiles/avspeech_train.csv', delimiter=',', header = None)
        elif Path('avspeech_train.csv').is_file():
            self.av_train = pd.read_csv('avspeech_train.csv', delimiter=',', header = None)
            
        if Path('./helperfiles/ids_finished.npy').is_file():
            self.idx = np.load('./helperfiles/ids_finished.npy')
        elif Path('ids_finished.npy').is_file():
            self.idx = np.load('./helperfiles/ids_finished.npy')
        else:
            self.idx = []

        self.yt_id = self.av_train.iloc[:,0]
        self.yt_id_bt = []
        self.start_bt = []
        self.stop_bt = []
        self.start = self.av_train.iloc[:,1]
        self.stop = self.av_train.iloc[:,2]
        self.duration = 3
        self.starttime = time.time()
        
        #stft
        self.fs = 16000
        self.nfft=512
        self.nperseg=324
        self.noverlap=162
        self.window="hann"
        
        #noise
        self.yt_id_bt = []
        self.start_bt = []
        self.stop_bt = []
         
    '''
    ------------------------------------------------------------------------------
    desc:      read audio file 
    param:    
        filename:      audio filename to read
    return:    audio data of left channel
    ------------------------------------------------------------------------------
    '''
    def read_audio(self, filename):

        #check if mono or stereo
        wave_file = wave.open(filename,'r')
        nchannels = wave_file.getnchannels()          
        
        #read audio file
        fs, audio = wav.read(filename)
        
        if(nchannels == 2):
            return audio[:,0]
        else:
            return audio
        
    '''
    ------------------------------------------------------------------------------
    desc:      download audio in a new Thread
    param:    
        i:      id of audio in av_speech dataset
        dest_dir:   destination of audio file
        wait:   wait for other audio download Threads to complete before starting audio download
        label:   (optional) to download a noise, set required label
    return:    -
    ------------------------------------------------------------------------------
    '''       
    def download_audio(self, i, dest_dir, wait=False, label="") :
        if label == "":
            filename = dest_dir+str(i)+'_'+self.yt_id[i]+'.wav'
            yt_id = self.yt_id[i]
            start = self.start[i]
        else:
            filename = dest_dir+label+'_'+self.yt_id_bt[i]+'.wav'
            yt_id = self.yt_id_bt[i]
            start = self.start_bt[i]
            
        if Path(filename).is_file():
            print("File "+filename+" exisitiert bereits") 
        elif len(self.idx) > 0 and i in self.idx:
            print("index "+str(i)+" exisitiert bereits") 
        else:
            if not os.path.exists(dest_dir):
                os.makedirs(dest_dir)
            url = "https://www.youtube.com/watch?v=" + yt_id
            print(url)
            vid = pafy.new(url)
            bestaud = vid.getbest()
            cmd_string = 'ffmpeg -i '+'"'+bestaud.url+'" '+' -ss  ' + str(start) + ' -t ' + str(self.duration)+' -vcodec libx264 '+'-ar '+str(self.fs)+' '+filename
            RunCMD.RunCmd(cmd_string,200,5,wait).Run()
        
    '''
    ------------------------------------------------------------------------------
    desc:      download audio from av_speech dataset from start id to stop id
    param:    
        dest_dir:      destination of audio files
        start:      start id of audio in av_speech dataset
        stop:      stop id of audio in av_speech dataset
        split_audio:      split audio into 100000 directory segments
        wait:      wait for other audio download Threads to complete before starting audio download
    return:    -
    ------------------------------------------------------------------------------
    '''
    def download_speech(self, dest_dir, start, stop, split_audio=False, wait=False):
        i=start
        if split_audio==True:
            aud_amnt = 100000
            aud_delta = (int(i / aud_amnt) + 1)*aud_amnt
            dest_dir = dest_dir+str(aud_delta)+'/'
        while i<=stop:                    
            try:
                print(i)
                print("dir: ",  dest_dir)
                print("start: ", start)
                print("stop: ", stop)
                self.download_audio(i, dest_dir, wait)
            except IOError:         # user or video not available
                print("IOError")
                print("i: ",i)
                print("yt-id: ",self.yt_id[i])
            except KeyError:         # user or video not available
                print("KeyError")
                print("i: ",i)
                print("yt-id: ",self.yt_id[i])
            i+=1

    '''
    ------------------------------------------------------------------------------
    desc:      download noise-audio from audioSet dataset from start id to stop id
    param:    
        dest_dir:      destination of audio files
        label:      required class of noise audio
        start:      start id of audio in av_speech dataset
        stop:      stop id of audio in av_speech dataset
        wait:      wait for other audio download Threads to complete before starting audio download
    return:    -
    ------------------------------------------------------------------------------
    '''       
    def download_noise(self, dest_dir, label, start, stop, wait=False):
        #get youtube id from balanced_train_segments
        self.get_ytIDs_aud(label)
        i=start
        while i<=stop:
            try:
                self.download_audio(i, dest_dir, wait, label)
            except IOError:         # user or video not available
                print("IOError")
            except KeyError:         # user or video not available
                print("KeyError")
            i+=1
    '''
    ------------------------------------------------------------------------------
    desc:      search audios in audioSet with the required class
    param:    
        labelstr:      required class
    return:    -
    ------------------------------------------------------------------------------
    '''        
    def get_ytIDs_aud(self,labelstr):
        label = self.get_label(labelstr)
        if Path('./balanced_train_segments.csv').is_file():
            balanced_train = 'balanced_train_segments.csv'
        elif Path('./helperfiles/balanced_train_segments.csv').is_file():
            balanced_train = './helperfiles/balanced_train_segments.csv'
        else:
            print("balanced_train_segments.csv doesn't exist")
            return 
        
        with open(balanced_train) as file:
            for actline in file:
                actline = actline.split()
                if(label in actline[3]):
                    self.yt_id_bt.append(re.sub(",|\"", "", actline[0]))
                    self.start_bt.append(re.sub(",", "", actline[1]))
                    self.stop_bt.append(re.sub(",", "", actline[2]))

    '''
    ------------------------------------------------------------------------------
    desc: map required class to label     
    param:    
        labelstr:      required class
    return:    label for the required class
    ------------------------------------------------------------------------------
    '''                  
    def get_label(self,labelstr):
        if Path('./class_labels_indices.csv').is_file():
            class_labels = 'class_labels_indices.csv'
        elif Path('./helperfiles/class_labels_indices.csv').is_file():
            class_labels = './helperfiles/class_labels_indices.csv'
        else:
            print("class_labels_indices.csv doesn't exist")
            return 
        
        with open(class_labels) as file:
            for actline in file:
                actline = actline.split(",")
                if(labelstr in actline[2]):
                    return actline[1]
    '''
    ------------------------------------------------------------------------------
    desc:      plot spectrogramm of audio 
    param:    
        fn_Zxx:      filename of stft of audio file
        fn_fig:      filename for spectrogram 
    return:    -
    ------------------------------------------------------------------------------
    ''' 
    
    def plot_spectrogram(self,fn_Zxx,fn_fig=''):
        plt.figure()
        Zxx = np.load(fn_Zxx)
        Zxx_dB = librosa.amplitude_to_db(np.abs(Zxx))
        librosa.display.specshow(Zxx_dB, sr=self.fs, hop_length= self.nperseg-self.noverlap, x_axis='time', y_axis='linear', cmap = cm.nipy_spectral)
        plt.colorbar(format='%2.0f db')
        if fn_fig != '':
            plt.savefig(fn_fig)

    '''
    ------------------------------------------------------------------------------
    desc:      plot the amplitude envelope of a audio 
    param:    
        audio_fn:      filename of librosa waveplot
        fn_fig:      filename for waveplot 
    return:    -
    ------------------------------------------------------------------------------
    ''' 
    
    def plot_wave(self, audio_fn, fn_fig=''):

        xticks = []
        i=0
        while i<=3:
            xticks.append(i)
            i+=0.3
        plt.xticks(xticks)
        
        fig, ax = plt.subplots(1,1)
        
        #EINSTELLUNGEN FÜR ANDERES LAYOUT DES PLOTS
        #ax.get_yaxis().set_visible(False)
        #ax.set_xticklabels(labels=[i+100 for i in xticks], fontdict={'fontsize':30,'fontname':'Calibri'})
        #matplotlib.rcParams.update({'font.size': 30})
        #font = {'fontname':'Calibri'}
        #plt.gcf().subplots_adjust(bottom=0.4)
        ax.set_xticks(xticks)
        plt.xlabel('')
        #audio, fs = librosa.load(audio_fn)
        audio = self.read_audio(audio_fn)
        librosa.display.waveplot(audio, sr = self.fs)
        
        plt.tight_layout()
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        if fn_fig != '':
            plt.savefig(fn_fig)

    '''
    ------------------------------------------------------------------------------
    desc:      calculate required masks for audio separation 
    param:    
        fn_Zxx_1:      filename of first audio stft of mixed audio
        fn_Zxx_2:      filename of second audio stft of mixed audio 
        fn_Zxx_mixed:      filename of mixed audio stft 
        fn_mask1:      filename for mask to receive first audio stft
        fn_mask2:      filename for mask to receive second audio stft
    return:    -
    ------------------------------------------------------------------------------
    '''             
                    
    def calculate_mask(self, fn_Zxx_1, fn_Zxx_2, fn_Zxx_mixed, fn_mask1, fn_mask2):
        Zxx_1 = np.load(fn_Zxx_1)
        Zxx_2 = np.load(fn_Zxx_2)
        Zxx_mixed = np.load(fn_Zxx_mixed)
        mask1 = Zxx_1/Zxx_mixed
        mask2 = Zxx_2/Zxx_mixed
        np.save(fn_mask1, mask1)
        np.save(fn_mask2, mask2)
        
    '''
    ------------------------------------------------------------------------------
    desc:      istft of wav file which is powerlaw-compressed
    param:    
        fn_Zxx:    filename of stft of required audio  
        fn_aud:    filename for audio 
    return:    -
    ------------------------------------------------------------------------------
    '''   
        
    def stft_to_wav(self, fn_Zxx, fn_aud):
        Zxx = np.load(fn_Zxx)
        R = abs(Zxx)
        phi = np.angle(Zxx)
        Zxx = R**(1/0.3) * np.exp(1j*phi)
        t, data =  scipy.signal.istft(Zxx, fs = self.fs, window = self.window, nperseg=self.nperseg, noverlap=self.noverlap, nfft=self.nfft)
        scipy.io.wavfile.write(fn_aud, self.fs, np.int16(data))
        
    '''
    ------------------------------------------------------------------------------
    desc:      istft of wav file which is not powerlaw-compressed
    param:    
        fn_Zxx:    filename of stft of required audio
        fn_aud:    filename for audio 
    return:    -
    ------------------------------------------------------------------------------
    '''   
    
    def stft_to_wav_no_compression(self, fn_Zxx, fn_aud):
        Zxx = np.load(fn_Zxx)
        t, data =  scipy.signal.istft(Zxx, fs = self.fs, window = self.window, nperseg=self.nperseg, noverlap=self.noverlap, nfft=self.nfft)
        scipy.io.wavfile.write(fn_aud, self.fs, np.int16(data))
        
    '''
    ------------------------------------------------------------------------------
    desc:      mix audio from av_speech dataset and audio or required class from audioSet with power-law compression
    param:    
        i1:    id of audio from av_speech
        file_noise:    filename of audio from audioSet
        dir_speech:    directory of audio from av_speech
        dir_noise:    directory of audio from audioSet
        dest_mixed:    directory for power-law compressed mixed audio 
        dest_speech:    directory for power-law compressed stft of audio from av_speech
        dest_noise:    directory for power-law compressed stft of audio from audioSet
    return:    -
    ------------------------------------------------------------------------------
    ''' 
    
    def mix_speech_noise(self, i1, file_noise, dir_speech, dir_noise, dest_mixed, dest_speech, dest_noise):
        file_speech = str(i1)+'_'+self.yt_id[i1]
        file_noise = os.path.splitext(file_noise)[0]
        filename_speech = dir_speech+file_speech
        filename_noise = dir_noise+file_noise
        file_mixed = str(i1)+'_'+file_noise
        filename_mixed = dest_mixed+file_mixed
        
        if not os.path.exists(dir_speech):
            os.makedirs(dir_speech)
        if not os.path.exists(dir_noise):
            os.makedirs(dir_noise)
        if not os.path.exists(dest_mixed):
            os.makedirs(dest_mixed)
            
        #mix audio
        if not Path(filename_speech+'.wav').is_file():
            print("File "+filename_speech+" exisitiert nicht")
        elif not Path(filename_noise+'.wav').is_file():
            print("File "+filename_noise+" exisitiert nicht")
        elif Path(filename_mixed+'.wav').is_file():
            print("File "+filename_mixed+" exisitiert bereits")
        else:            
            audio_speech = self.read_audio(filename_speech+'.wav')
            audio_noise = self.read_audio(filename_noise+'.wav')
            
            f, t, Zxx_id1 = scipy.signal.stft(audio_speech, fs = self.fs, window = self.window, nperseg=self.nperseg, noverlap=self.noverlap, nfft=self.nfft)
            f, t, Zxx_id2 = scipy.signal.stft(audio_noise, fs = self.fs, window = self.window, nperseg=self.nperseg, noverlap=self.noverlap, nfft=self.nfft)
            
            Zxx_mixed = Zxx_id1+Zxx_id2
            R_mixed = abs(Zxx_mixed)
            phi_mixed = np.angle(Zxx_mixed)
            Zxx_mixed = R_mixed**0.3 * np.exp(1j*phi_mixed)
            
            R1 = abs(Zxx_id1)
            phi1 = np.angle(Zxx_id1)
            Zxx_id1 = R1**0.3 * np.exp(1j*phi1)
            R2 = abs(Zxx_id2)
            phi2 = np.angle(Zxx_id2)
            Zxx_id2 = R2**0.3 * np.exp(1j*phi2)
        
            print("power law compressed")
        
            if Path(dest_speech+file_speech+'.npy').is_file():
                print("File "+dest_speech+file_speech+'.npy'+" exisitiert bereits")
            else:
                np.save(dest_speech+file_speech+'.npy', Zxx_id1)
            if Path(dest_noise+file_noise+'.npy').is_file():
                print("File "+dest_noise+file_noise+'.npy'+" exisitiert bereits")
            else:
                np.save(dest_noise+file_noise+'.npy', Zxx_id2)
            np.save(dest_mixed+file_mixed+'.npy', Zxx_mixed)
            
    '''
    ------------------------------------------------------------------------------
    desc:      mix audio from av_speech dataset and audio or required class from audioSet without power-law compression
    param:    
        i1:    id of audio from av_speech
        file_noise:    filename of audio from audioSet
        dir_speech:    directory of audio from av_speech
        dir_noise:    directory of audio from audioSet
        dest_mixed:    directory for mixed audio 
        dest_speech:    directory for stft of audio from av_speech
        dest_noise:    directory for stft of audio from audioSet
    return:    -
    ------------------------------------------------------------------------------
    ''' 
            
    def mix_speech_noise_no_compression(self, i1, file_noise, dir_speech, dir_noise, dest_mixed, dest_speech, dest_noise):
        file_speech = str(i1)+'_'+self.yt_id[i1]
        file_noise = os.path.splitext(file_noise)[0]
        filename_speech = dir_speech+file_speech
        filename_noise = dir_noise+file_noise
        file_mixed = str(i1)+'_'+file_noise
        filename_mixed = dest_mixed+file_mixed
        
        if not os.path.exists(dir_speech):
            os.makedirs(dir_speech)
        if not os.path.exists(dir_noise):
            os.makedirs(dir_noise)
        if not os.path.exists(dest_mixed):
            os.makedirs(dest_mixed)
            
        #mix audio
        if not Path(filename_speech+'.wav').is_file():
            print("File "+filename_speech+" exisitiert nicht")
        elif not Path(filename_noise+'.wav').is_file():
            print("File "+filename_noise+" exisitiert nicht")
        elif Path(filename_mixed+'.wav').is_file():
            print("File "+filename_mixed+" exisitiert bereits")
        else:            
            audio_speech = self.read_audio(filename_speech+'.wav')
            audio_noise = self.read_audio(filename_noise+'.wav')
            
            f, t, Zxx_id1 = scipy.signal.stft(audio_speech, fs = self.fs, window = self.window, nperseg=self.nperseg, noverlap=self.noverlap, nfft=self.nfft)
            f, t, Zxx_id2 = scipy.signal.stft(audio_noise, fs = self.fs, window = self.window, nperseg=self.nperseg, noverlap=self.noverlap, nfft=self.nfft)
            
            Zxx_mixed = Zxx_id1+Zxx_id2
        
            if Path(dest_speech+file_speech+'.npy').is_file():
                print("File "+dest_speech+file_speech+'.npy'+" exisitiert bereits")
            else:
                np.save(dest_speech+file_speech+'.npy', Zxx_id1)
            if Path(dest_noise+file_noise+'.npy').is_file():
                print("File "+dest_noise+file_noise+'.npy'+" exisitiert bereits")
            else:
                np.save(dest_noise+file_noise+'.npy', Zxx_id2)
            np.save(dest_mixed+file_mixed+'.npy', Zxx_mixed)

    '''
    ------------------------------------------------------------------------------
    desc:      mix audio from av_speech dataset with powerlaw-compression
    param:    
        i1:    first id of audio from av_speech
        i2:    second id of audio from av_speech
        dir_id1:    directory of first id 
        dir_id2:    directory of second id
        dest_mixed:    directory for power-law compressed mixed audio 
        dest_id1:    directory for power-law compressed stft of first audio from av_speech
        dest_noise:    directory for power-law compressed stft of second audio from av_speech
    return:    -
    ------------------------------------------------------------------------------
    ''' 
        
    def mix_audio(self,i1, i2, dir_id1, dir_id2, dest_mixed, dest_id1, dest_id2, label=""):
        #set filenames
        file_id1 = str(i1)+'_'+self.yt_id[i1]
        file_id2 = str(i2)+'_'+self.yt_id[i2]
        file_mixed = str(i1)+'_'+str(i2)
        
        filename_id1 = dir_id1+file_id1
        filename_id2 = dir_id2+file_id2
        filename_mixed = dest_mixed+file_mixed
        
        if not os.path.exists(dir_id1):
            os.makedirs(dir_id1)
        if not os.path.exists(dir_id2):
            os.makedirs(dir_id2)
        if not os.path.exists(dest_mixed):
            os.makedirs(dest_mixed)
            
        #mix audio
        if not Path(filename_id1+'.wav').is_file():
            print("File "+filename_id1+" exisitiert nicht")
        elif not Path(filename_id2+'.wav').is_file():
            print("File "+filename_id2+" exisitiert nicht")
        elif Path(filename_mixed+'.wav').is_file():
            print("File "+filename_mixed+" exisitiert bereits")
        else:            
            audio_id1 = self.read_audio(filename_id1+'.wav')
            audio_id2 = self.read_audio(filename_id2+'.wav')
            
            f, t, Zxx_id1 = scipy.signal.stft(audio_id1, fs = self.fs, window = self.window, nperseg=self.nperseg, noverlap=self.noverlap, nfft=self.nfft)
            f, t, Zxx_id2 = scipy.signal.stft(audio_id2, fs = self.fs, window = self.window, nperseg=self.nperseg, noverlap=self.noverlap, nfft=self.nfft)
            
            Zxx_mixed = Zxx_id1+Zxx_id2
            R_mixed = abs(Zxx_mixed)
            phi_mixed = np.angle(Zxx_mixed)
            Zxx_mixed = R_mixed**0.3 * np.exp(1j*phi_mixed)
            
            R1 = abs(Zxx_id1)
            phi1 = np.angle(Zxx_id1)
            Zxx_id1 = R1**0.3 * np.exp(1j*phi1)
            R2 = abs(Zxx_id2)
            phi2 = np.angle(Zxx_id2)
            Zxx_id2 = R2**0.3 * np.exp(1j*phi2)
        
            print("power law compressed")
        
            if Path(dest_id1+file_id1+'.npy').is_file():
                print("File "+dest_id1+file_id1+'.npy'+" exisitiert bereits")
            else:
                np.save(dest_id1+file_id1+'.npy', Zxx_id1)
            if Path(dest_id2+file_id2+'.npy').is_file():
                print("File "+dest_id2+file_id2+'.npy'+" exisitiert bereits")
            else:
                np.save(dest_id2+file_id2+'.npy', Zxx_id2)
            np.save(filename_mixed+'.npy', Zxx_mixed)
    '''
    ------------------------------------------------------------------------------
    desc:      mix audio from av_speech dataset without powerlaw-compression
    param:    
        i1:    first id of audio from av_speech
        i2:    second id of audio from av_speech
        dir_id1:    directory of first id 
        dir_id2:    directory of second id
        dest_mixed:    directory for power-law compressed mixed audio 
        dest_id1:    directory for power-law compressed stft of first audio from av_speech
        dest_noise:    directory for power-law compressed stft of second audio from av_speech
    return:    -
    ------------------------------------------------------------------------------
    '''
           
    def mix_audio_no_compression(self,i1, i2, dir_id1, dir_id2, dest_mixed, dest_id1, dest_id2, label=""):
        #set filenames
        file_id1 = str(i1)+'_'+self.yt_id[i1]
        file_id2 = str(i2)+'_'+self.yt_id[i2]
        file_mixed = str(i1)+'_'+str(i2)
        
        filename_id1 = dir_id1+file_id1
        filename_id2 = dir_id2+file_id2
        filename_mixed = dest_mixed+file_mixed
        
        if not os.path.exists(dir_id1):
            os.makedirs(dir_id1)
        if not os.path.exists(dir_id2):
            os.makedirs(dir_id2)
        if not os.path.exists(dest_mixed):
            os.makedirs(dest_mixed)
            
        #mix audio
        if not Path(filename_id1+'.wav').is_file():
            print("File "+filename_id1+" exisitiert nicht")
        elif not Path(filename_id2+'.wav').is_file():
            print("File "+filename_id2+" exisitiert nicht")
        elif Path(filename_mixed+'.wav').is_file():
            print("File "+filename_mixed+" exisitiert bereits")
        else:            
            audio_id1 = self.read_audio(filename_id1+'.wav')
            audio_id2 = self.read_audio(filename_id2+'.wav')
            
            f, t, Zxx_id1 = scipy.signal.stft(audio_id1, fs = self.fs, window = self.window, nperseg=self.nperseg, noverlap=self.noverlap, nfft=self.nfft)
            f, t, Zxx_id2 = scipy.signal.stft(audio_id2, fs = self.fs, window = self.window, nperseg=self.nperseg, noverlap=self.noverlap, nfft=self.nfft)
            
            Zxx_mixed = Zxx_id1+Zxx_id2
        
            if Path(dest_id1+file_id1+'.npy').is_file():
                print("File "+dest_id1+file_id1+'.npy'+" exisitiert bereits")
            else:
                np.save(dest_id1+file_id1+'.npy', Zxx_id1)
            if Path(dest_id2+file_id2+'.npy').is_file():
                print("File "+dest_id2+file_id2+'.npy'+" exisitiert bereits")
            else:
                np.save(dest_id2+file_id2+'.npy', Zxx_id2)
            np.save(filename_mixed+'.npy', Zxx_mixed)
            
