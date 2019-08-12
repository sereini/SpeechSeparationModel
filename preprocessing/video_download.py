# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 12:18:44 2019

@author: sscherrer

This program is used to download the raw video data from youtube. 
Start the program through the command prompt (ensure you have installed all 
required python packages) or with the powershell-script 
"Run-AVSpeechDownload.ps1" which will install the virtual environment 
"env_AVSpeech" in the current directory (if it's not already set up).
Following parameter have to be defined:
    --start_id
    --stop_id
Additional:
    --data_dir (default "images_AVSpeech")
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

import cv2
import pafy
#from pytube.exceptions import VideoUnavailable

from termcolor import colored

import time
import os
import argparse
import sys

class Video:
    def __init__(self):
        
        self.data_dir = ""
                
        # load model for face detection        
        self.conf_threshold = 0.6

        DNN = "CAFFE"
        if DNN == "CAFFE":
            modelFile = "./helperfiles/res10_300x300_ssd_iter_140000.caffemodel"
            configFile = "./helperfiles/deploy.prototxt.txt"
            self.net = cv2.dnn.readNetFromCaffe(configFile, modelFile)
        else:
            modelFile = "./helperfiles/opencv_face_detector_uint8.pb"
            configFile = "./helperfiles/opencv_face_detector.pbtxt"
            self.net = cv2.dnn.readNetFromTensorflow(modelFile, configFile)

        # sequenz length of video [s]
        self.seq_length = 3
            
        # load avspeech dataset
        self.av_data = pd.read_csv('./helperfiles/avspeech_train.csv', delimiter=',', header=None)

        self.yt_id = self.av_data.iloc[:,0]
        self.start = self.av_data.iloc[:,1]
        
        # centerpoint
        self.cp_x = self.av_data.iloc[:,3]
        self.cp_y = self.av_data.iloc[:,4]

        self.frameToDetect = 0
        self.firstFrameDetected = False
        
        self.nDown = 0
        self.starttime = time.time()
        
        self.save_frames = []
        self.width = 0
        self.height = 0
        
    '''
    ------------------------------------------------------------------------------
    desc:        calculate the missing values if the face detection could not
                 detect a face per frame
                    - median over 15 values if the first or/and last frame 
                      is missing
                    - linear interpolation if a frame between the first and 
                      last frame is missing
    param: 
       v_edges:  array of the coords from the face detection (all frames)
                    - nan if face could not be detected
    return:      
        v_edges: interpolated array of coords
    ------------------------------------------------------------------------------
    '''
    def f_calcMissingVal(self, v_edges):
        v_last = v_edges.shape[1] - 1
        if np.isnan(v_edges[1, v_last]):            
            v_edges[1, v_last] = np.nanmedian(v_edges[1, (v_last-15):])
            v_edges[2, v_last] = np.nanmedian(v_edges[2, (v_last-15):])
            v_edges[0, v_last] = v_last
        for l in range(v_edges.shape[1]):
            if np.isnan(v_edges[1, l]):
                v_edges[0, l] = l
                if l == 0:                    
                    v_edges[1, l] = np.nanmedian(v_edges[1, :15])
                    v_edges[2, l] = np.nanmedian(v_edges[2, :15])
                else:
                    for m in range(l, v_edges.shape[1]):
                        if not np.isnan(v_edges[1, m]):
                            # linear interpolation
                            a1, b1 = np.polyfit(np.array([0, m - l + 1]), np.array([v_edges[1, l - 1], v_edges[1, m]]),
                                                1)  
                            a2, b2 = np.polyfit(np.array([0, m - l + 1]), np.array([v_edges[2, l - 1], v_edges[2, m]]),
                                                1) 
                            # calc absolute value
                            v_edges[1, l] = a1 + b1
                            v_edges[2, l] = a2 + b2
                            break
        return v_edges

    '''
    ------------------------------------------------------------------------------
    desc:       calculate fixed window size (median over all frames) and
                the smoothed main focus (face center point) over all frames 
                (using savgol filter)
    param:  
        x:      x-coords as an array (2D)  
        y:      y-coords as an array (2D)
            
    return:     
        windows_size:   fixed window size
        main_focus:     smoothed main focus     
    ------------------------------------------------------------------------------
    '''
    def f_analyzeVideo(self, x, y):
        # window size
        diff_x = (x[2, :] - x[1, :])
        diff_y = (y[2, :] - y[1, :])
        med_x = np.median(diff_x)
        med_y = np.median(diff_y)

        # main focus
        center_x = ((x[2, :] + x[1, :]) / 2)
        center_y = ((y[2, :] + y[1, :]) / 2)
        filter_x = savgol_filter(center_x, 39, 2)
        filter_y = savgol_filter(center_y, 39, 2)

        # arrays of non-smoothed an smoothed
        window_size = np.array([diff_x, diff_y, med_x, med_y])
        main_focus = np.array([center_x, center_y, filter_x, filter_y])

        return window_size, main_focus
    
    
    '''
    ------------------------------------------------------------------------------
    desc:           plot function for analysis
    
    param:  
        v_path:     dir to save the figure
        v_str:      title, x and y labels as an array (strings)
        v_legend:   legend as an array (strings)
        v_plt:      data to plot in same figure
    
    return:         -
    ------------------------------------------------------------------------------
    '''
    def f_plot(self, v_path, v_str, v_legend, v_plt):
        plt.figure(v_str[0], clear=True)
        plt.title(v_str[0])
        for cnt in range(0, v_plt.shape[0], 2):
            plt.plot(v_plt[cnt], v_plt[cnt + 1])
        plt.xlabel(v_str[1])
        plt.ylabel(v_str[2])
        plt.gca().legend(v_legend)
        plt.savefig(v_path)
        plt.close(v_str[0])
               

    '''
    ------------------------------------------------------------------------------
    desc:               subplot function for analysis
    
    param:  
        v_path:         dir to save the figure
        v_str:          title, x and y labels as an array (strings)
        v_legend:       legend as an array (strings)
        v_plt[p, cnt]:  data to plot
                        p:   data for multiple subplots
                        cnt: data to plot in same figure
    return:             -
    ------------------------------------------------------------------------------
    '''
    def f_subplot(self, v_path, v_str, v_legend, v_plt):
        plt.figure(v_str[0, 0], clear=True)
        for p in range(0, v_plt.shape[0]):
            plt.subplot(2, 1, (p + 1))
            for cnt in range(0, v_plt.shape[1], 2):
                plt.plot(v_plt[p, cnt], v_plt[p, cnt + 1])
            plt.xlabel(v_str[p, 1])
            plt.ylabel(v_str[p, 2])
            plt.gca().legend(v_legend[p, :])
        plt.savefig(v_path)
        plt.close(v_str[0, 0])


    '''
    ------------------------------------------------------------------------------
    desc:       plot analysis of edges (coords), window size and main focus
    param:  
        idx:    iterator from avspeech dataset (control variable)
        x:      x-coords of the current video
        y:      y-coords of the current video
    
    return:     -
    ------------------------------------------------------------------------------
    '''
    def f_plotAnalysis(self, idx, x, y):
        
        d = "./plots"
        if not os.path.exists(d):
            os.makedirs(d)
            print("Folder created:", d)
    
        # define path
        plot_save = "./plots/" + self.yt_id[idx] + "._coords.png"
        plot_saveD = "./plots/" + self.yt_id[idx] + "_window_size.png"
        plot_saveC = "./plots/" + self.yt_id[idx] + "_main_focus.png"

        window_size, main_focus = self.f_analyzeVideo(x, y)

        # plot edges
        self.f_subplot(plot_save,
                  np.array([['coords edges', '# Frame', 'x Koordinate'], ['coords edges', '# Frame', 'y Koordinate']]),
                  np.array([['x0', 'x1'], ['y0', 'y1']]),
                  np.array([[x[0, :], x[1, :], x[0, :], x[2, :]], [y[0, :], y[1, :], y[0, :], y[2, :]]]))

        # plot window size
        self.f_plot(plot_saveD, np.array(['window size', '#Frame', 'diff']), np.array(['rect width','median width','rect height','median height']),
                                     np.array([x[0, :], window_size[0], [0,74], [window_size[2], window_size[2]], y[0, :], window_size[1], [0,74], [window_size[3], window_size[3]]]))

        # plot main focus with filter
        self.f_plot(plot_saveC, np.array(['main focus', '#Frame', 'center']),
               np.array(['sa', 'filter_a', 'sb', 'filter_b']),
               np.array([x[0, :], main_focus[0], x[0, :], main_focus[2], y[0, :], main_focus[1], y[0, :], main_focus[3]]))
    
    
    '''
    ------------------------------------------------------------------------------
    desc:       face detection in whole frame    
    param:  
        frame:               current frame
        width:               width of the video stream
        height:              height of the video stream
        edges_x:             x-coords array (needed for save detection)
        edges_y:             y-coords array (needed for save detection)
        coordinates_faceCut: define smaller area for face detection 
                             (needed for func facedetection_cutFrame)
        cp_scaled:           centerpoint (given from avspeech dataset)
                             absolute pixel values      
                            
    return:     
        edges_x, edges_y:    updated arrays (detected coords)   
        coordinates_faceCut: smaller area for face detection 
    ------------------------------------------------------------------------------
    '''
    def facedetection_wholeFrame(self, frame, width, height, edges_x, edges_y, k, coordinates_faceCut, cp_scaled):
        self.firstFrameDetected = False;
        # get shape of frame
        h, w = frame.shape[:2]  
        blob = cv2.dnn.blobFromImage(frame, 1.0, (244, 244), (104.0, 117.0, 123.0))
        self.net.setInput(blob)
        detections = self.net.forward()

        if (len(detections) > 0):
            for j in range(0, detections.shape[2]):
                confidence = detections[0, 0, j, 2]
                if (confidence > self.conf_threshold):
                    x0 = int(detections[0, 0, j, 3] * w)
                    y0 = int(detections[0, 0, j, 4] * h)
                    x1 = int(detections[0, 0, j, 5] * w)
                    y1 = int(detections[0, 0, j, 6] * h)
        
                    # detect if the centerpoint is in the rectangle
                    if ((x0 < cp_scaled[0] and cp_scaled[0] < x1) and (
                            y0 < cp_scaled[1] and cp_scaled[1] < y1)):
                        
                        # resize the rectangle of the detected face to a bigger rectangle
                        offsetX = int((x1 - x0) * 0.3);
                        offsetY = int((y1 - y0) * 0.3);
                        coordinates_faceCut = np.array([x0 - offsetX, y0 - offsetY, x1 + offsetX,
                                                        y1 + offsetY])
                        
                        # check if the coords are out of bounds
                        if (x0 - offsetX < 0):
                            coordinates_faceCut[0] = 0;
                        if (y0 - offsetY < 0):
                            coordinates_faceCut[1] = 0;
                        if (x1 + offsetX > width):
                            coordinates_faceCut[2] = width;
                        if (y1 + offsetY > height):
                            coordinates_faceCut[3] = height;
        
                        # add detected coordinates
                        edges_x[:, k] = np.array([k, x0, x1]);
                        edges_y[:, k] = np.array([k, y0, y1]);
                        self.firstFrameDetected = True;
        
                        return edges_x, edges_y, coordinates_faceCut

        self.firstFrameDetected = False;
        return edges_x, edges_y, coordinates_faceCut

    '''
    ------------------------------------------------------------------------------
    desc:       face detection in smaller area of the whole frame    
    param:  
        frame:               current frame
        edges_x:             x-coords array (needed for save detection)
        edges_y:             y-coords array (needed for save detection)
        coordinates_faceCut: smaller area for face detection
        
    return:     
        edges_x, edges_y:    updated arrays (detected coords)
    ------------------------------------------------------------------------------
    '''
    def facedetection_cutFrame(self, frame, edges_x, edges_y, k, coordinates_faceCut):

        # cut frame
        faceCut = frame[coordinates_faceCut[1]:coordinates_faceCut[3],
                  coordinates_faceCut[0]:coordinates_faceCut[2]] 
        
        # get shape of frame
        h, w = faceCut.shape[:2]  
        blob = cv2.dnn.blobFromImage(faceCut, 1.0, (244, 244), (104.0, 117.0, 123.0))
        self.net.setInput(blob)
        detections = self.net.forward()        

        # check the detections
        if (len(detections) > 0):
            for j in range(0, detections.shape[2]):
                confidence = detections[0, 0, j, 2]
                if (confidence > self.conf_threshold):
                    x0 = int(detections[0, 0, j, 3] * w)
                    y0 = int(detections[0, 0, j, 4] * h)
                    x1 = int(detections[0, 0, j, 5] * w)
                    y1 = int(detections[0, 0, j, 6] * h)
                    
                    # add detected coordinates
                    edges_x[:, k] = np.array([k, coordinates_faceCut[0] + x0, coordinates_faceCut[0] + x1]);
                    edges_y[:, k] = np.array([k, coordinates_faceCut[1] + y0, coordinates_faceCut[1] + y1]);
                    return edges_x, edges_y;

        return edges_x, edges_y;

    '''
    ------------------------------------------------------------------------------
    desc:       save detected faces as video (75 frames correspond to 3 sec)   
    param:  
        idx:        iterator from avspeech dataset (control variable)
        width:      width of video (fixed window size)
        height:     height of video (fixed window size)
        center_x:   smoothed main focus (x)
        center_y:   smoothed main focus (y)
        
    return:         -
    ------------------------------------------------------------------------------
    '''
    def downloadCutVideo(self, idx, width, height, center_x, center_y):
        d = "./videos"
        if not os.path.exists(d):
            os.makedirs(d)
            print("Folder created:", d)
                
        # relative destination path
        path = "./videos/" + self.yt_id[idx] + ".avi"
        
        # init video writer
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(path, fourcc, 25, (width, height))

        j = 0

        while (j < 75):
            x0 = int(center_x[j]-width/2)
            x1 = int(center_x[j]+width/2)
            y0 = int(center_y[j]-height/2)
            y1 = int(center_y[j]+height/2)
            
            if(x0 < 0) or (x1 > self.width) or (y0 < 0) or (y1 > self.height):
                print("Face is out of bounds, the video can not be extracted!")
                break
            
            # save extracted face
            frame = self.save_frames[j][y0:y1, x0:x1]
            out.write(frame)

            j += 1

        out.release()

    '''
    ------------------------------------------------------------------------------
    desc:       save detected faces as single images (75 frames correspond to 3 sec)   
    param:  
        idx:        iterator from avspeech dataset (control variable)
        width:      width of video (fixed window size)
        height:     height of video (fixed window size)
        center_x:   smoothed main focus (x)
        center_y:   smoothed main focus (y)
        
    return:         -
    ------------------------------------------------------------------------------
    '''
    def downloadCutImages(self, idx, width, height, center_x, center_y):
        d = self.data_dir + "/"+str(idx)+"_"+str(self.yt_id[idx])+"/"
        print(d)
        if not os.path.exists(d):
            os.makedirs(d)
            print("Folder created:", d)

        k = 0

        while k < 75:
                
            # workaround: check if the coords are out of bounds
            x0 = int(center_x[k]-width/2)
            if(x0 < 0):
                x0 = 0
            
            x1 = int(center_x[k]+width/2)
            if (x1 > self.width):
                x1 = int(self.width)
                
            y0 = int(center_y[k]-height/2)
            if (y0 < 0):
                y0 = 0
                
            y1 = int(center_y[k]+height/2)
            if(y1 > self.height):
                y1 = int(self.height)
                
            frame = self.save_frames[k][y0:y1, x0:x1]
            
            # save extracted face as single image if frame not null
            if frame[0].size > 0 and frame[1].size > 0:
                cv2.imwrite(d+str(self.yt_id[idx])+"_"+str(k)+".png", frame)
                
                
            k += 1
           
            
    '''
    ------------------------------------------------------------------------------
    desc:       get youtube videostream   
    param:  
        idx:    iterator from avspeech dataset (control variable)
        
    return:     
        cap:    youtube stream
    ------------------------------------------------------------------------------
    '''
    def getVideostream(self, idx):
        url = "https://www.youtube.com/watch?v=" + self.yt_id[idx]
        
        vid = pafy.new(url)
        best = vid.getbest()
        cap = cv2.VideoCapture(best.url)
        
        # pytube as an alternative, if pafy (youtube_dl) do not work
        '''vid = pytube.YouTube(url)
        vid_res = vid.streams.filter(progressive=True).order_by('resolution').desc()
        best = vid_res.all()[0]

        print(best.url)'''
        
        # Debug
        print(self.yt_id[idx])
        print(idx)

        return cap


    '''
    ------------------------------------------------------------------------------
    desc:       get frames from stream (resampling to 25 fps, nearest neighbor 
                interpolation) and detect faces
                    - if fps < 25, single frames are used multiple times
                    - if fps > 25, single frames are skipped
    param:  
        idx:                    iterator from avspeech dataset (control variable)
        cap:                    video stream
        edges_x:                empty array to save the x-coords 
                                (detected face per frame)
        edges_y:                empty array to save the y-coords 
                                (detected face per frame)
        coordinates_faceCut:    empty array for smaller face detection area
        
    return:    
        edges_x, edges_y:       updated arrays (detected coords)        
    ------------------------------------------------------------------------------
    '''
    def getFaces(self, idx, cap, edges_x, edges_y, coordinates_faceCut):    
        # get size of video stream
        self.width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

        # centerpoint scaled to frame size
        cp_scaled = np.array([self.cp_x[idx] * self.width, self.cp_y[idx] * self.height], np.int32)  

        cap.set(cv2.CAP_PROP_POS_MSEC, int(self.start[idx] * 1000))
        
        
        print("Msec_start: ", int(cap.get(cv2.CAP_PROP_POS_MSEC)))
        
        # get fps of video as integer
        fps = np.around(cap.get(cv2.CAP_PROP_FPS))
        
        # number of frames, which have to be read 
                    # fps <= 25 : nofFrames = 75
                    # fps > 25  : nofFrames > fps*3s
        nofFrames = int(fps * self.seq_length)
        
        if(nofFrames < 75):
            nofFrames = 75
        
        k = 0       # current frame for new video (25 fps)
        nofF = 0    # current frame of video stream (youtube video)
        n = 0       # if fps > 25: positive shift per frame that is skipped
                    # if fps < 25: negative shift per frame that is used multiple times                 
            
        
        while k < nofFrames:
            
            if(fps >= 25 or (fps < 25 and abs((nofF)/25-(nofF-n)/int(fps)) <= abs((nofF)/25-(nofF-n-1)/int(fps)))):
                ret, frame = cap.read()
            
            if (nofFrames == 75 or (abs(nofF/25-(nofF+n)/int(fps)) <= abs(nofF/25-(nofF+n+1)/int(fps))) and nofF < 75):
                if not ret:
                    self.nDown += 1
                    break
                
                # save frame to list
                self.save_frames.append(frame)
                
                if (nofF == self.frameToDetect):
                    # detect a face in whole frame within the center point
                    edges_x, edges_y, coordinates_faceCut = self.facedetection_wholeFrame(frame, self.width, 
                                                                           self.height, edges_x,edges_y, nofF, coordinates_faceCut, cp_scaled)
                    if (self.firstFrameDetected == False):
                        self.frameToDetect += 1;
                
                if (self.firstFrameDetected):
                    # detect face in cut frame
                    edges_x, edges_y = self.facedetection_cutFrame(frame, edges_x, edges_y, nofF, coordinates_faceCut)


                if (fps < 25 and abs((nofF)/25-(nofF-n)/int(fps)) > abs((nofF)/25-(nofF-n-1)/int(fps))):
                    n += 1    
            
                nofF += 1
            
            
            elif (fps > 25 and abs(nofF/25-(nofF+n)/int(fps)) > abs(nofF/25-(nofF+n+1)/int(fps))):
                n += 1                
            
            k += 1
                 
            if (nofF % 10 == 0):
                self.frameToDetect = nofF;
                self.firstFrameDetected = False;                 
                 
        self.save_frames = np.array(self.save_frames)         
        print("Msec_end: ", int(cap.get(cv2.CAP_PROP_POS_MSEC)))
        
        return edges_x, edges_y
    
    '''
    ------------------------------------------------------------------------------
    desc:       download detected faces
    param:  
        mode:      mode to save detected face
                        - 'image': single images (75)
                        - 'video': frames as video (*.avi file)
        idx:       iterator from avspeech dataset (control variable)
        edges_x:   array with x-coords
        edges_y:   array with y-coords 
    
    return:        -
    ------------------------------------------------------------------------------
    '''
    def downloadFaces(self, mode, idx, edges_x, edges_y):
        det_fail = np.count_nonzero(np.isnan(edges_x[0, :]))
        print("Fails in Detection: ", det_fail)
        
        if (det_fail > 10):
            print("Too many fails in face detection")
        else:
            edges_x = self.f_calcMissingVal(edges_x)
            edges_y = self.f_calcMissingVal(edges_y)

            # uncomment the next line if the analysis should be plotted
            #self.f_plotAnalysis(idx, edges_x, edges_y)
            windowsize, mainfocus = self.f_analyzeVideo(edges_x, edges_y)
            
            if mode is "video":
                self.downloadCutVideo(idx, int(windowsize[2]), int(windowsize[3]), mainfocus[2], mainfocus[3])

            elif mode is "image":
                self.downloadCutImages(idx, int(windowsize[2]), int(windowsize[3]), mainfocus[2], mainfocus[3])

    
    '''
    ------------------------------------------------------------------------------
    desc:       basic function
    param:  
        start_id:   start id (iterator from avpeech dataset as control variable)
        stop_id:    stop id 
        data_dir:   dir to save data
    
    return:         -
    ------------------------------------------------------------------------------
    '''
    def downloadVideo(self, start_id, stop_id, data_dir):

        self.data_dir = data_dir
        dirs = ["./" + self.data_dir ]
        for d in dirs:
            if not os.path.exists(d):
                os.makedirs(d)
                print("Folder created:", d)

        i_start = start_id
        i_end = stop_id
        i = i_start
        starttime = time.time()
        
        cap = None
        edges_x = np.full((3,75), dtype=np.float64, fill_value=np.nan)
        edges_y = np.full((3,75), dtype=np.float64, fill_value=np.nan)
        coordinates_faceCut = np.full((4,1), dtype=np.float64, fill_value=np.nan)
        
        while i < i_end:
            try:
                cap = self.getVideostream(i)
                edges_x, edges_y = self.getFaces(i, cap, edges_x, edges_y, coordinates_faceCut)
                cap.release() 
            except IOError as e:
                print("IOError")
                print(e)
                self.nDown += 1
            except AttributeError as e:
                print("AttributeError")
                print(e)
                self.nDown += 1
            except KeyError as e:
                print("KeyError")
                print(e)
                self.nDown += 1    
            #except VideoUnavailable:            #pytube exception
            #    print("This video is unavailable.")
            #    self.nDown += 1
            else:
                self.downloadFaces("image", i, edges_x, edges_y)
            finally:
                #test
                edges_x[:] = np.nan
                edges_y[:] = np.nan
                coordinates_faceCut[:] = np.nan
                
                self.frameToDetect = 0
                self.firstFrameDetected = False
                self.save_frames = []    
                cv2.destroyAllWindows()
                
                i += 1 
                if i - i_start - self.nDown > 0:
                    timePerVideo = (time.time() - starttime) / (i - i_start - self.nDown)
                    print('Time per video in seconds: ', timePerVideo)
                print('-------------------------------------------------')
                
                
def main(args):
    vid = Video()
    vid.downloadVideo(args.start_id, args.stop_id, args.data_dir)

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--start_id', type=int,
        help='Define start id of AVSpeech Dataset.', required=True)
    parser.add_argument('--stop_id', type=int,
        help='Define stop id of AVSpeech Dataset.', required=True)
    parser.add_argument('--data_dir', type=str,
        help='Define a directory (relative path) to save the images.', default="images_AVSpeech")
    
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
