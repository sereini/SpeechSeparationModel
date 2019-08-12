# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 18:08:36 2019

@author: chalbeisen
source: https://stackoverflow.com/questions/4158502/kill-or-terminate-subprocess-when-timeout
This program is used to start a subprocess in a Thread 
"""

import subprocess as sub
import threading
import psutil
import time

class RunCmd(threading.Thread):
    '''
    ------------------------------------------------------------------------------
    desc:      init RunCmd class and set global parameters
    param:    -
    return:    -
    ------------------------------------------------------------------------------
    '''
    def __init__(self, cmd, timeout, maxCount, wait=False):
        threading.Thread.__init__(self)
        self.cmd = cmd
        self.timeout = timeout
        self.maxCount = maxCount
        self.wait = wait

    '''
    ------------------------------------------------------------------------------
    desc:      get act amount of parallel running threads 
    param:    -
    return:   act amount of parallel running threads 
    ------------------------------------------------------------------------------
    '''
    def childCount(self):
        current_process = psutil.Process()
        children = current_process.children()
        return(len(children))
    '''
    ------------------------------------------------------------------------------
    desc:      start subprocess in own thread,
               set maximum amount of parallel Threads, if maximum amount is reached -> wait for a Thread to complete
               if wait parameter is set, wait for running thread to complete to start a new thread (only one running thread)
    param:    -
    return:    -   
    ------------------------------------------------------------------------------
    '''
    def run(self):
        self.p = sub.Popen(self.cmd)
        if(self.wait==True):
            self.p.communicate()
            
        while self.childCount() > self.maxCount:
            time.sleep(0.25)
            self.p.communicate() 
            
    '''
    ------------------------------------------------------------------------------
    desc:      start thread and kill it if timeout exceeds 
    param:    -
    return:    -   
    ------------------------------------------------------------------------------
    '''
    def Run(self):
        self.start()
        self.join(self.timeout)
        
        if self.is_alive():
            self.p.kill()      
            self.join()