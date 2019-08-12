# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 16:03:59 2019

@author: sscherrer/chalbeisen

This program defines the neural network for the audio-visual speech separation 
model for two independent speakers.
"""

from __future__ import absolute_import, division, print_function

import tensorflow as tf
import os
import random


tf.logging.set_verbosity(tf.logging.INFO)

'''
------------------------------------------------------------------------------
desc:            defines the loss as mse between labels and predictions
param: 
    calc_mask1:  prediction of the avspeech model for speaker 1
    calc_mask2:  prediction of the avspeech model for speaker 2
    aud_mix:     superimposed audio signal (both speaker)
    lbl_aud1:    label of speaker 1
    lbl_aud2:    label of speaker 2

return:          loss (scalar)
------------------------------------------------------------------------------
'''
def loss_mean_squared_error(calc_mask1, calc_mask2, aud_mix, lbl_aud1, lbl_aud2):
    calc_mask1 = tf.complex(calc_mask1[:,:,:,0], calc_mask1[:,:,:,1])
    calc_mask2 = tf.complex(calc_mask2[:,:,:,0], calc_mask2[:,:,:,1])
    aud_mix = tf.complex(aud_mix[:,:,:,0], aud_mix[:,:,:,1])
    lbl_aud1 = tf.complex(lbl_aud1[:,:,:,0], lbl_aud1[:,:,:,1])
    lbl_aud2 = tf.complex(lbl_aud2[:,:,:,0], lbl_aud2[:,:,:,1])
     
    pred1 = tf.math.multiply(aud_mix,calc_mask1)
    pred2 = tf.math.multiply(aud_mix,calc_mask2)

    l1 = tf.square(tf.real(pred1-lbl_aud1)) + tf.square(tf.imag(pred1-lbl_aud1))
    l2 = tf.square(tf.real(pred2-lbl_aud2)) + tf.square(tf.imag(pred2-lbl_aud2))
    return tf.reduce_mean(tf.add(l1, l2))

'''
------------------------------------------------------------------------------
desc:          model function for the neural network
param: 
    features:  input data as dictionary
    labels:    labels as dictionary
    mode:      mode (EVAL or TRAIN)

return:        EstimatorSpec for TRAIN or EVAL
------------------------------------------------------------------------------
'''
def cnn_model_fn(features, labels, mode):
    
    # define names for the nodes in tensorboard, used for prediction
    vid1 = tf.identity(features["vid1"], name="vid1")
    vid2 = tf.identity(features["vid2"], name="vid2")
    aud_mix = tf.identity(features["aud"], name="aud_mix")
    aud1 = labels["aud1"]
    aud2 = labels["aud2"]
    
    #audio stream
    def dnn_audio(layer_name, x):
        
        with tf.variable_scope(layer_name): 
            input_layer_raw = tf.reshape(x, [-1, x.shape[1], x.shape[2], x.shape[3]])
            input_layer = tf.transpose(input_layer_raw,[0,2,1,3])
            
            # 1. convolutional layer 
            with tf.variable_scope('layer1_a'):
                conv1_a = tf.layers.conv2d(
                  inputs=input_layer,
                  filters=96,
                  kernel_size=[1, 7],
                  dilation_rate=(1,1),
                  padding="same",
                  activation=tf.nn.relu,
                  kernel_initializer=tf.initializers.he_uniform(),
                  name="conv1_a")
                
                # 1. batch normalization
                norm1_a = tf.layers.batch_normalization(
                        inputs=conv1_a,
                        name="norm1_a",
                        training= mode==tf.estimator.ModeKeys.TRAIN)
            
            # 2. convolutional layer 
            with tf.variable_scope('layer2_a'):
                conv2_a = tf.layers.conv2d(
                  inputs=norm1_a,
                  filters=96,
                  kernel_size=[7, 1],
                  dilation_rate=(1,1),
                  padding="same",
                  activation=tf.nn.relu,
                  kernel_initializer=tf.initializers.he_uniform(),
                  name="conv2_a")
                
                # 2. batch normalization
                norm2_a = tf.layers.batch_normalization(
                        inputs=conv2_a,
                        name="norm2_a",
                        training= mode==tf.estimator.ModeKeys.TRAIN)
            
            # 3. convolutional layer 
            with tf.variable_scope('layer3_a'):
                conv3_a = tf.layers.conv2d(
                  inputs=norm2_a,
                  filters=96,
                  kernel_size=[5, 5],
                  dilation_rate=(1,1),
                  padding="same",
                  activation=tf.nn.relu,
                  kernel_initializer=tf.initializers.he_uniform(),
                  name="conv3_a")
                
                # 3. batch normalization
                norm3_a = tf.layers.batch_normalization(
                        inputs=conv3_a,
                        name="norm3_a",
                        training= mode==tf.estimator.ModeKeys.TRAIN)
            
            # 4. convolutional layer 
            with tf.variable_scope('layer4_a'):
                conv4_a = tf.layers.conv2d(
                  inputs=norm3_a,
                  filters=96,
                  kernel_size=[5, 5],
                  dilation_rate=(2,1),
                  padding="same",
                  activation=tf.nn.relu,
                  kernel_initializer=tf.initializers.he_uniform(),
                  name="conv4_a")
                
                # 4. batch normalization
                norm4_a = tf.layers.batch_normalization(
                        inputs=conv4_a,
                        name="norm4_a",
                        training= mode==tf.estimator.ModeKeys.TRAIN)
            
            # 5. convolutional layer 
            with tf.variable_scope('layer5_a'):
                conv5_a = tf.layers.conv2d(
                  inputs=norm4_a,
                  filters=96,
                  kernel_size=[5, 5],
                  dilation_rate=(4,1),
                  padding="same",
                  activation=tf.nn.relu,
                  kernel_initializer=tf.initializers.he_uniform(),
                  name="conv5_a")
                
                # 5. batch normalization
                norm5_a = tf.layers.batch_normalization(
                        inputs=conv5_a,
                        name="norm5_a",
                        training= mode==tf.estimator.ModeKeys.TRAIN)
            
            # 6. convolutional layer 
            with tf.variable_scope('layer6_a'):
                conv6_a = tf.layers.conv2d(
                  inputs=norm5_a,
                  filters=96,
                  kernel_size=[5, 5],
                  dilation_rate=(8,1),
                  padding="same",
                  activation=tf.nn.relu,
                  kernel_initializer=tf.initializers.he_uniform(),
                  name="conv6_a")
                
                # 6. batch normalization
                norm6_a = tf.layers.batch_normalization(
                        inputs = conv6_a,
                        name="norm6_a",
                        training= mode==tf.estimator.ModeKeys.TRAIN)
            
            # 7. convolutional layer 
            with tf.variable_scope('layer7_a'):
                conv7_a = tf.layers.conv2d(
                  inputs=norm6_a,
                  filters=96,
                  kernel_size=[5, 5],
                  dilation_rate=(16,1),
                  padding="same",
                  activation=tf.nn.relu,
                  kernel_initializer=tf.initializers.he_uniform(),
                  name="conv7_a")
                
                # 7. batch normalization
                norm7_a = tf.layers.batch_normalization(
                        inputs=conv7_a,
                        name="norm7_a",
                        training= mode==tf.estimator.ModeKeys.TRAIN)
            
            # 8. convolutional layer 
            with tf.variable_scope('layer8_a'):
                conv8_a = tf.layers.conv2d(
                  inputs=norm7_a,
                  filters=96,
                  kernel_size=[5, 5],
                  dilation_rate=(32,1),
                  padding="same",
                  activation=tf.nn.relu,
                  kernel_initializer=tf.initializers.he_uniform(),
                  name="conv8_a")
                
                # 8. batch normalization
                norm8_a = tf.layers.batch_normalization(
                        inputs=conv8_a,
                        name="norm8_a",
                        training= mode==tf.estimator.ModeKeys.TRAIN)
            
            # 9. convolutional layer 
            with tf.variable_scope('layer9_a'):
                conv9_a = tf.layers.conv2d(
                  inputs=norm8_a,
                  filters=96,
                  kernel_size=[5, 5],
                  dilation_rate=(1,1),
                  padding="same",
                  activation=tf.nn.relu,
                  kernel_initializer=tf.initializers.he_uniform(),
                  name="conv9_a")
                
                # 9. batch normalization
                norm9_a = tf.layers.batch_normalization(
                        inputs=conv9_a,
                        name="norm9_a",
                        training= mode==tf.estimator.ModeKeys.TRAIN)
            
            # 10. convolutional layer 
            with tf.variable_scope('layer10_a'):
                conv10_a = tf.layers.conv2d(
                  inputs=norm9_a,
                  filters=96,
                  kernel_size=[5, 5],
                  dilation_rate=(2,2),
                  padding="same",
                  activation=tf.nn.relu,
                  kernel_initializer=tf.initializers.he_uniform(),
                  name="conv10_a")
                
                # 10. batch normalization
                norm10_a = tf.layers.batch_normalization(
                        inputs=conv10_a,
                        name="norm10_a",
                        training= mode==tf.estimator.ModeKeys.TRAIN)
            
            # 11. convolutional layer 
            with tf.variable_scope('layer11_a'):
                conv11_a = tf.layers.conv2d(
                  inputs=norm10_a,
                  filters=96,
                  kernel_size=[5, 5],
                  dilation_rate=(4,4),
                  padding="same",
                  activation=tf.nn.relu,
                  kernel_initializer=tf.initializers.he_uniform(),
                  name="conv11_a")
                
                # 11. batch normalization
                norm11_a = tf.layers.batch_normalization(
                        inputs=conv11_a,
                        name="norm11_a",
                        training= mode==tf.estimator.ModeKeys.TRAIN)
        
            # 12. convolutional layer 
            with tf.variable_scope('layer12_a'):
                conv12_a = tf.layers.conv2d(
                  inputs=norm11_a,
                  filters=96,
                  kernel_size=[5, 5],
                  dilation_rate=(8,8),
                  padding="same",
                  activation=tf.nn.relu,
                  kernel_initializer=tf.initializers.he_uniform(),
                  name="conv12_a")
                
                # 12. batch normalization
                norm12_a = tf.layers.batch_normalization(
                        inputs=conv12_a,
                        name="norm12_a",
                        training= mode==tf.estimator.ModeKeys.TRAIN)
            
            # 13. convolutional layer 
            with tf.variable_scope('layer13_a'):
                conv13_a = tf.layers.conv2d(
                  inputs=norm12_a,
                  filters=96,
                  kernel_size=[5, 5],
                  dilation_rate=(16,16),
                  padding="same",
                  activation=tf.nn.relu,
                  kernel_initializer=tf.initializers.he_uniform(),
                  name="conv13_a")
                
                # 13. batch normalization
                norm13_a = tf.layers.batch_normalization(
                        inputs=conv13_a,
                        name="norm13_a",
						training= mode==tf.estimator.ModeKeys.TRAIN)
            
            # 14. convolutional layer 
            with tf.variable_scope('layer14_a'):
                conv14_a = tf.layers.conv2d(
                  inputs=norm13_a,
                  filters=96,
                  kernel_size=[5, 5],
                  dilation_rate=(32,32),
                  padding="same",
                  activation=tf.nn.relu,
                  kernel_initializer=tf.initializers.he_uniform(),
                  name="conv14_a")
                
                # 14. batch normalization
                norm14_a = tf.layers.batch_normalization(
                        inputs=conv14_a,
                        name="norm14_a",
						training= mode==tf.estimator.ModeKeys.TRAIN)
            
            # 15. convolutional layer 
            with tf.variable_scope('layer15_a'):
                conv15_a = tf.layers.conv2d(
                  inputs=norm14_a,
                  filters=8,
                  kernel_size=[1, 1],
                  dilation_rate=(1,1),
                  padding="same",
                  activation=tf.nn.relu,
                  kernel_initializer=tf.initializers.he_uniform(),
                  name="conv15_a")
                
                # 15. batch normalization
                norm15_a = tf.layers.batch_normalization(
                        inputs=conv15_a,
                        name="norm15_a",
						training= mode==tf.estimator.ModeKeys.TRAIN)
						
            with tf.variable_scope('output_audio'):
                output_a = tf.reshape(norm15_a,[-1,norm15_a.shape[1],norm15_a.shape[2]*norm15_a.shape[3]], name="output_a")
            
            return output_a
    
    
    # visual stream
    def dnn_video(layer_name, x, audio_time, sharedWeights):        
        with tf.name_scope(layer_name):
            
            input_reshape = tf.reshape(x,[-1,x.shape[1], x.shape[2],1])
            input_video = tf.transpose(input_reshape,[0,1,3,2])
            with tf.variable_scope('dnn_speaker', reuse=sharedWeights):
                
                # 1. convolutional layer 
                with tf.variable_scope('layer1_v'):
                    conv1_v = tf.layers.conv2d(
                            inputs=input_video,
                            filters=256,
                            kernel_size=[7,1],
                            dilation_rate=(1,1),
                            padding="same",
                            activation=tf.nn.relu,
                            kernel_initializer=tf.initializers.he_uniform(),
                            name="conv1_v")
                    
                    # 1. batch normalization
                    conv1_v_norm = tf.layers.batch_normalization(
                            conv1_v,
                            name="norm1_v",
                            training= mode==tf.estimator.ModeKeys.TRAIN)
                 
                # 2. convolutional layer 
                with tf.variable_scope('layer2_v'):
                    conv2_v = tf.layers.conv2d(
                            inputs=conv1_v_norm,
                            filters=256,
                            kernel_size=[5,1],
                            dilation_rate=(1,1),
                            padding="same",
                            activation=tf.nn.relu,
                            kernel_initializer=tf.initializers.he_uniform(),
                            name="conv2_v")
                    
                    # 2. batch normalization
                    conv2_v_norm = tf.layers.batch_normalization(
                            conv2_v,
                            name="norm2_v",
                            training= mode==tf.estimator.ModeKeys.TRAIN)
                
                # 3. convolutional layer 
                with tf.variable_scope('layer3_v'):
                    conv3_v = tf.layers.conv2d(
                            inputs=conv2_v_norm,
                            filters=256,
                            kernel_size=[5,1],
                            dilation_rate=(2,1),
                            padding="same",
                            activation=tf.nn.relu,
                            kernel_initializer=tf.initializers.he_uniform(),
                            name="conv3_v")
                    
                    # 3. batch normalization
                    conv3_v_norm = tf.layers.batch_normalization(
                            conv3_v,
                            name="norm3_v",
                            training= mode==tf.estimator.ModeKeys.TRAIN)
                
                # 4. convolutional layer 
                with tf.variable_scope('layer4_v'):
                    conv4_v = tf.layers.conv2d(
                            inputs=conv3_v_norm,
                            filters=256,
                            kernel_size=[5,1],
                            dilation_rate=(4,1),
                            padding="same",
                            activation=tf.nn.relu,
                            kernel_initializer=tf.initializers.he_uniform(),
                            name="conv4_v")
                    
                    # 4. batch normalization
                    conv4_v_norm = tf.layers.batch_normalization(
                            conv4_v,
                            name="norm4_v",
                            training= mode==tf.estimator.ModeKeys.TRAIN)
                
                # 5. convolutional layer 
                with tf.variable_scope('layer5_v'):
                    conv5_v = tf.layers.conv2d(
                            inputs=conv4_v_norm,
                            filters=256,
                            kernel_size=[5,1],
                            dilation_rate=(8,1),
                            padding="same",
                            activation=tf.nn.relu,
                            kernel_initializer=tf.initializers.he_uniform(),
                            name="conv5_v")
                    
                    # 5. batch normalization
                    conv5_v_norm = tf.layers.batch_normalization(
                            conv5_v,
                            name="norm5_v",
                            training= mode==tf.estimator.ModeKeys.TRAIN)
                
                # 6. convolutional layer 
                with tf.variable_scope('layer6_v'):
                    conv6_v = tf.layers.conv2d(
                            inputs=conv5_v_norm,
                            filters=256,
                            kernel_size=[5,1],
                            dilation_rate=(16,1),
                            padding="same",
                            activation=tf.nn.relu,
                            kernel_initializer=tf.initializers.he_uniform(),
                            name="conv6_v")
                    
                    # 6. batch normalization
                    conv6_v_norm = tf.layers.batch_normalization(
                            conv6_v,
                            name="norm6_v",
                            training= mode==tf.estimator.ModeKeys.TRAIN)
                    
                # resize time axis (nearest neighbor interpolation)                   
                with tf.variable_scope('output_video'):
                    conv6_reshape = tf.transpose(conv6_v_norm, [0,1,3,2])
                    
                    conv6_resize = tf.image.resize_nearest_neighbor(
                            conv6_reshape,
                            align_corners=True,
                            size=[audio_time,conv6_reshape.shape[2]],
                            name="ResizeVideo")
                    
                    output_v = tf.reshape(conv6_resize, [-1,conv6_resize.shape[1],conv6_resize.shape[2]])
                
        return output_v
    
    # audio stream
    audio = dnn_audio("dnn_audio",aud_mix)
    
    #visual stream of two speakers (shared weights)
    with tf.variable_scope("dnn_visual"):
        s1 = dnn_video("speaker1", vid1, audio.shape[1], False)
        s2 = dnn_video("speaker2", vid2, audio.shape[1], True)
        # concatenate the two visual streams
        visual = tf.concat([s1, s2], axis=2, name="fusion_visual")
        
    # audio-visual fusionfusion
    with tf.variable_scope("fusion"):
        fusion = tf.concat([audio, visual], axis=2, name="audio_visual")
    
    # bidirectional LSTM
    with tf.variable_scope("BLSTM"):
        cell_fw = tf.nn.rnn_cell.LSTMCell(200)
        cell_bw = tf.nn.rnn_cell.LSTMCell(200)
        
        outputs, states =  tf.nn.bidirectional_dynamic_rnn(
                cell_fw=cell_fw,
                cell_bw=cell_bw,
                inputs=fusion,
                dtype=tf.float32,
                scope="blstm")
        
        output = tf.concat(outputs,2)
    
    # fully connected layer
    with tf.variable_scope('fc_layers'):
        fc1 = tf.layers.dense(
                inputs=output,
                units=600,
                activation=tf.nn.relu,
                kernel_initializer=tf.initializers.he_uniform(),
                name="fully_connected1")
        
        fc2 = tf.layers.dense(
                inputs=fc1,
                units=600,
                activation=tf.nn.relu,
                kernel_initializer=tf.initializers.he_uniform(),
                name="fully_connected2")
        
        fc3 = tf.layers.dense(
                inputs=fc2,
                units=600,
                activation=tf.nn.relu,
                kernel_initializer=tf.initializers.he_uniform(),
                name="fully_connected3")
        
    # fully connected layer for complex masks
    with tf.variable_scope('output'):
        complex_mask = tf.layers.dense(
                inputs=fc3,
                units=257*2*2,
                activation=tf.nn.sigmoid,
                name="complex_mask")
        
        complex_mask_out = tf.reshape(complex_mask, [-1,fc3.shape[1],257,2,2])
        complex_mask_out_t = tf.transpose(complex_mask_out,[0,2,1,3,4], name="output_masks")
        
    # define a name for output tensor (contains the complex masks, stacked)
    prediction_masks = tf.identity(complex_mask_out_t, name="prediction_masks")
    
    # check the shapes
    print("complex mask: ", complex_mask_out_t.shape)    
    print("predictions: ", prediction_masks.shape)   
    
    
    # calculate Loss (for both TRAIN and EVAL modes)
    with tf.name_scope('loss'):
        loss = loss_mean_squared_error(complex_mask_out_t[:,:,:,:,0], complex_mask_out_t[:,:,:,:,1], aud_mix, aud1, aud2)
        
        # add summary for tensorboard
        mse_loss_metric = tf.metrics.mean(loss)
        tf.summary.scalar('normal/mse_metric', mse_loss_metric[1])
        tf.summary.scalar('normal/loss', loss)
        tf.summary.histogram('normal/loss', loss)
       
    # define a name for the loss (needed for the logging hook)
    mse_loss = tf.identity(loss, name="mse_loss")
      
    # visualize biases and weights as histograms
    for var in tf.trainable_variables():
        tf.summary.histogram(var.name.replace(':','_'), var)
    
    # configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(learning_rate=0.00003)
        
        # update the UPDATE_OPS (needed for batch norm)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = optimizer.minimize(
                loss=loss,
                global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)
    
    # define a metric for EVAL mode
    eval_metric_ops = {
         "mse_loss": mse_loss_metric
	}

    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)#, evaluation_hooks=evaluation_hooks


'''
------------------------------------------------------------------------------
desc:          parser for TFRecords
param: 
    record:    single TFRecord
    
return:         
    audio:     input: audio mixed spectrogram [257,298,2]
    video:     input: face embeddings [2,75,512] (stacked)
    label:     label: clean spectrogram per speaker [2,257,298,2] (stacked)
------------------------------------------------------------------------------
'''
def parser(record):
    keys_to_features = {
        'audio_s1': tf.FixedLenFeature(shape=[1], dtype=tf.int64),
        'audio_s2': tf.FixedLenFeature(shape=[1], dtype=tf.int64),
        'audio_s3': tf.FixedLenFeature(shape=[1], dtype=tf.int64),
        'video_s1': tf.FixedLenFeature(shape=[1], dtype=tf.int64),
        'video_s2': tf.FixedLenFeature(shape=[1], dtype=tf.int64),
        'video_s3': tf.FixedLenFeature(shape=[1], dtype=tf.int64),
        'label_s1': tf.FixedLenFeature(shape=[1], dtype=tf.int64),
        'label_s2': tf.FixedLenFeature(shape=[1], dtype=tf.int64),
        'label_s3': tf.FixedLenFeature(shape=[1], dtype=tf.int64),
        'label_s4': tf.FixedLenFeature(shape=[1], dtype=tf.int64),
        'audio': tf.FixedLenFeature(shape=[257,298,2], dtype=tf.float32),
        'video': tf.FixedLenFeature(shape=[2,75,512], dtype=tf.float32),
        'label': tf.FixedLenFeature(shape=[2,257,298,2], dtype=tf.float32)
        
    }
    
    features = tf.parse_single_example(record, keys_to_features)

    audio = tf.cast(features['audio'], tf.float32)
    video = tf.cast(features['video'], tf.float32)
    label = tf.cast(features['label'], tf.float32)
    
    #print(train_audio.shape)
    return audio, video, label

'''
------------------------------------------------------------------------------
desc:                       input function for TRAIN mode
param: 
    train_dir_tfrecord:     dirs of train data as a list
    doshuffle:              True if you want to shuffle the data (default=True)
    batch_size:             batch_size for training (default=4)
    num_epoches:            number of epochs to train (default=1)
    
return:
    x:      input data (features) as dictionary for TRAIN
    y:      labels as dictionary for TRAIN
------------------------------------------------------------------------------
'''
def train_input_fn(train_dir_tfrecord, doshuffle=True, batch_size=4, num_epochs=1):
    filenames = []
    for t in train_dir_tfrecord:
        for dirs,_,files in os.walk(t):
            for f in files:
                filenames.append(os.path.abspath(os.path.join(dirs, f)))
    
    print("Trainingsdaten: ", len(filenames))
    random.shuffle(filenames)
    
    # define a dataset for train data
    train_dataset = tf.data.TFRecordDataset(filenames,num_parallel_reads=4)   
    
    if doshuffle:
        train_dataset = train_dataset.shuffle(buffer_size=500)
    
    train_dataset = train_dataset.map(map_func=parser, num_parallel_calls=4)
    train_dataset = train_dataset.batch(batch_size)
    train_dataset = train_dataset.repeat(num_epochs)
    train_dataset = train_dataset.prefetch(4)
    train_iterator = train_dataset.make_one_shot_iterator()
    
    train_audio, train_video, train_labels = train_iterator.get_next()
    
    x={"vid1":  train_video[:,0,:,:],
       "vid2":  train_video[:,1,:,:],
       "aud": train_audio}
    y={"aud1": train_labels[:,0,:,:,:],
       "aud2": train_labels[:,1,:,:,:]}
    
    return x, y


'''
------------------------------------------------------------------------------
desc:                       input function for EVAL mode
param: 
    eval_dir_tfrecord:      dirs of eval data as a list
    doshuffle:              True if you want to shuffle the data (default=False)
    batch_size:             batch_size for training (default=4)
    num_epoches:            number of epochs to train (default=1)
    
return:
    x:      input data (features) as dictionary for EVAL
    y:      labels as dictionary for EVAL    
------------------------------------------------------------------------------
'''
def eval_input_fn(eval_dir_tfrecord, doshuffle=False, batch_size=4, num_epochs=1): 
    
    filenames = []
    for e in eval_dir_tfrecord:
        for dirs,_,files in os.walk(e):
            for f in files:
                filenames.append(os.path.abspath(os.path.join(dirs, f)))
    print("Evaluierungsdaten: ", len(filenames))
    
    # define a dataset for eval data
    eval_dataset = tf.data.TFRecordDataset(filenames,num_parallel_reads=4)
        
    if doshuffle:
        eval_dataset = eval_dataset.shuffle(buffer_size=500)
    eval_dataset = eval_dataset.map(map_func=parser, num_parallel_calls=4)
    eval_dataset = eval_dataset.batch(batch_size)
    eval_dataset = eval_dataset.repeat(num_epochs)
    eval_dataset = eval_dataset.prefetch(4)
    eval_iterator = eval_dataset.make_one_shot_iterator()
    
    eval_audio, eval_video, eval_labels = eval_iterator.get_next()
    
    x={"vid1":  eval_video[:,0,:,:],
       "vid2":  eval_video[:,1,:,:],
       "aud": eval_audio}
    y={"aud1": eval_labels[:,0,:,:,:],
       "aud2": eval_labels[:,1,:,:,:]}
    
    return x, y


'''
------------------------------------------------------------------------------
desc:              basic function, defines:
                        - train and eval data dir as lists
                        - run configuration (gpu options, save_checkpoints_steps)
                        - custom estimator (classifier)
                        - logging hook during training
                        - train und eval specification with lambda operator
param: 
    model_dir:     dir where the model should be saved

return:            -    
------------------------------------------------------------------------------
'''
def traineval(model_dir):
    # define the train data dirs as a list (absolute path)
    train_dir_tfrecord = ["D:/BA_LipRead/03_DeepNet/data_tfrecord_train/",          
                          "D:/BA_LipRead/03_DeepNet/data_tfrecord_train_1/",        
                          "D:/BA_LipRead/03_DeepNet/data_tfrecord_train_2/",        
                          "C:/temp/BA_LipRead/03_DeepNet/data_tfrecord_train_2_1/", 
                          "G:/BA_LipRead/data_tfrecord_train_2_2/",                 
                          "G:/BA_LipRead/data_tfrecord_train_3/",                   
                          "D:/BA_LipRead/03_DeepNet/data_tfrecord_train_4/",        
                          "G:/BA_LipRead/data_tfrecord_train_5/"]          

    # define the train data dirs as a list (absolute path)         
    eval_dir_tfrecord = ["D:/BA_LipRead/03_DeepNet/data_tfrecord_eval/",            
                         "D:/BA_LipRead/03_DeepNet/data_tfrecord_eval_1/",          
                         "C:/temp/BA_LipRead/03_DeepNet/data_tfrecord_eval_2/",     
                         "C:/temp/BA_LipRead/03_DeepNet/data_tfrecord_eval_3/",     
                         "G:/BA_LipRead/data_tfrecord_eval_4/"]                     
    
    # option use dynamic gpu memory
    gpu_options = tf.GPUOptions(allow_growth=True)
    sess_config = tf.ConfigProto(gpu_options=gpu_options) 

    # define run config, save checkpoints after 200 train steps
    run_config = tf.estimator.RunConfig(session_config = sess_config,save_checkpoints_steps=200)
    # create the custom estimator
    avspeech_classifier = tf.estimator.Estimator(
            model_fn=cnn_model_fn, 
            model_dir=model_dir, 
            config=run_config) 
    
    # set up a logging hook, logging tensor "mse_loss" every 10 steps of training
    tensors_to_log = {"error": "mse_loss"}
    
    logging_hook = tf.train.LoggingTensorHook(
            tensors=tensors_to_log, every_n_iter=10) 
    
    # define the train specification with lambda operator
    train_spec = tf.estimator.TrainSpec(
            input_fn=lambda: train_input_fn(train_dir_tfrecord, 
                                            doshuffle=True, 
                                            batch_size=2,
                                            num_epochs=200),
            hooks=[logging_hook])

    # define the eval specification with lambda operator
    # throttle sec: after how many seconds the model should be evaluated
    eval_spec = tf.estimator.EvalSpec(
            input_fn=lambda: eval_input_fn(eval_dir_tfrecord,
                                           doshuffle=False,
                                           batch_size=2,
                                           num_epochs=1),
                                           throttle_secs=100)
    
    # start training
    tf.estimator.train_and_evaluate(avspeech_classifier, train_spec, eval_spec)
 
    
'''
------------------------------------------------------------------------------
desc:       main function
param: 
    -
------------------------------------------------------------------------------
'''
def main(unused_argv):
    # define model dir
    model_dir="./logs/model_AVSpeech/"
    
    traineval(model_dir)
        
if __name__ == "__main__":
    tf.app.run()

        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    