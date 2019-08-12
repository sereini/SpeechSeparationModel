"""
Exports the embeddings of a directory of images as numpy arrays.
Following structure:
    D:\images:
        folder1:
            img_0
            ...
            img_74
        folder2:
            img_0
            ...
            img_74
            
Output:
embeddings.npy -- Embeddings as np array (with names "folder1", "folder2", etc.)

Use --is_aligned False, if your images aren't already pre-aligned
Use --image_batch to dictacte how many images to load in memory at a time.


Started with export_embeddings.py from Charles Jekel, and modified the program
to export the face embeddings for the audio-visual speech separation model. The
pretrained model is from David Sandberg's facenet repository:
    https://github.com/davidsandberg/facenet
export_embedding.py from same project:
    https://github.com/davidsandberg/facenet/tree/master/contributed


Ensure you have set the PYTHONPATH for the pretrained facenet (3.):
    https://github.com/davidsandberg/facenet/wiki/Validate-on-LFW
Execution:
    python export_FaceEmbedding.py models\20180402-114759\20180402-114759.pb D:\images --is_aligned False --image_size 160 --gpu_memory_fraction 0.5 --image_batch 75


Sereina Scherrer 2019
"""

# MIT License
#
# Copyright (c) 2016 David Sandberg
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
from scipy import misc
import tensorflow as tf
import numpy as np
import sys
import os
import argparse
import facenet
import align.detect_face
import re
import glob

from six.moves import xrange

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [atoi(c) for c in re.split(r'(\d+)', text)]

def main(args):
    train_set = facenet.get_dataset(args.data_dir)
    image_list, label_list = facenet.get_image_paths_and_labels(train_set)
    
    # sort the image:s img_0 ... img_74
    image_list.sort(key=natural_keys)
    
    # fetch the classes (labels as strings) exactly as it's done in get_dataset
    path_exp = os.path.expanduser(args.data_dir)
    classes = [path for path in os.listdir(path_exp) \
               if os.path.isdir(os.path.join(path_exp, path))]
    classes.sort()
    # get the label strings
    label_strings = [name for name in classes if \
       os.path.isdir(os.path.join(path_exp, name))]

    # define path to save the embeddings
    dirs = ["./emb/embeddings_AVspeech/"]
    for d in dirs:
        if not os.path.exists(d):
            os.makedirs(d)
            print("Folder created:", d)
            
    with tf.Graph().as_default():

        with tf.Session() as sess:

            # Load the model
            facenet.load_model(args.model_dir)

            # Get input and output tensors
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

            # Run forward pass to calculate embeddings
            nrof_images = len(image_list)
            print('Number of images: ', nrof_images)
            batch_size = args.image_batch
            if nrof_images % batch_size == 0:
                nrof_batches = nrof_images // batch_size
            else:
                nrof_batches = (nrof_images // batch_size) + 1
            print('Number of batches: ', nrof_batches)
            embedding_size = embeddings.get_shape()[1]
            emb_array = np.zeros((nrof_images, embedding_size))
            start_time = time.time()

            for i in range(nrof_batches):
                if i == nrof_batches -1:
                    n = nrof_images
                else:
                    n = i*batch_size + batch_size
                # Get images for the batch
                if args.is_aligned is True:
                    images = facenet.load_data(image_list[i*batch_size:n], False, False, args.image_size)
                else:
                    images = load_and_align_data(image_list[i*batch_size:n], args.image_size, args.margin, args.gpu_memory_fraction)
                feed_dict = { images_placeholder: images, phase_train_placeholder:False }
                # Use the facenet model to calcualte embeddings
                embed = sess.run(embeddings, feed_dict=feed_dict)
                emb_array[i*batch_size:n, :] = embed
                
                # export the embedding
                s = dirs[0] + label_strings[i] + ".npy" 
                np.save(s, embed)
                
                print('Completed batch', i+1, 'of', nrof_batches)

            run_time = time.time() - start_time
            print('Run time: ', run_time)
            print('Time per video: ',run_time/nrof_batches)



def load_and_align_data(image_paths, image_size, margin, gpu_memory_fraction):


    print('Creating networks and loading parameters')
    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = align.detect_face.create_mtcnn(sess, None)

    nrof_samples = len(image_paths)
    img_list = [None] * nrof_samples
    for i in xrange(nrof_samples):
        print(image_paths[i])
        img = misc.imread(os.path.expanduser(image_paths[i]))
        
        aligned = misc.imresize(img, (image_size, image_size), interp='bilinear')
        prewhitened = facenet.prewhiten(aligned)
        img_list[i] = prewhitened   
        
        # uncomment if you want to save the aligned images
        '''f = os.path.basename(image_paths[i])
        #print(f)
        tmp_folder = re.split(r'\\', image_paths[i])
        tmp_f = tmp_folder[-2]
        d = "./aligned/" + tmp_f + "/"
        if not os.path.exists(d):
            os.makedirs(d)
            print("Folder created:", d)
        
        misc.imsave(d + f, aligned)'''
        
    images = np.stack(img_list)
    return images

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('model_dir', type=str,
        help='Directory containing the meta_file and ckpt_file')
    parser.add_argument('data_dir', type=str,
        help='Directory containing images. If images are not already aligned and cropped include --is_aligned False.')
    parser.add_argument('--is_aligned', type=str,
        help='Is the data directory already aligned and cropped?', default=True)
    parser.add_argument('--image_size', type=int,
        help='Image size (height, width) in pixels.', default=160)
    parser.add_argument('--margin', type=int,
        help='Margin for the crop around the bounding box (height, width) in pixels.',
        default=44)
    parser.add_argument('--gpu_memory_fraction', type=float,
        help='Upper bound on the amount of GPU memory that will be used by the process.',
        default=1.0)
    parser.add_argument('--image_batch', type=int,
        help='Number of images stored in memory at a time. Default 75.',
        default=75)

    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
