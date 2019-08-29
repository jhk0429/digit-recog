import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import urllib
import tarfile
import os
from tqdm import tqdm_notebook
import matplotlib.pyplot as plt
%matplotlib inline
from imageio import imread, imsave, mimsave
from PIL import Image
import glob
import shutil
import xml.etree.ElementTree as ET # for parsing XML

root_images="C:/Users/Alvin Kim/Downloads/Gan/all-dogs/all-dogs/"
root_annots="C:/Users/Alvin Kim/Downloads/Gan/Annotation/Annotation/"

all_images=os.listdir("C:/Users/Alvin Kim/Downloads/Gan/all-dogs/all-dogs/")

breeds = glob.glob('C:/Users/Alvin Kim/Downloads/Gan/Annotation/Annotation/*')
annotation=[]

for b in breeds:
    annotation+=glob.glob(b+"/*")

breed_map={}
for annot in annotation:
    breed=annot.split("\\")[-2]
    index=breed.split("-")[0]
    breed_map.setdefault(index,breed)
    
    
def bounding_box(image):
    bpath=root_annots+str(breed_map[image.split("_")[0]])+"/"+str(image.split(".")[0])
    tree = ET.parse(bpath)
    root = tree.getroot()
    objects = root.findall('object')
    for o in objects:
        bndbox = o.find('bndbox') # reading bound box
        xmin = int(bndbox.find('xmin').text)
        ymin = int(bndbox.find('ymin').text)
        xmax = int(bndbox.find('xmax').text)
        ymax = int(bndbox.find('ymax').text)
        
    return (xmin,ymin,xmax,ymax)

def get_crop_image(image):
    bbox=bounding_box(image)
    im=Image.open(os.path.join(root_images,image))
    im=im.crop(bbox)
    return im

""" PLOTTING
plt.figure(figsize=(10,10))
for i,image in enumerate(all_images):
    im=get_crop_image(image)
    plt.subplot(3,3,i+1)
    plt.axis("off")
    plt.imshow(im)    
    if(i==8):
        break
"""

path = 'C:/Users/Alvin Kim/Downloads/Gan/all-dogs'
dataset = 'all-dogs'
data_path = os.path.join(path, dataset)
images = glob.glob(os.path.join(data_path, '*.*')) 

z_dim = 1000

WIDTH = 64
HEIGHT = 64

OUTPUT_DIR = 'samples_dogs'
if not os.path.exists(OUTPUT_DIR):
    os.mkdir(OUTPUT_DIR)

GEN_DIR = 'generated_dogs'
if not os.path.exists(GEN_DIR):
    os.mkdir(GEN_DIR)
    
X = tf.placeholder(dtype=tf.float32, shape=[None, HEIGHT, WIDTH, 3], name='X')
noise = tf.placeholder(dtype=tf.float32, shape=[None, z_dim], name='noise')
is_training = tf.placeholder(dtype=tf.bool, name='is_training')

def lrelu(x, leak=0.2):
    return tf.maximum(x, leak * x)

def sigmoid_cross_entropy_with_logits(x, y):
    return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, labels=y)

def discriminator(image, reuse=None, is_training=is_training):
    momentum = 0.9
    with tf.variable_scope('discriminator', reuse=reuse):
        h0 = lrelu(tf.layers.conv2d(image, kernel_size=5, filters=64, strides=2, padding='same'))
        
        h1 = tf.layers.conv2d(h0, kernel_size=5, filters=128, strides=2, padding='same')
        h1 = lrelu(tf.layers.batch_normalization(h1, training=is_training, momentum=momentum))
        
        h2 = tf.layers.conv2d(h1, kernel_size=5, filters=256, strides=2, padding='same')
        h2 = lrelu(tf.layers.batch_normalization(h2, training=is_training, momentum=momentum))
        
        h3 = tf.layers.conv2d(h2, kernel_size=5, filters=512, strides=2, padding='same')
        h3 = lrelu(tf.layers.batch_normalization(h3, training=is_training, momentum=momentum))

        h4 = tf.layers.flatten(h3)
        h4 = tf.layers.dense(h4, units=1)
        return tf.nn.sigmoid(h4), h4
    
def generator(z, is_training=is_training):
    momentum = 0.9
    with tf.variable_scope('generator', reuse=None):
        d = 4
        h0 = tf.layers.dense(z, units=d * d * 512)
        h0 = tf.reshape(h0, shape=[-1, d, d, 512])
        h0 = tf.nn.relu(tf.layers.batch_normalization(h0, training=is_training, momentum=momentum))
        
        h1 = tf.layers.conv2d_transpose(h0, kernel_size=5, filters=256, strides=2, padding='same')
        h1 = tf.nn.relu(tf.layers.batch_normalization(h1, training=is_training, momentum=momentum))
        
        h2 = tf.layers.conv2d_transpose(h1, kernel_size=5, filters=128, strides=2, padding='same')
        h2 = tf.nn.relu(tf.layers.batch_normalization(h2, training=is_training, momentum=momentum))
        
        h3 = tf.layers.conv2d_transpose(h2, kernel_size=5, filters=64, strides=2, padding='same')
        h3 = tf.nn.relu(tf.layers.batch_normalization(h3, training=is_training, momentum=momentum))
        
        h4 = tf.layers.conv2d_transpose(h3, kernel_size=5, filters=3, strides=2, padding='same', activation=tf.nn.tanh, name='g')
        return h4
    
    