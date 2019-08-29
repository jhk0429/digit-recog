import numpy as np
import pandas as pd 
import keras
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from PIL import Image
import warnings
import time
from glob import glob
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np, pandas as pd, os
import xml.etree.ElementTree as ET 
import matplotlib.pyplot as plt, zipfile 
from PIL import Image 
from glob import glob

all_images=os.listdir("C:/Users/Alvin Kim/Downloads/Gan/all-dogs/all-dogs/")

DogsOnly = False

ROOT = 'C:/Users/Alvin Kim/Downloads/Gan/'
IMAGES = os.listdir(ROOT + 'all-dogs/all-dogs/')
breeds = os.listdir(ROOT + 'annotation/Annotation/') 

idxIn = 0; namesIn = []
imagesIn = np.zeros((25000,64,64,3))

# CROP WITH BOUNDING BOXES TO GET DOGS ONLY
if DogsOnly:
    for breed in breeds:
        for dog in os.listdir(ROOT+'annotation/Annotation/'+breed):
            try: img = Image.open(ROOT+'all-dogs/all-dogs/'+dog+'.jpg') 
            except: continue           
            tree = ET.parse(ROOT+'annotation/Annotation/'+breed+'/'+dog)
            root = tree.getroot()
            objects = root.findall('object')
            for o in objects:
                bndbox = o.find('bndbox') 
                xmin = int(bndbox.find('xmin').text)
                ymin = int(bndbox.find('ymin').text)
                xmax = int(bndbox.find('xmax').text)
                ymax = int(bndbox.find('ymax').text)
                w = np.min((xmax - xmin, ymax - ymin))
                img2 = img.crop((xmin, ymin, xmin+w, ymin+w))
                img2 = img2.resize((64,64), Image.ANTIALIAS)
                imagesIn[idxIn,:,:,:] = np.asarray(img2)
                #if idxIn%1000==0: print(idxIn)
                namesIn.append(breed)
                idxIn += 1
    idx = np.arange(idxIn)
    np.random.shuffle(idx)
    imagesIn = imagesIn[idx,:,:,:]
    namesIn = np.array(namesIn)[idx]
    
# RANDOMLY CROP FULL IMAGES
else:
    IMAGES = np.sort(IMAGES)
    np.random.seed(810)
    x = np.random.choice(np.arange(20579),10000)
    np.random.seed(None)
    for k in range(len(x)):
        img = Image.open(ROOT + 'all-dogs/all-dogs/' + IMAGES[x[k]])
        w = img.size[0]; h = img.size[1];
        if (k%2==0)|(k%3==0):
            w2 = 100; h2 = int(h/(w/100))
            a = 18; b = 0          
        else:
            a=0; b=0
            if w<h:
                w2 = 64; h2 = int((64/w)*h)
                b = (h2-64)//2
            else:
                h2 = 64; w2 = int((64/h)*w)
                a = (w2-64)//2
        img = img.resize((w2,h2), Image.ANTIALIAS)
        img = img.crop((0+a, 0+b, 64+a, 64+b))    
        imagesIn[idxIn,:,:,:] = np.asarray(img)
        namesIn.append(IMAGES[x[k]])
        #if idxIn%1000==0: print(idxIn)
        idxIn += 1
    
# DISPLAY CROPPED IMAGES
        """
x = np.random.randint(0,idxIn,25)
for k in range(5):
    plt.figure(figsize=(15,3))
    for j in range(5):
        plt.subplot(1,5,j+1)
        img = Image.fromarray( imagesIn[x[k*5+j],:,:,:].astype('uint8') )
        plt.axis('off')
        if not DogsOnly: plt.title(namesIn[x[k*5+j]],fontsize=11)
        else: plt.title(namesIn[x[k*5+j]].split('-')[1],fontsize=11)
        plt.imshow(img)
    plt.show()
    """
    
IMG_SIZE = tf.keras.Input((12288,))
IMG_SIZE_2 = tf.keras.Input((10000,))
NOISE_SIZE = 10000
#BATCH_SIZE = 256 # orig gives ~7.24
#BATCH_SIZE = 512 # gives ~7.25
#BATCH_SIZE = 128 # gives 7.22594
#BATCH_SIZE = 128 # gives ~7.222
BATCH_SIZE = 64

def discriminatorFunction():
    input_layer = tf.keras.layers.Dense(12288, activation='sigmoid')(IMG_SIZE_2) 
    input_layer = tf.keras.layers.Reshape((2,12288,1))(tf.keras.layers.concatenate([IMG_SIZE,input_layer]))
    discriminator = tf.keras.layers.Conv2D(filters = 1, kernel_size=[2,1],use_bias=False, name = 'layer_1')(input_layer)
    out = tf.keras.layers.Flatten()(discriminator)
    return out

print("Discriminator")
model = discriminatorFunction()
model_discriminator = tf.keras.Model([IMG_SIZE,IMG_SIZE_2], model)
model_discriminator.get_layer('layer_1').trainable = False
model_discriminator.get_layer('layer_1').set_weights([np.array([[[[-1.0 ]]],[[[1.0]]]])])
model_discriminator.summary()
model_discriminator.compile(optimizer='adam', loss='binary_crossentropy')

################################################
def GeneratorFunction(noise_shape=(NOISE_SIZE,)):
    input_layer = tf.keras.Input(noise_shape)
    generated = tf.keras.layers.Dense(12288, activation='linear')(input_layer)
# COMPILE
    model = tf.keras.models.Model(inputs=input_layer,outputs = [generated,tf.keras.layers.Reshape((10000,))(input_layer)])
    model.summary()
    return model
print("Generator")
model_generator = GeneratorFunction(noise_shape=(NOISE_SIZE,))

#####################################################
train_y = (imagesIn[:10000,:,:,:]/255.).reshape((-1,12288))
train_X = np.zeros((10000,10000))
for i in range(10000): train_X[i,i] = 1
zeros = np.zeros((10000,12288))

# ---------------------
#  Train Discriminator
# ---------------------

# orig
lr = 0.5
# Let's play with lr
#lr = 0.3 # gives ~7.251
for k in range(5):
    LR_Scheduler = tf.keras.callbacks.LearningRateScheduler(lambda x: lr)
    h = model_discriminator.fit([zeros,train_X], train_y, epochs = 10,batch_size = BATCH_SIZE, callbacks=[LR_Scheduler], verbose=0)
    print('Epoch',(k+1)*10,'/50 - loss =',h.history['loss'][-1] )
    if h.history['loss'][-1]<0.533: lr = 0.1

del train_X, train_y, imagesIn
#############################################################

print('Discriminator Recalls from Memory Dogs')    
for k in range(5):
    plt.figure(figsize=(15,3))
    for j in range(5):
        xx = np.zeros((10000))
        xx[np.random.randint(10000)] = 1
        plt.subplot(1,5,j+1)
        img = model_discriminator.predict([zeros[0,:].reshape((-1,12288)),xx.reshape((-1,10000))]).reshape((-1,64,64,3))
        img = Image.fromarray( (255*img).astype('uint8').reshape((64,64,3)))
        plt.axis('off')
        plt.imshow(img)
    plt.show()
    
###############################
model_discriminator.trainable = False #discriminator is not trainable for GANs
z = tf.keras.Input(shape=(NOISE_SIZE,))
img = model_generator(z)
real = model_discriminator(img)

# COMPILE GAN
gan = tf.keras.models.Model(z, real)
gan.get_layer('model_1').get_layer('layer_1').set_weights([np.array([[[[-1 ]]],[[[255.]]]])])
gan.compile(optimizer=tf.keras.optimizers.Adam(5), loss='mean_squared_error')

# DISPLAY ARCHITECTURE
print("Model created based on Discriminator and Generator")
gan.summary()