#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from PIL import Image, ImageFont, ImageDraw
from sys import platform
from random import choice
from string import ascii_letters
import os
import sys
import random
from urllib.request import urlretrieve
import tarfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from shutil import copyfile
from random import seed
from random import sample
from tensorflow.keras import Input
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K

if len(sys.argv) != 3:
    sys.exit("Invalid args.")

if sys.argv[1] == 'g':
    NOISE_TYPE = 'gaussian'
    NOISE_PARAM = (NOISE_TYPE, 25)
elif sys.argv[1] == 'p':
    NOISE_TYPE = 'poisson'
    NOISE_PARAM = (NOISE_TYPE, 50)
elif sys.argv[1] == 'b':
    NOISE_TYPE = 'multiplicative_bernoulli'
    NOISE_PARAM = (NOISE_TYPE, 0.5)
elif sys.argv[1] == 't':
    NOISE_TYPE = 'text'
    NOISE_PARAM = (NOISE_TYPE, 0.2)
else:
    sys.exit("Invalid noise arg.")

# dir paths
if sys.argv[2] == 'kodak':
    train_dir = 'imgs/kodak_dir'
    num_imgs = 24
    img_multiplier = 100
    if not os.path.isdir('imgs'):
        os.mkdir('imgs')
    if not os.path.isdir(train_dir):
        os.makedirs(train_dir)
    ## DOWNLOAD KODAK DATA
    for i in range(1, 25):
        imgname = 'kodim%02d.png' % i
        url = "http://r0k.us/graphics/kodak/kodak/" + imgname
        print ('Downloading', url)
        urlretrieve(url, os.path.join('imgs/kodak_dir', imgname))
    print ('Kodak validation set successfully downloaded.')

elif sys.argv[2] == 'bsds300':
    train_dir = 'imgs/BSDS300/images/train'
    num_imgs = 200 
    img_multiplier = 12
    if not os.path.isdir('imgs'):
        os.mkdir('imgs')
    ## DOWNLOAD BSD300 DATA
    #pip install python3-wget
    tarpath = './BSDS300-images.tgz'
    my_tar = tarfile.open(tarpath)
    my_tar.extractall('imgs') # specify which folder to extract to
    my_tar.close()

plot_dir = 'plots'
if not os.path.isdir(plot_dir):
    os.mkdir(plot_dir)
model_dir = 'models'
if not os.path.isdir(model_dir):
    os.mkdir(model_dir)


class NoisyDataset():
    
    def __init__(self, root_dir, crop_size=128, train_noise_model=('gaussian', 50), clean_targ=False):
        """
        root_dir: Path of image directory
        crop_size: Crop image to given size
        clean_targ: Use clean targets for training
        """
        self.root_dir = root_dir
        self.crop_size = crop_size
        self.clean_targ = clean_targ
        self.noise = train_noise_model[0]
        self.noise_param = train_noise_model[1]
        self.imgs = os.listdir(root_dir)

    def _random_crop_to_size(self, img):
        
        w, h = img.size
        assert w >= self.crop_size and h >= self.crop_size, 'Cannot be croppped. Invalid size'

        i = np.random.randint(0, w - self.crop_size)
        j = np.random.randint(0, h - self.crop_size)

        cropped_img = img.crop((i, j, i+self.crop_size, j+self.crop_size))
        return cropped_img
    
    def _add_gaussian_noise(self, image):
        """
        Added only gaussian noise
        """
        w, h = image.size
        c = len(image.getbands())
        
        std = np.random.uniform(0, self.noise_param)
        _n = np.random.normal(0, std, (h, w, c))
        noisy_image = np.array(image) + _n
        noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
        return {'image':Image.fromarray(noisy_image), 'mask': None, 'use_mask': False}

    def _add_poisson_noise(self, image):
        """
        Added poisson Noise
        """
        ld = np.random.uniform(0, self.noise_param)
        factor = 1 / ld
        img = np.array(image)
        img = img.astype(np.uint16)
        noisy_image = np.random.poisson(img * factor) / float(factor)
        noisy_image = np.clip(noisy_image, 0, 255, noisy_image).astype(np.uint8)
        return {'image':Image.fromarray(noisy_image), 'mask': None, 'use_mask': False}

    def _add_m_bernoulli_noise(self, image):
        """
        Multiplicative bernoulli
        """
        sz = np.array(image).shape[0]
        prob_ = random.uniform(0, self.noise_param)
        mask = np.random.choice([0, 1], size=(sz, sz), p=[prob_, 1 - prob_])
        mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
        noisy_image = np.multiply(image, mask).astype(np.uint8)
        return {'image':Image.fromarray(noisy_image), 'mask':mask.astype(np.uint8), 'use_mask': True}


    def _add_text_overlay(self, image):
        """
        Add text overlay to image
        """
        assert self.noise_param < 1, 'Text parameter should be probability of occupancy'

        w, h = image.size
        c = len(image.getbands())

        if platform == 'linux':
            serif = '/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf'
        else:
            serif = 'Times New Roman.ttf'

        text_img = image.copy()
        text_draw = ImageDraw.Draw(text_img)
        mask_img = Image.new('1', (w, h))
        mask_draw = ImageDraw.Draw(mask_img)

        max_occupancy = np.random.uniform(0, self.noise_param)
        max_text_count = 10
        text_count = 0
        def get_occupancy(x):
            y = np.array(x, np.uint8)
            return np.sum(y) / y.size
        while 1:
            font = ImageFont.truetype(serif, np.random.randint(16, 21))
            length = np.random.randint(10, 25)
            chars = ''.join(choice(ascii_letters) for i in range(length))
            color = tuple(np.random.randint(0, 255, c))
            pos = (np.random.randint(0, w), np.random.randint(0, h))
            text_draw.text(pos, chars, color, font=font)
            # Update mask and check occupancy
            mask_draw.text(pos, chars, 1, font=font)
            if get_occupancy(mask_img) > max_occupancy:
                break
            if text_count > max_text_count:
                break
            text_count += 1
            print(text_count)
        
        return {'image':text_img, 'mask':None, 'use_mask': False}

    def corrupt_image(self, image):
        
        if self.noise == 'gaussian':
            return self._add_gaussian_noise(image)
        elif self.noise == 'poisson':
            return self._add_poisson_noise(image)
        elif self.noise == 'multiplicative_bernoulli':
            return self._add_m_bernoulli_noise(image)
        elif self.noise == 'text':
            return self._add_text_overlay(image)
        else:
            raise ValueError('No such image corruption supported')

    def __getitem__(self, index):
        """
        Read a image, corrupt it and return it
        """
        img_path = os.path.join(self.root_dir, self.imgs[index])
        image = Image.open(img_path).convert('RGB')

        if self.crop_size > 0:
            image = self._random_crop_to_size(image)

        source_img_dict = self.corrupt_image(image)

        if self.clean_targ:
            target = image
        else:
            _target_dict = self.corrupt_image(image)
            target = _target_dict['image']

        if source_img_dict['use_mask']:
            return [source_img_dict['image'], source_img_dict['mask'], target]
        else:
            return [source_img_dict['image'], 0, target]

    def __len__(self):
        return len(self.imgs)


if NOISE_TYPE != 'text':
    train_ds = NoisyDataset(train_dir, crop_size=128, train_noise_model=NOISE_PARAM, clean_targ=False)
    x = []
    y = []
    for i in range(num_imgs):
        # modify here to generate 5k pairs
        for _ in range(img_multiplier):
            xt, _, yt = train_ds.__getitem__(i)
            x.append(np.array(xt))
            y.append(np.array(yt))
     
    x = np.array(x)
    y = np.array(y)
    print(NOISE_TYPE+" data prep done.")
    print(type(x))
    print(y.shape)
    
    np.save('x', x)
    np.save('y', y)
    
    load_x = np.load('x.npy').astype('float32')
    load_y = np.load('y.npy').astype('float32')
else: # text data processing does not work on xsede somehow. package difference maybe?
    if sys.argv[2] == 'kodak':
        load_x = np.load('text_kodak_x.npy').astype('float32')
        load_y = np.load('text_kodak_y.npy').astype('float32')
    elif sys.argv[2] == 'bsds300':
        load_x = np.load('text_bsds300_x.npy').astype('float32')
        load_y = np.load('text_bsds300_y.npy').astype('float32')
 
# normalize
load_x /= 255
load_y /= 255

epochs = 100
batch_size = 32

print(load_x.shape)
print(load_y.shape)

indices = np.arange(load_x.shape[0])
np.random.shuffle(indices)
load_x = load_x[indices]
load_y = load_y[indices]

train_num = int(load_x.shape[0] * 0.7)

train_x = load_x[:train_num]
val_x = load_x[train_num:]
print(train_x.shape)
print(val_x.shape)

train_y = load_y[:train_num]
val_y = load_y[train_num:]
print(train_y.shape)
print(val_y.shape)

input_tensor = Input(shape=(128, 128, 3))   #input shape has to be 2**n  64, 128, 256 ... 

enc_conv0 = layers.Conv2D(48, 3, padding='same', activation='relu')(input_tensor)
enc_conv1 = layers.Conv2D(48, 3, padding='same', activation='relu')(enc_conv0)
max_pool1 = layers.MaxPool2D(2,  padding='same', strides=2)(enc_conv1)

enc_conv2 = layers.Conv2D(48, 3, padding='same', activation='relu')(max_pool1)
max_pool2 = layers.MaxPool2D(2,  padding='same', strides=2)(enc_conv2)

enc_conv3 = layers.Conv2D(48, 3, padding='same', activation='relu')(max_pool2)
max_pool3 = layers.MaxPool2D(2,  padding='same', strides=2)(enc_conv3)

enc_conv4 = layers.Conv2D(48, 3, padding='same', activation='relu')(max_pool3)
max_pool4 = layers.MaxPool2D(2,  padding='same', strides=2)(enc_conv4)

enc_conv5 = layers.Conv2D(48, 3, padding='same', activation='relu')(max_pool4)
max_pool5 = layers.MaxPool2D(2,  padding='same', strides=2)(enc_conv5)
enc_conv6 = layers.Conv2D(48, 3, padding='same', activation='relu')(max_pool5)

upsample5 = layers.UpSampling2D(2)(enc_conv6)

concate5 = layers.concatenate([upsample5, max_pool4], axis=-1)
dec_conv5a = layers.Conv2D(96, 3, padding='same', activation='relu')(concate5)
dec_conv5b = layers.Conv2D(96, 3, padding='same', activation='relu')(dec_conv5a)

upsample4 = layers.UpSampling2D(2)(dec_conv5b)
#print(K.int_shape(upsample4))
#print(K.int_shape(input_tensor))
concate4 = layers.concatenate([upsample4, max_pool3], axis=-1)
dec_conv4a = layers.Conv2D(96, 3, padding='same', activation='relu')(concate4)
dec_conv4b = layers.Conv2D(96, 3, padding='same', activation='relu')(dec_conv4a)

upsample3 = layers.UpSampling2D(2)(dec_conv4b)
concate3 = layers.concatenate([upsample3, max_pool2], axis=-1)
dec_conv3a = layers.Conv2D(96, 3, padding='same', activation='relu')(concate3)
dec_conv3b = layers.Conv2D(96, 3, padding='same', activation='relu')(dec_conv3a)

upsample2 = layers.UpSampling2D(2)(dec_conv3b)
concate2 = layers.concatenate([upsample2, max_pool1], axis=-1)
dec_conv2a = layers.Conv2D(96, 3, padding='same', activation='relu')(concate2)
dec_conv2b = layers.Conv2D(96, 3, padding='same', activation='relu')(dec_conv2a)

upsample1 = layers.UpSampling2D(2)(dec_conv2b)
concate1 = layers.concatenate([upsample1, input_tensor], axis=-1)
dec_conv1a = layers.Conv2D(64, 3, padding='same', activation='relu')(concate1)
dec_conv1b = layers.Conv2D(32, 3, padding='same', activation='relu')(dec_conv1a)
## MISSING LAYER
dec_conv1c = layers.Conv2D(3, 3, padding='same')(dec_conv1b)

last_layer = layers.LeakyReLU(alpha=0.1)(dec_conv1c)

model = Model(input_tensor, last_layer)
print(model.summary())

# Using adam as specified in the paper, beta1 = 0.9, beta2 = 0.99, e = 10^-8
opt = optimizers.Adam(lr=1e-4, beta_1=0.9, beta_2=0.99)

model.compile(loss='mean_squared_error',
              optimizer=opt,
              metrics=['mean_squared_error'])

history = model.fit(train_x, train_y,
                   batch_size=batch_size,
                   epochs=epochs,
                   validation_data=(val_x, val_y),
                   shuffle=True)

# prepare loss acc plots
mse = history.history['mean_squared_error'] # use acc instead
val_mse = history.history['val_mean_squared_error']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(mse) + 1)

plt.clf()
plt.plot(epochs, mse, 'bo', label = 'Training mse')
plt.plot(epochs, val_mse, 'b', label = 'Validation mse')
plt.title('Training and validation mse')
plt.xlabel('Epochs')
plt.ylabel('mse')
plt.legend()
plt.savefig(os.path.join(plot_dir, NOISE_TYPE+'_'+sys.argv[2]+'_mse_plot.png'))
# plt.show()

plt.clf()
plt.plot(epochs, loss, 'bo', label = 'Training loss')
plt.plot(epochs, val_loss, 'b', label = 'Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig(os.path.join(plot_dir, NOISE_TYPE+'_'+sys.argv[2]+'_loss_plot.png'))
# plt.show()

#save model
model_name = NOISE_TYPE+'_'+sys.argv[2]+'_model.h5'
model_path = os.path.join(model_dir, model_name)
model.save(model_path)
print('Saved trained model at %s ' % model_path)
print(NOISE_TYPE+'_'+sys.argv[2]+" model done.")

