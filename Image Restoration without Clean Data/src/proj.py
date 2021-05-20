import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
import os
from shutil import copyfile
from random import seed
from random import sample
from tensorflow.keras import Input
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K

batch_size = 32
epochs = 3

load_x = np.load('x.npy').astype('float32')
load_y = np.load('y.npy').astype('float32')
load_x /= 255
load_y /= 255

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
# Cannot find  e = 10^-8
opt = optimizers.Adam(learning_rate=1e-4, beta_1=0.9, beta_2=0.99)

model.compile(loss='mean_squared_error',
              optimizer=opt,
              metrics=['mse'])
# TODO: accuracy as metric?

history = model.fit(train_x, train_y,
                   batch_size=batch_size,
                   epochs=epochs,
                   validation_data=(val_x, val_y),
                   shuffle=True)

#print(history.history)
mse = history.history['mse'] # use acc instead
val_mse = history.history['val_mse']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(mse) + 1)

plt.clf()
plt.plot(epochs, mse, 'bo', label = 'Training mse')
plt.plot(epochs, val_mse, 'b', label = 'Validation mse')
plt.title('Training and validation MSE')
plt.xlabel('Epochs')
plt.ylabel('MSE')
plt.legend()
plt.savefig('binary_acc_v2')
# plt.show()

plt.clf()
plt.plot(epochs, loss, 'bo', label = 'Training loss')
plt.plot(epochs, val_loss, 'b', label = 'Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('binary_loss_v2')
# plt.show()

"""
#save model
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'binary_classification_trained_model_v2.h5'
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_path = os.path.join(save_dir, model_name)
model.save(model_path)
print('Saved trained model at %s ' % model_path)
"""