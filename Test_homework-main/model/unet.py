from glob import glob

from IPython.core.debugger import set_trace
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import tensorflow as tf
from tensorflow.keras import layers

#from tkinter import filedialog
import os
import numpy as np
#import pydicom as dicom
#import pylibjpeg
import cv2
#import matplotlib.pyplot as plt
from PIL import Image

pgm_orgpath = '/Users/chenshirley/Desktop/ICA'
pgm_gtpath='/Users/chenshirley/Desktop/ICA'
org_path = os.path.join(pgm_orgpath,'1.pgm')
gt_path = os.path.join(pgm_gtpath,'1_gt.pgm')

def readpgm(name, img_size=None, use_rgb=False):
    with open(name) as f:
        lines = f.readlines()

    # Ignores commented lines
    for l in list(lines):
        if l[0] == '#':
            lines.remove(l)

    # Makes sure it is ASCII format (P2)
    assert lines[0].strip() == 'P2' 

    # Converts data to a list of integers
    data = []
    for line in lines[1:]:
        data.extend([int(c) for c in line.split()])

    data =(np.array(data[3:]),(data[1],data[0]),data[2])
    img = np.array(np.reshape(data[0],data[1]),dtype = np.uint8)
    if img_size is not None:        
        img = Image.fromarray(img)
        if use_rgb:
            img = img.convert('RGB')
        img = img.resize(img_size)
        img = np.array(img)
        
    return img



sample_or = readpgm(org_path, (256,256))
plt.imshow(sample_or, cmap='gray', vmin=0, vmax=255)
plt.show()
sample_or.shape

sample = readpgm(gt_path, (256,256))

plt.imshow(sample, cmap='gray', vmin=0, vmax=255)
plt.show()

def load_paths(path,test_size=0.2):
    images=os.listdir('/home/shirley/Downloads/ICA/Database_134_Angiograms')
    masks=os.listdir('/home/shirley/Downloads/ICA/ground truth')
    images = [m for m in images if '.pgm' in m]
    masks = [m for m in masks if '.pgm' in m]
    images = sorted(images, key=lambda x:int(x.split('.')[0]))
    masks = sorted(masks, key=lambda x:int(x.split('_')[0]))
    
    for name in masks:
        if 'ini' in name:
           masks.remove(name)
    print(len(masks))
    
    images =  [os.path.join(path, f"Database_134_Angiograms/{name}") for name in images]
    masks  =  [os.path.join(path, f"ground truth/{name}") for name in masks]
    
    X_train, X_test, y_train, y_test = train_test_split(images, masks, test_size=test_size, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=test_size, random_state=42)
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)



path='/home/shirley/Downloads/ICA'
(X_train, y_train), (X_val, y_val), (X_test, y_test)=load_paths(path)

H = 256
W = 256

from tensorflow import keras
from tensorflow.keras.preprocessing.image import load_img
class ICA(keras.utils.Sequence):
    """Helper to iterate over the data (as Numpy arrays)."""

    def __init__(self, batch_size, img_size, input_img_paths, target_img_paths):
        self.batch_size = batch_size
        self.img_size = img_size
        self.input_img_paths = input_img_paths
        self.target_img_paths = target_img_paths

    def __len__(self):
        return len(self.target_img_paths) // self.batch_size

    def __getitem__(self, idx):
        """Returns tuple (input, target) correspond to batch #idx."""
        i = idx * self.batch_size
        batch_input_img_paths = self.input_img_paths[i : i + self.batch_size]
        batch_target_img_paths = self.target_img_paths[i : i + self.batch_size]
        x = np.zeros((self.batch_size,) + self.img_size + (1,), dtype="float32")
        for j, path in enumerate(batch_input_img_paths):
            img = readpgm(path, self.img_size)
            x[j] = np.expand_dims(img, -1)/255
        y = np.zeros((self.batch_size,) + self.img_size + (1,), dtype="uint8")
        for j, path in enumerate(batch_target_img_paths):
            mask = readpgm(path, self.img_size)
            y[j] = np.expand_dims(mask, -1)/255
        return x, y

H = 512
W = 512
batch_size=8
img_size=(H,W)
train_gen=ICA(batch_size, img_size, X_train, y_train)
val_gen=ICA(batch_size, img_size, X_val, y_val)
img_size

i=5
batch_img, batch_mask=train_gen[i]
batch_img.shape

batch_mask.shape

plt.imshow(batch_img[0,:,:,:], cmap='gray')
plt.show()

plt.imshow(batch_mask[0,:,:,:].astype(np.uint8), cmap='gray')
plt.show()

print(batch_mask.max(), batch_mask.min(), batch_mask.dtype, np.bincount(batch_mask.flatten()))

def get_model(img_size, num_classes):
    inputs = keras.Input(shape=img_size + (1,))

    ### [First half of the network: downsampling inputs] ###

    # Entry block
    x = layers.Conv2D(32, 3, strides=2, padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    # Blocks 1, 2, 3 are identical apart from the feature depth.
    for filters in [64, 128, 256]:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = layers.Conv2D(filters, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    ### [Second half of the network: upsampling inputs] ###

    for filters in [256, 128, 64, 32]:
        x = layers.Activation("relu")(x)
        x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.UpSampling2D(2)(x)

        # Project residual
        residual = layers.UpSampling2D(2)(previous_block_activation)
        residual = layers.Conv2D(filters, 1, padding="same")(residual)
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual
        # Add a per-pixel classification layer
    outputs = layers.Conv2D(num_classes, 1, activation="sigmoid")(x)

    # Define the model
    model = keras.Model(inputs, outputs)
    return model

if __name__ == '__main__':

    from keras import layers
    img_size=(H,W)
    num_classes=1
    model=get_model(img_size, num_classes)
    model.summary()

    tf.keras.optimizers.Adam(learning_rate=0.001,beta_1=0.9,beta_2=0.999,epsilon=1e-07,amsgrad=False,name="Adam")
    tf.keras.optimizers.Adadelta(learning_rate=0.001, rho=0.95, epsilon=1e-07, name="Adadelta")
    tf.keras.optimizers.SGD(learning_rate=0.001, momentum=0.5, nesterov=False, name="SGD")
    tf.keras.optimizers.RMSprop(learning_rate=0.001,rho=0.9,momentum=0.0,epsilon=1e-07,centered=False,name="RMSprop")


    model.compile(
    optimizer="RMSprop", loss='binary_crossentropy',metrics=['acc'],
    )
    callbacks=[
    keras.callbacks.ModelCheckpoint('ica_seg.h5', save_best_only=True)
    ]
    if os.path.exists('./ica_seg.h5'):
        os.remove('./ica_seg.h5')
    epochs=50
    history=model.fit(train_gen, epochs=epochs, validation_data=val_gen, callbacks=callbacks)



    history.history.keys()



    import matplotlib.pyplot as plt
    acc=history.history['acc']
    val_acc=history.history['val_acc']
    loss=history.history['acc']
    al_loss=history.history['val_loss']
    epochs=range(len(acc))
    plt.figure()
    plt.plot(epochs, acc,'b', label='Validation accuracy')
    plt.plot(epochs, val_acc,'b', label='Validation accuracy')
    plt.plot(epochs, loss,'g', label='Training loss')
    plt.plot(epochs,val_loss,'y', label='Validation accuracy')
    plt.title('Training and validation accuracy and loss')
    plt.show()

