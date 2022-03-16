#https://github.com/kritiyer/AngioNet
from glob import glob

from IPython.core.debugger import set_trace
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Conv2D, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l1_l2

#from tkinter import filedialog
import os
import numpy as np
#import pydicom as dicom
#import pylibjpeg
import cv2
#import matplotlib.pyplot as plt
from PIL import Image
from model import DeepLabV3, unet

pgm_orgpath = '/Users/Boyce/OneDrive/Documents/ICA/ICA/Database_134_Angiograms'
pgm_gtpath='/Users/Boyce/OneDrive/Documents/ICA/ICA/ground truth'
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
plt.imshow(sample_or,cmap='gray', vmin=0, vmax=255)
plt.show()
sample_or.shape

sample = readpgm(gt_path, (256,256))

plt.imshow(sample, cmap='gray', vmin=0, vmax=255)
plt.show()

def load_paths(path,test_size=0.2):
    images=os.listdir('/Users/Boyce/OneDrive/Documents/ICA/ICADatabase_134_Angiograms')
    masks=os.listdir('/Users/Boyce/OneDrive/Documents/ICA/ICA/ground truth')
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



path='/Users/Boyce/OneDrive/Documents/ICA/ICA/'
(X_train, y_train), (X_val, y_val), (X_test, y_test)=load_paths(path)

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
        x = np.zeros((self.batch_size,) + self.img_size + (3,), dtype="float32")
        for j, path in enumerate(batch_input_img_paths):
            img = readpgm(path, self.img_size, use_rgb=True)
            x[j] = img/255
        y = np.zeros((self.batch_size,) + self.img_size + (1,), dtype="uint8")
        for j, path in enumerate(batch_target_img_paths):
            mask = readpgm(path, self.img_size)            
            y[j] = np.expand_dims(mask, -1)/255
        return x, y

H=256
W=256
batch_size=8
img_size=(H,W)
train_gen=ICA(batch_size, img_size, X_train, y_train)
val_gen=ICA(batch_size, img_size, X_val, y_val)
img_size

i=5
batch_img, batch_mask=train_gen[i]
batch_img.shape
batch_mask.shape


def AngioNet(L1=0., L2= 0., DL_weights=None):
    inputs = Input(shape=(H,W,3))
    activation_func = None
    X1 = Conv2D(1, (5,5), strides=(1, 1), padding='same', dilation_rate=(1, 1), activation=activation_func,
                use_bias=False, data_format="channels_last")(inputs)
#(filters, kernal_size)
    X2 = Conv2D(1, (3,3), strides=(1, 1), padding='same', dilation_rate=(1, 1), activation=activation_func,
                use_bias=False, data_format="channels_last")(X1)
    X3 = Conv2D(14, (5,5), strides=(1, 1), padding='same', dilation_rate=(1, 1), activation=activation_func,
                use_bias=False, data_format="channels_last")(X2)
    X4 = Conv2D(14, (5,5), strides=(1, 1), padding='same', dilation_rate=(1, 1), activation=activation_func,
                use_bias=False, data_format="channels_last")(X3)
    X5 = Conv2D(14, (5,5), strides=(1, 1), padding='same', dilation_rate=(1, 1), activation=activation_func,
                use_bias=False, data_format="channels_last")(X4)
    X6 = Conv2D(1, (3,3), strides=(1, 1), padding='same', dilation_rate=(1, 1), activation='tanh',
                use_bias=False, data_format="channels_last")(X5)
    X7 = concatenate([X6, X6, X6], axis=3)

    unsharp_mask_model = Model(inputs=inputs, outputs=X7)
    unsharp_mask_model._name = "Preprocessing_Network"

    deeplab_model = DeepLabV3.DeeplabV3Plus(image_size=(H,W), num_classes=3)
    for layer in deeplab_model.layers:
        layer.kernel_regularizer = l1_l2(l1 = L1, l2=L2)
########greggregegrgrg
    combined_inputs = Input(shape=(H, W, 3))
    unsharp_mask_img = unsharp_mask_model(combined_inputs)
    deeplab_img = deeplab_model(unsharp_mask_img)
    model = Model(combined_inputs, deeplab_img)

    return model

model = AngioNet(L1=0, L2=0, DL_weights=None)
model.summary()


binary_crossentropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.1),
    loss=binary_crossentropy,
    metrics=["accuracy"],
)

history = model.fit(train_gen, validation_data=val_gen, epochs=50)
history.history.keys()

plt.plot(history.history["loss"])
plt.title("Training Loss")
plt.ylabel("loss")
plt.xlabel("epoch")
plt.show()

plt.plot(history.history["accuracy"])
plt.title("Training Accuracy")
plt.ylabel("accuracy")
plt.xlabel("epoch")
plt.show()

plt.plot(history.history["val_loss"])
plt.title("Validation Loss")
plt.ylabel("val_loss")
plt.xlabel("epoch")
plt.show()

plt.plot(history.history["val_accuracy"])
plt.title("Validation Accuracy")
plt.ylabel("val_accuracy")
plt.xlabel("epoch")
plt.show()

import matplotlib.pyplot as plt
acc=history.history['accuracy']
val_acc=history.history['val_accuracy']
loss=history.history['accuracy']
val_loss=history.history['val_loss']
epochs=range(len(acc))
plt.figure()
plt.plot(epochs, acc,'b', label='Validation accuracy')
plt.plot(epochs, val_acc,'b', label='Validation accuracy')
plt.plot(epochs, loss,'g', label='Training loss')
plt.plot(epochs,val_loss,'y', label='Validation accuracy')
plt.title('Training and validation accuracy and loss')
plt.show()