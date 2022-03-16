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

pgm_orgpath = '/home/shirley/Downloads/ICA/Database_134_Angiograms'
pgm_gtpath='/home/shirley/Downloads/ICA/ground truth'
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
        
H = 256
W = 256
batch_size=8
img_size=(H,W)
train_gen=ICA(batch_size, img_size, X_train, y_train)
val_gen=ICA(batch_size, img_size, X_val, y_val)
img_size

i=5
batch_img, batch_mask=train_gen[i]
batch_img.shape

batch_mask.shape

def convolution_block(
    block_input,
    num_filters=256,
    kernel_size=3,
    dilation_rate=1,
    padding="same",
    use_bias=False,
):
    x = layers.Conv2D(
        num_filters,
        kernel_size=kernel_size,
        dilation_rate=dilation_rate,
        padding="same",
        use_bias=use_bias,
        kernel_initializer=keras.initializers.HeNormal(),
    )(block_input)
    x = layers.BatchNormalization()(x)
    return tf.nn.relu(x)


def DilatedSpatialPyramidPooling(dspp_input):
    dims = dspp_input.shape
    x = layers.AveragePooling2D(pool_size=(dims[-3], dims[-2]))(dspp_input)
    x = convolution_block(x, kernel_size=1, use_bias=True)
    out_pool = layers.UpSampling2D(
        size=(dims[-3] // x.shape[1], dims[-2] // x.shape[2]), interpolation="bilinear",
    )(x)

    out_1 = convolution_block(dspp_input, kernel_size=1, dilation_rate=1)
    out_6 = convolution_block(dspp_input, kernel_size=3, dilation_rate=6)
    out_12 = convolution_block(dspp_input, kernel_size=3, dilation_rate=12)
    out_18 = convolution_block(dspp_input, kernel_size=3, dilation_rate=18)

    x = layers.Concatenate(axis=-1)([out_pool, out_1, out_6, out_12, out_18])
    output = convolution_block(x, kernel_size=1)
    return output

def DeeplabV3Plus(image_size, num_classes):
    model_input = keras.Input(shape=(H, W, 3))
    resnet50 = keras.applications.ResNet50(
        weights="imagenet", include_top=False, input_tensor=model_input
    )
    x = resnet50.get_layer("conv4_block6_2_relu").output
    x = DilatedSpatialPyramidPooling(x)

    input_a = layers.UpSampling2D(
        size=(image_size[0] // 4 // x.shape[1], image_size[0] // 4 // x.shape[2]),
        interpolation="bilinear",
    )(x)
    input_b = resnet50.get_layer("conv2_block3_2_relu").output
    input_b = convolution_block(input_b, num_filters=48, kernel_size=1)

    x = layers.Concatenate(axis=-1)([input_a, input_b])
    x = convolution_block(x)
    x = convolution_block(x)
    x = layers.UpSampling2D(
        size=(image_size[0] // x.shape[1], image_size[0] // x.shape[2]),
        interpolation="bilinear",
    )(x)
    model_output = layers.Conv2D(num_classes, kernel_size=(1, 1), padding="same")(x)
    return keras.Model(inputs=model_input, outputs=model_output)

if __name__ == '__main__':
    model = DeeplabV3Plus(image_size=img_size, num_classes=3)
    model.summary()


    loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.1),
        loss=loss,
        metrics=["accuracy"],
    )


    history = model.fit(train_gen, validation_data=val_gen, epochs=10)
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