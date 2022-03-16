from grpc import insecure_channel
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image

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
    images=os.listdir('/Users/Boyce/OneDrive/Documents/ICA/ICA/Database_134_Angiograms')
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


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias = False), 
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace = True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias = False), 
        )

def forward(self, x):
    return self.conv(x)

class UNET(nn.Module):
    def __init__(
        self,  in_channels = 3, out_channels = 1, features = [64, 128, 256, 512],
        ):
        super(UNET, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # 161 x 161, output: 160x160

        # down part of unet
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # up part of unet
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(
                    feature*2, feature, kernel_size=2, stride=2,
                )
            )
            self.ups.append(DoubleConv(feature*2, feature))

        self.bottleneck = DoubleConv(feature[-1], feature[-1]*2)
        self.final_conv = nn.Conv2d(feature[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

            x = self.bottleneck(x)
            skip_connections = skip_connections[::-1]


            for idx in range(0, len(self.ups), 2):
                x = self.ups[idx](x)
                skip_connections = skip_connections[idx//2]

                if x.shape !=skip_connections.shape:
                    x = TF.resize(x, size = skip_connections.shpae[2:])
                concat_skip = torch.cat((skip_connections, x), dim =1)
                x = self.ups[idx+1](concat_skip)

            return self.final_conv(x)
def test():
    x = torch.randn((3,1,256,256))
    model = UNET(in_channels=1, out_channels=1)
    preds = model(x)
    print(preds.shape)
    print(x.shape)
    assert preds.shape == x.shape

test()


