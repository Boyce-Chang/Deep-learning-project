#from tkinter import filedialog
from dataclasses import dataclass
import os
from sys import path
import numpy as np
# from sklearn import tree
import pydicom 
# import pylibjpeg
import cv2
import matplotlib.pyplot as plt
from PIL import Image



def Dcm2png(file_path):
    img = pydicom.dcmread(file_path)
    img = img.pixel_array
    img[img<175]=0
    return img


in_path = '/Users/chenshirley/Downloads/1095041/11-M.dcm'
sample = Dcm2png(in_path)
cv2.imshow('aa',sample)
cv2.waitKey(0)
cv2.destroyAllWindows
# plt.title('11-M')
# plt.show()

in_path2 = '/Users/chenshirley/Downloads/1095041/11-C.dcm'
sample2 = Dcm2png(in_path2)
sample2.show()
plt.imshow(sample2)
plt.title('11-C')
plt.show()



