# Load a model
import numpy as np
import matplotlib.pyplot as plt
import os
import tensorflow as tf
import time

from tensorflow.keras.models import load_model
from PIL import Image 

# global variables
imgW = 256  # image width
imgH = 256  # image height
imgCh = 1   # channels 

# load a single image
def loadImage(path):
    global imgW, imgH, imgCh
    imgSize = (imgW, imgH) # (W, H)
    img = Image.open(path)
    img = img.resize(imgSize, resample = Image.BILINEAR).convert('L') 
    imgData = np.array(img) / 255.0
    imgData = np.reshape(imgData, [imgW, imgH, imgCh])
    return imgData  
   

timeS = time.time()
# Loading data

model = load_model('caltech_best.h5')

Xr = np.concatenate([ [loadImage('starlux.jpg')], [loadImage('CBR.jpg')], [loadImage('ship.jpg')]])
#Xr = np.concatenate([[loadImage(imgMainDir + 'airplanes/image_0015.jpg')], [loadImage(imgMainDir + 'Motorbikes/image_0012.jpg')], [loadImage(imgMainDir + 'car_side/image_0017.jpg')]])
print (Xr.shape)
Yr = np.array(model.predict(Xr))
np.set_printoptions(precision=5, suppress = True, floatmode = 'fixed')
print(Yr)

timeE = time.time()
print('Time: ', timeE - timeS)
