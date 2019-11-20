import numpy as np
import matplotlib.pyplot as plt
import os
import tensorflow as tf
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import DepthwiseConv2D 
from PIL import Image 

# global variables
imgW = 256  # image width
imgH = 256  # image height
imgCh = 1   # channels 
nPx = imgW * imgH
# load a single image
def loadImage(path):
    global imgW, imgH
    imgSize = (imgW, imgH) # (W, H)
    img = Image.open(path)
    img = img.resize(imgSize).convert('L') 
    imgData = np.array(img) / 255.0
    return imgData  

imgMainDir = '101_ObjectCategories/' 
imgCategory = 'airplanes/' 
img = loadImage(imgMainDir + imgCategory + 'image_0002.jpg')


kernel_size = 3  # set the filter size of Gaussian filter
kernel_weights = [[0., 0., 0.], [0., 0.8, 0.], [0., 0., 0.]]

imgCh = 1  # the number of input channels
kernel_weights = np.expand_dims(kernel_weights, axis=-1)
kernel_weights = np.repeat(kernel_weights, imgCh, axis=-1) # apply the same filter on all the input channels
kernel_weights = np.expand_dims(kernel_weights, axis=-1)  # for shape compatibility reasons
#print(kernel_weights)

X = np.reshape(img, [1, imgW, imgH, imgCh])

model = tf.keras.Sequential()
nLayers = 1
preLayer = X
for i in range(nLayers):    
    g_layer = DepthwiseConv2D(kernel_size, use_bias=False, padding='same', input_shape = (imgW, imgH, imgCh))
    preLayer = g_layer(preLayer)
    g_layer.set_weights([kernel_weights])
    g_layer.trainable = False  # the weights should not change during training
    model.add(g_layer)
model.summary()
opt = tf.keras.optimizers.Adam(learning_rate=1.1, beta_1=0.1, beta_2=0.29, amsgrad=True)
model.compile(loss='mse', optimizer=opt, metrics=['mse','mae'], verbose=0)

Y = model.predict(X, verbose=0)
imgY = np.reshape(Y, [imgW, imgH])

imgOrg = Image.fromarray(np.uint8(img * 255.0))
imgOut = Image.fromarray(np.uint8(imgY * 255.0))
f = plt.figure()
f.add_subplot(1, 2, 1)
plt.imshow(imgOrg, cmap='gray', vmin=0, vmax=255)
f.add_subplot(1, 2, 2)
plt.imshow(imgOut, cmap='gray', vmin=0, vmax=255)
plt.show(block=True)
