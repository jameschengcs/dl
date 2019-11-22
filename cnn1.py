# Small CNN 
import numpy as np
import matplotlib.pyplot as plt
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential  #用來啟動 NN
from tensorflow.keras.layers import Conv2D  # Convolution Operation
from tensorflow.keras.layers import MaxPooling2D # Pooling
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense # Fully Connected Networks
from PIL import Image 

# global variables
imgW = 256  # image width
imgH = 256  # image height
imgCh = 1   # channels 
train_ratio = 0.9  # the ratio of the amount of training data
test_ratio = 1.0 - train_ratio # the ratio of the amount of test data

# load a single image
def loadImage(path):
    global imgW, imgH, imgCh
    imgSize = (imgW, imgH) # (W, H)
    img = Image.open(path)
    img = img.resize(imgSize, resample = Image.BILINEAR).convert('L') 
    imgData = np.array(img) / 255.0
    imgData = np.reshape(imgData, [imgW, imgH, imgCh])
    return imgData  
   
# load a set of images from a dirctroy    
def loadImageSet(path):
    global imgW, imgH
    imgNames = os.listdir(path)
    imgShape = (imgW, imgH, imgCh)
    n = len(imgNames)
    imgSetShape = [n] + list(imgShape)
    imgSet = np.zeros(imgSetShape)
    i = 0
    for imgName in imgNames:
        imgPath = path + imgName
        imgSet[i] = loadImage(imgPath)
        i += 1    
    print(imgSet.shape)
    return imgSet

# Loading data
imgMainDir = '101_ObjectCategories/'
YNames = ['airplanes', 'Motorbikes', 'schooner']

XinA = [loadImageSet(imgMainDir + YNames[i] + '/') for i in range(3)]
Xin = [XinA[i][:int(len(XinA[i]) * train_ratio)] for i in range(3)]
XTin = [XinA[i][int(len(XinA[i]) * train_ratio):] for i in range(3)]
# Training dataset
X = np.concatenate(Xin)
Y = np.concatenate([np.full((Xin[i].shape[0]), i) for i in range(3)])
# Test dataset
XT = np.concatenate(XTin)
YT = np.concatenate([np.full((XTin[i].shape[0]), i) for i in range(3)])

print(X.shape)
print(Y.shape)
print(XT.shape)
print(YT.shape)


# initializing CNN
initializer = tf.keras.initializers.Ones()
#initializer = tf.keras.initializers.RandomNormal()
model = Sequential()  
model.add(Conv2D(8, 3, 3, input_shape = (imgW, imgH, imgCh), kernel_initializer = initializer, activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))
# Second convolutional layer
model.add(Conv2D(4, 3, 3, kernel_initializer = initializer, activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))

# Flatten
model.add(Flatten())

# Fully Connected Networks
model.add(Dense(6, kernel_initializer = initializer, activation = 'relu'))
model.add(Dense(3, kernel_initializer = initializer, activation = 'softmax'))

# Compiling
#opt = tf.keras.optimizers.Adadelta()
opt = tf.keras.optimizers.Adam(learning_rate=0.05, beta_1=0.01, beta_2=0.29, amsgrad=True)
#opt = tf.keras.optimizers.SGD(learning_rate=0.1, momentum=0.0, nesterov=False)
model.compile(loss = tf.keras.losses.sparse_categorical_crossentropy, optimizer=opt, metrics = ['sparse_categorical_accuracy'])


model.fit(X, Y, epochs=10, batch_size=10,  shuffle = True, verbose=1)

model.evaluate(XT,  YT, verbose=2)
