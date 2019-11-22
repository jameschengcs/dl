# Simple Image I/O by PIL and numpy
import numpy as np
from PIL import Image 
import matplotlib.pyplot as plt
imgMainDir = '101_ObjectCategories/' 
imgCategory = 'airplanes/' 
imgSize = (256, 256) # (W, H)
img = Image.open(imgMainDir + imgCategory + 'image_0002.jpg')
img = img.resize(imgSize, resample = Image.BILINEAR).convert("L")
imgData = np.asarray(img) / 255.0
print(imgData.shape, imgData.dtype)
print(imgData[0])
img2 = Image.fromarray(np.uint8(imgData * 255.0))
plt.imshow(img2, cmap='gray')
