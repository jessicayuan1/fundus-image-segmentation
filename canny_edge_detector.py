import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
 
img = cv.imread('./imageDB/007-1789-100.jpg', cv.IMREAD_GRAYSCALE)
edges = cv.Canny(img, 23, 25)

plt.subplot(121),plt.imshow(img,cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(edges,cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])

plt.show()