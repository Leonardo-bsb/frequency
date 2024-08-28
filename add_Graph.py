import cv2
import numpy as np
grey_level=255
height,width=10,10
blank_image=grey_level* np.ones((height,width),dtype=np.uint8)
blank_image[5,5]=100
blank_image[5,6]=100
blank_image[6,5]=100
blank_image[8,8]=100
print(blank_image)

for i in range (height-1):
    for j in range(width-1):
        print(str(i)+'_'+str(j))

