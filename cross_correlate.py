import cv2
import numpy as np
grey_level=255
height,width=40,200
blank_image=grey_level* np.ones((height,width),dtype=np.uint8)
font = cv2.FONT_HERSHEY_SIMPLEX
org = (0, 30)
fontScale = 1
color = (0)
thickness = 1
image = cv2.putText(blank_image, 'abcdefghijkl', org, font, 
                   fontScale, color, thickness, cv2.LINE_AA)
image1=cv2.putText(blank_image, 'bcdefghijkl', org, font, 
                   fontScale, color, thickness, cv2.LINE_AA)
# cv2.imshow('Image',image[3:35,2:24])
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# np.max(image[3:35,2:24],axis=0)
# np.max(image[3:35,2:24],axis=1)
# import pandas as pd 
# DF=pd.DataFrame(image)
# DF.to_csv('a')

cv2.imshow('Image',image1[3:35,2:24])
cv2.waitKey(0)
cv2.destroyAllWindows()
a=image[14:32,1:17]
a_row=a.flatten(order='C')
a_row=255-a_row
np.pad(a_row,(0,512-a_row.size))

b=image1[14:32,1:17]
b_row=a.flatten(order='C')
b_row=255-a_row
np.pad(b_row,(0,512-b_row.size))

import numpy as np
from matplotlib import pyplot as plt
duration=1

sample_rate=1024


t = np.linspace(0, duration, a_row.size, endpoint=False)

plt.stem(t,a_row)
plt.xlabel("t-->"),plt.ylabel("a")
plt.title("Line 1")
plt.show()

X1=np.fft.fft(a_row)
X2=np.fft.fft(b_row)
Xcorr=np.fft.ifft(X1*np.conj(X2))

fig,ax=plt.subplots()
ax.plot(np.real(Xcorr))
ax.set_xlabel('Sample')
ax.set_ylabel('Cross-correlation')
ax.set_title('Cross-correlation of a_row and b_row')
plt.show()

corr1=np.correlate(a_row,b_row,'full')

fig,ax=plt.subplots()
ax.plot(np.abs(corr1))
ax.set_xlabel('Sample-1')
ax.set_ylabel('Cross-correlation numpy')
ax.set_title('Cross-correlation of x1 and x2')
plt.show()