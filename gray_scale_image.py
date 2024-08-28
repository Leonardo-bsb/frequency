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
# cv2.imshow('Image',image[3:35,2:24])
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# np.max(image[3:35,2:24],axis=0)
# np.max(image[3:35,2:24],axis=1)
# import pandas as pd 
# DF=pd.DataFrame(image)
# DF.to_csv('a')
a=image[14:32,1:17]
a_row=a.flatten(order='C')
a_row=255-a_row


import numpy as np
from matplotlib import pyplot as plt
duration=1

sample_rate=1024
N=a_row.size
np.pad(a_row,(0,512-N))
t = np.linspace(0, duration, a_row.size, endpoint=False)

plt.stem(t,a_row)
plt.xlabel("t-->"),plt.ylabel("a")
plt.title("Line 1")
plt.show()

from scipy.fft import rfft, rfftfreq

# Note the extra 'r' at the front
yf = rfft(a_row)
xf = rfftfreq(N, 1 / sample_rate)
plt.subplot(2,1,1)
plt.stem(xf, 10*np.log10(np.abs(yf)))
plt.xlabel("f-->"),plt.ylabel("y em dB")
plt.title("Frequencies em dB")
plt.subplot(2,1,2)
plt.stem(xf, np.abs(yf))
plt.xlabel("f-->"),plt.ylabel("y")
plt.title("Frequencies Linear")
plt.show()