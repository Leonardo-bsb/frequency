import numpy as np
from matplotlib import pyplot as plt

duration=1
sample_rate=64
N=16
t = np.linspace(0, duration, N, endpoint=False)
y=np.zeros(N)
y[8]=1
y[9]=1
y[12]=1
y[13]=1
plt.stem(t,y)
plt.xlabel("t-->"),plt.ylabel("y")
plt.title("Line 1")
plt.show()

from scipy.fft import rfft, rfftfreq

# Note the extra 'r' at the front
yf = rfft(y)
xf = rfftfreq(N, 1 / sample_rate)
plt.subplot(2,1,1)
plt.plot(xf, 10*np.log10(np.abs(yf)))
plt.xlabel("t-->"),plt.ylabel("y em dB")
plt.title("Frequencies em dB")
plt.subplot(2,1,2)
plt.stem(xf, np.abs(yf))
plt.xlabel("t-->"),plt.ylabel("y")
plt.title("Frequencies Linear")
plt.show()


plt.show()