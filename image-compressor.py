import math
import random
import numpy as np
import matplotlib.pylab as plt

from PIL import Image

# loading image
image = Image.open("mrz3.jpg")
pix = image.load()

# setting image parameters
S = 3
c_max = 255
H = image.height
W = image.width
n = m = 4
N = n * m * S
L = int(H / n * W / m)
Xq = []

# divide on squares
for h in range(0, H, n):
    for w in range(0, W, m):

        Xqhw = np.empty(N)
        # implementation of one square
        for j in range(n):
            for k in range(m):
                for i in range(S):  # rgb
                    Xqhw[i + S * (j + k * n)] = 2 * pix[j + h, k + w][i] / c_max - 1

        Xq.append(Xqhw)

# setting variables
p = N // 2
e = p * 150

# initialize matrices
W_first = np.empty((N, p))
W_second = np.empty((p, N))
X_out = np.empty((L, N))
X_delta = np.empty((L, N))