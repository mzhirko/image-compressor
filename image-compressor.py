import math
import random
import numpy as np
import matplotlib.pylab as plt

from PIL import Image


# def normalize_matrix counts adaptive learning step
def adaptive_learning_step(matrix):
    tmp = np.dot(matrix, np.transpose(matrix))
    s = 0
    for i_f in range(len(tmp)):
        s += tmp[i_f]

    return 1.0 / (s * 10)


# def normalize_matrix normalizes given matrix
def normalize_matrix(matrix):
    for i_f in range(len(matrix[0])):
        s = 0
        for j_f in range(len(matrix)):
            s += matrix[j_f][i_f] * matrix[j_f][i_f]
        s = math.sqrt(s)
        for j_f in range(len(matrix)):
            matrix[j_f][i_f] = matrix[j_f][i_f] / s


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

p = N // 3

print((N * L) / ((N + L) * p + 2))

# adding axis to array (for easier matrix transposing)
Xq = np.expand_dims(Xq, axis=1)
X_out = np.expand_dims(X_out, axis=1)
X_delta = np.expand_dims(X_delta, axis=1)

# fielding weight matrices with random values
random.seed()
for i in range(N):
    W_first[i] = np.random.uniform(-1, 1, p)

W_second = np.transpose(W_first)

iteration_counter = 0
E = 10000000
while E > 1000:
    E = 0

    for k in range(L):
        # counting Y
        Y = np.dot(Xq[k], W_first)

        # Counting X'
        X_out[k] = np.dot(Y, W_second)

        # Counting delta X
        X_delta[k] = X_out[k] - Xq[k]

        # Counting W'
        alpha_second = adaptive_learning_step(Y)
        # print(alpha_second)
        W_second = W_second - alpha_second * np.dot(np.transpose(Y), X_delta[k])

        # Counting W
        alpha_first = adaptive_learning_step(Xq[k])
        # print(alpha_first)
        W_first = W_first - alpha_first * np.dot(np.dot(np.transpose(Xq[k]), X_delta[k]),  np.transpose(W_second))

        # normalizing matrices
        normalize_matrix(W_first)
        normalize_matrix(W_second)

        Y = np.dot(Xq[k], W_first)
        X_out[k] = np.dot(Y, W_second)
        X_delta[k] = X_out[k] - Xq[k]

        # counting error
        for i in range(N):
            E += X_delta[k][0][i] * X_delta[k][0][i]

    # increasing iteration counter

    iteration_counter += 1
    print("Iteration number: {}, error {}".format(iteration_counter, E))

print("Final iteration count: {}, final error {}".format(iteration_counter, E))
