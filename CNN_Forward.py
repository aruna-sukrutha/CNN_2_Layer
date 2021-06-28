import numpy as np
import h5py
import matplotlib.pyplot as plt
%matplotlib inline
plt.rcParams['figure.figsize'] = (5.0, 4.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'
%load_ext autoreload
%autoreload 2
np.random.seed(1)

def zero_pad(X, pad):
    X_pad = np.pad(X, ((0, 0), (pad, pad), (pad, pad), (0, 0)), 'constant', constant_values=0)
    return X_pad

def conv_single_step(a_slice_prev, W, b):
    s = np.multiply(a_slice_prev, W) + b
    Z = np.sum(s)
    return Z

def conv_forward(A_prev, W, b, hparameters):
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
    (f, f, n_C_prev, n_C) = W.shape
    stride = hparameters['stride']
    pad = hparameters['pad']
    n_H = int((n_H_prev - f + 2 * pad) / stride) + 1
    n_W = int((n_W_prev - f + 2 * pad) / stride) + 1
    Z = np.zeros((m, n_H, n_W, n_C))
    A_prev_pad = zero_pad(A_prev, pad)
    for i in range(m):                                 # loop over the batch of training examples
        a_prev_pad = A_prev_pad[i]                     # Select ith training example's padded activation
        for h in range(n_H):                           # loop over vertical axis of the output volume
            for w in range(n_W):                       # loop over horizontal axis of the output volume
                for c in range(n_C):                   # loop over channels (= #filters) of the output volume
                    vert_start = h * stride
                    vert_end = vert_start + f
                    horiz_start = w * stride
                    horiz_end = horiz_start + f
                    a_slice_prev = a_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :]
                    Z[i, h, w, c] = conv_single_step(a_slice_prev, W[...,c], b[...,c])
    assert(Z.shape == (m, n_H, n_W, n_C))
    cache = (A_prev, W, b, hparameters)
    return Z, cache

def pool_forward(A_prev, hparameters, mode = "max"):
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
    f = hparameters["f"]
    stride = hparameters["stride"]
    n_H = int(1 + (n_H_prev - f) / stride)
    n_W = int(1 + (n_W_prev - f) / stride)
    n_C = n_C_prev
    A = np.zeros((m, n_H, n_W, n_C))
    for i in range(m):                           # loop over the training examples
        for h in range(n_H):                     # loop on the vertical axis of the output volume
            for w in range(n_W):                 # loop on the horizontal axis of the output volume
                for c in range (n_C):            # loop over the channels of the output volume
                    vert_start = h * stride
                    vert_end = vert_start + f
                    horiz_start = w * stride
                    horiz_end = horiz_start + f
                    a_prev_slice = A_prev[i, vert_start:vert_end, horiz_start:horiz_end, c]
                    if mode == "max":
                        A[i, h, w, c] = np.max(a_prev_slice)
                    elif mode == "average":
                        A[i, h, w, c] = np.mean(a_prev_slice)
    cache = (A_prev, hparameters)
    assert(A.shape == (m, n_H, n_W, n_C))
    return A, cache

def relu(x):
   return np.maximum(0,x)

np.random.seed(1)
x = np.random.randn(4, 3, 3, 2)
x_pad = zero_pad(x, 2)
print ("x.shape =", x.shape)
print ("x_pad.shape =", x_pad.shape)
print ("x[1, 1] =", x[1, 1])
print ("x_pad[1, 1] =", x_pad[1, 1])
fig, axarr = plt.subplots(1, 2)
axarr[0].set_title('x')
axarr[0].imshow(x[0,:,:,0])
axarr[1].set_title('x_pad')
axarr[1].imshow(x_pad[0,:,:,0])

np.random.seed(1)
Input = np.random.randn(5, 6, 6, 2) # m, n_H_prev, n_W_prev, n_C_prev
W1 = np.random.randn(2, 2, 2, 2) # f, f, n_C_prev, n_C
b1 = np.random.randn(1, 1, 1, 2) # 1, 1, 1, n_C
hparameters = {"pad" : 2,
               "stride": 2}
print("Input-size:",len(Input),len(Input[0]))
C1, cache_conv1 = conv_forward(Input, W1, b1, hparameters)
print("\nCovolution layer-1:")
print("C1-size:",len(C1),len(C1[0]))
print(C1)
print("C1's mean =", np.mean(C1))
print("cache_conv1[0][1][2][3] =", cache_conv1[0][1][2][3])

np.random.seed(1)
hparameters = {"stride" : 1, "f": 2}
P1, cache = pool_forward(C1, hparameters)
print("\nPooling layer 1: mode = max")
print("P1: size=",len(P1),len(P1[0]))
print("P1 =\n",P1)

R1 = relu(P1)
print("\nReLu for Pooling layer-1: R1 =\n",R1)#reLu implementation for Pooling layer 1


np.random.seed(1)
W2 = np.random.randn(7, 7, 2, 1) # f, f, n_C_prev, n_C
b2 = np.random.randn(1, 1, 2, 1) # 1, 1, 1, n_C
hparameters = {"pad" : 2,
               "stride": 1}
C2, cache_conv2 = conv_forward(R1, W2, b2, hparameters)
print("\nCovolution layer-2:")
print("C2-size:",len(C2),len(C2[0]))
print(C2)
print("C2's mean =", np.mean(C2))
print("cache_conv2[0][1][2][3] =", cache_conv2[0][1][2][3])

np.random.seed(1)
hparameters = {"stride" : 1, "f": 2}
P2, cache = pool_forward(C2, hparameters)
print("\nPooling layer 2: mode = max")
print("P2: size=",len(P2),len(P2[0]))
print("P2 =\n",P2)

R2 = relu(P2)
print("\nReLu for Pooling layer-2: R2 =\n",R2)#reLu implementation for Pooling layer 1

# flattening the pooled layer
# fully connected layer
FC = relu(R2).reshape((len(R2)*len(R2[0])*1,1)) # n_C = 1 from prev layer n_C #relu for fully connected layer
print("\nFully connected layer:\n",FC)
