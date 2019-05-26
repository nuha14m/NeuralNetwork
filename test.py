import pickle
import pandas as pd
import numpy as np
import math


def sigmoid(x):
    return 1 / (1 + pow(math.e, -x))

f=  open("mypickle", "rb")
test_set=list()
with open("mnist_test.csv") as fw:
    for line in fw:
        list = line.split(",")
        yval= int(list[0])
        list=list[1:]
        list=[float(x)/256 for x in list]
        sub = np.full((1, 10), 0)
        sub[0,yval] =1
        test_set.append((list,sub))


W, b, layers = pickle.load(f)

#print(W,b)
err=0
a=dict()
A=np.vectorize(sigmoid)
for (x,y) in test_set:
    a[0]=x
    for L in range(1, layers):
        dot_L = a[L-1] * W[L-1] + b[L]
        a[L] = A(dot_L)
    if np.argmax(a[layers-1]) != np.argmax(y): err+=1
print(err)

