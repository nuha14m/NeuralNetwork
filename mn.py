import math
import numpy as np
import time
import random
import pandas as pd
import csv
import pickle

def dfsigmoid(x):
    return (pow(math.e, -x))/(pow(1+pow(math.e, -x), 2))

def sigmoid(x):
    return 1 / (1 + pow(math.e, -x))

def back_prop_matrix(training_set, layers, lmda):
    A = np.vectorize(sigmoid)
    dA = np.vectorize(dfsigmoid)
    a,b,W, delta= dict(), dict(), dict(), dict()
    N=len(layers)-1
    error=500000000
    while error>10:
        counter=0
        for i in range(N):
            W[i]=  np.matrix(2*np.random.rand(layers[i], layers[i+1])-1)
            b[i+1]= np.matrix(2*np.random.rand(1, layers[i+1])-1)
        while error>10:
            err=0
            counter=0
            for (x,y) in training_set:
                counter+=1
                a[0]=x
                for L in range(1, N+1):
                    dot_L = a[L-1] * W[L-1] + b[L]
                    a[L] = A(dot_L)
                delta[N] = np.multiply(dA(a[N-1]*W[N-1]+b[N]), y-a[N])
                for L in range(N-1, 0, -1):
                    delta[L] = np.multiply(dA(a[L-1]*W[L-1]+b[L]),delta[L+1]*(W[L].transpose()))
                for L in range(N): # update weights
                    W[L] = W[L] + ((lmda*a[L].transpose())*delta[L+1])
                    b[L+1]= b[L+1] + (lmda*delta[L+1])
                print(counter,np.linalg.norm(y-a[N])**2)
                if np.argmax(a[N]) != np.argmax(y): err+=1
            error=err
            lmda= error/60000
            with open("mypickle", "wb") as f:
                print("Pickled")
                pickle.dump([W,b, len(layers)], f)


def convert(imgf, labelf, outf, n):
    f = open(imgf, "rb")
    o = open(outf, "w")
    l = open(labelf, "rb")
    
    f.read(16)
    l.read(8)
    images = []
    
    for i in range(n):
        image = [ord(l.read(1))]
        for j in range(28*28):
            image.append(ord(f.read(1)))
        images.append(image)
    
    for image in images:
        o.write(",".join(str(pix) for pix in image)+"\n")
    f.close()
    o.close()
    l.close()

#convert("train-images-idx3-ubyte", "train-labels-idx1-ubyte", "mnist_train.csv", 60000)
#convert("t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte", "mnist_test.csv", 10000)

training_set=list()

with open("mnist_train.csv") as fw:
    for line in fw:
        list = line.split(",")
        yval= int(list[0])
        list=list[1:]
        list=[float(x)/256 for x in list]
        sub = np.full((1, 10), 0)
        sub[0,yval] =1
        training_set.append((np.mat(list),np.mat(sub)))

back_prop_matrix(training_set, [784, 400, 100, 10], 0.1)





