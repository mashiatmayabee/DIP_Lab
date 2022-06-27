# import cv2, numpy, matplotlib
import cv2
import numpy as np
import matplotlib.pyplot as plt

def main():
    img = plt.imread('G:/4y1s/DIP_Lab/Assignment-4/hydra.jpg')
    gray =cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    plt.subplot(1,2,1)
    plt.imshow(img)
    plt.show()

    r,c = gray.shape
    h = np.zeros(256)
    n = range(0, 256)
    for i in range(r):
        for j in range(c):
            h[gray[i,j]] = h[gray[i,j]]+1
    plt.subplot(2,1,1)
    plt.stem(n,h)
    plt.title('Without Built-in Function')

    plt.subplot(2,1,2)
    plt.hist(gray.ravel(),256,[0,256])
    plt.title('With Built-in(plt.hist()) function')

    plt.show()

if __name__=='__main__':
    main()
