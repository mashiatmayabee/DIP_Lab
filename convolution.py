import matplotlib.pyplot as plt
import cv2
import numpy as np

def main():
    img = plt.imread('G:/4y1s/DIP_Lab/Assignment-4/hydra.jpg')
    gray =  cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    r, c = gray.shape
    kernel = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
    kr,kc = kernel.shape

    m = r + kr//2
    n = c + kc//2
    padded_image = np.zeros((m,n), dtype= np.int8)
    for i in range(m):
        for j in range(n):
            padded_image[i+kr//2, j+kc//2]= gray[i, j]

    for i in range(r):
        for j in range(c):
            for k in range(kr):
                for l in range(kc):
                     
                     
if __name__=='__main__':
    main()