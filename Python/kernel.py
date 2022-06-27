import matplotlib.pyplot as plt
import numpy as np
import cv2
def main():
    kernel= np.ones((3,3),dtype = np.int8)*10
    print(kernel, kernel.dtype)
    kernel2= np.zeros((3,3))
    print(kernel2, kernel2.dtype)

    kernel3 = np.array([[0,-1,0], [-1,4,-1],[0,-1,0]])
    print(kernel3,kernel3.dtype)
if __name__ == '__main__':
    main()