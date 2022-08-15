import matplotlib.pyplot as plt
import numpy as np
import cv2

def main():
    image = plt.imread("G:/4y1s/DIP_Lab/Assingment-7/hydra.jpg")
    processed_image = np.fft.fft2(image)
    inver = np.fft.ifft2(processed_image)
    plt.imshow(processed_image)
