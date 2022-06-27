import cv2
from cv2 import threshold
import matplotlib.pyplot as plt

def main():
    img = plt.imread('G:/4y1s/DIP_Lab/Assignment-4/pea.jpg')
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
    red = img[:, :, 0]
    green = img[:, :, 1]
    blue = img[:, :, 2]
    plt.subplot(3,2,1)
    plt.imshow(img)
    plt.subplot(3,2,2)
    plt.imshow(red)
    plt.subplot(3,2,3)
    plt.imshow(green)
    plt.subplot(3,2,4)
    plt.imshow(blue)
    plt.subplot(3,2,5)
    plt.imshow(gray, cmap='gray')
    plt.subplot(3,2,6)
    plt.imshow(binary, cmap='gray')
    print( len(binary.shape))
    print( len(img.shape))
    plt.show()

if __name__ == '__main__':
    main()