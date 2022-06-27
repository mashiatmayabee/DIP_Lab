from concurrent.futures import process
from cv2 import bitwise_and
import numpy as np
import matplotlib.pyplot as plt
import cv2

def main():
    img_path = 'G:/4y1s/DIP_Lab/Assignment-5/pea.jpg'
    img = plt.imread(img_path)
    plt.subplot(2,5,1)
    plt.imshow(img)
    plt.title("Original Image")    

    gray =cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    plt.subplot(2,5,2)
    plt.imshow(gray, cmap = 'gray')
    plt.title('Grayscale')    

    r,c = gray.shape
    processed_img = np.zeros( (r,c), dtype = np.int8)
    n = 1
    for k in range(8):
        p = n<<k
        print(p)
        processed_img= cv2.bitwise_and(gray, p, mask=None)
        
        ch = len(img.shape)
        plt.subplot(2,5,k+3)
        if(ch == 3):
            plt.imshow(processed_img)
        else:
            plt.imshow(processed_img, cmap = 'gray')

        plt.title(k+1)    
    plt.savefig('bit_slicing.jpg')
    plt.show()
        
if __name__ == '__main__':
    main()