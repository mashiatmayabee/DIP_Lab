from pickletools import uint8
import cv2
import matplotlib.pyplot as plt
import numpy as np
def main():
    img = cv2.imread('G:/4y1s/DIP_Lab/practice_for_labtest/hydra.jpg')
    red = img[:, :, 0]
    green = img[:, :, 1]
    blue = img[:, :, 2]
    # gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray = grays(img)
    binary = binary_img(gray)
    # _, binary = cv2.threshold(gray, 150, 255,cv2.THRESH_BINARY)
    # _,binary =cv2.threshold(gray, 149, 255, cv2.THRESH_BINARY)


    img_set = [ gray, binary, red, green, blue]
    title_set = [ 'gray','binary', 'Reds', 'Greens', 'Blues']
    plot_img(img_set, title_set)
    

def plot_img(img_set, title_set):
    n = len(img_set)
    plt.figure(figsize= (18,18))
    n1 = range(0,256)
    j = 1
    for i in range(n):
        img = img_set[i]
        ch = len(img.shape)
        plt.subplot(n,2,j)
        j+=1
        if(ch == 3):
            plt.imshow(img_set[i])
        else:
            plt.imshow(img_set[i], cmap = title_set[i])
        plt.title(title_set[i])   
        h =histogram_imp(img_set[i])
        plt.subplot(n,2,(j))
        j+=1
        
        plt.stem(n1,h)
    plt.show()
  
def histogram_imp(img):
    r, c = img.shape
    n = range(0, 256)
    h = np.zeros(256)
    for i in range(r):
        for j in range(c):
            h[img[i,j]] = h[img[i,j]] +1
    # plt.plot(n,h)
    # plt.show()
    return h

def grays(img):
    r,c,l = img.shape
    grayy = np.zeros((r,c), dtype = np.uint8)
    for i in range (r):
        for j in range(c):
            grayy[i,j] = img[i,j,0]*0.299+img[i,j,1]*0.587+img[i,j,2]*0.114
    return grayy

def binary_img(image):
    r,c =image.shape
    bin = np.zeros((r,c), dtype=np.uint8)
    for i in range (r):
        for j in range(c):
            if image[i,j] <130:
                bin[i,j] = 0
            else:
                bin[i,j] = 255
    return bin
    
main()