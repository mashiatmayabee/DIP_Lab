from turtle import right
from cv2 import COLOR_RGB2GRAY, imshow
import matplotlib.pyplot as plt
import numpy as np
import cv2
def main():
    img_path = 'G:/4y1s/DIP_Lab/Assingment-7/hydra.jpg'
    rgb = plt.imread(img_path)
    gray = cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
    
    r, c = gray.shape
    right = np.zeros((r,c), dtype=np.uint8)
    left = np.zeros((r,c), dtype=np.uint8)
    mid = np.zeros((r,c), dtype=np.uint8)
    
    for i in range(r):
        for j in range(c):
            temp = gray[i,j]
            temp = temp + 120
            if temp > 255:
                temp = 255
            right[i,j] = temp
    for i in range(r):
        for j in range(c):
            temp = gray[i,j]
            temp = temp - 120
            if temp < 0:
                temp = 0
            left[i,j] = temp
    for i in range(r):
        for j in range(c):
            temp = gray[i,j]
            if temp < 50:
                temp = 50
            elif temp >200:
                temp = 200
            mid[i,j] = temp
    img_set = [ gray, right , left, mid]
    title = ['Original Image', 'Moved Right', 'Moved left', 'Narrow Band']
    plot_img(img_set, title)
def plot_img(img_set, title_set):
    n = len(img_set)
    plt.figure(figsize= (20,20))
    j = 1
    for i in range(n):
        img = img_set[i]
        ch = len(img.shape)
        plt.subplot(2,n,j)
        j+=1 
        if(ch == 3):
            plt.imshow(img)
        else:
            plt.imshow(img, cmap = 'gray')
        plt.subplot(2,n,j)
        j+=1
        plt.hist(img.ravel(),256,[1,254])

        
        plt.title(title_set[i])    
    plt.savefig('histogram_equalization.jpg')
    plt.show()
       

if __name__ == '__main__':
    main()