from cv2 import bitwise_and
import numpy as np
import cv2
import matplotlib.pyplot as plt

def main():
    img_path1 = 'H:/u.png'
    img1 = plt.imread(img_path1, 0)    
    gray1 =cv2.cvtColor(img1,cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray1, 50, 255, cv2.THRESH_BINARY)

    img_path2 = 'H:/k.png'
    img2 = plt.imread(img_path2, 0)
    gray2 =cv2.cvtColor(img2,cv2.COLOR_RGB2GRAY)
    r,c , ch = img2.shape
    print(gray1.shape)
    print(gray2.shape)
    
    processed_img = cv2.bitwise_and(img2,binary, mask=None)
    processed_img2 = cv2.bitwise_or(img2,img1, mask=None)
    img_set = [ img1, img2, processed_img, processed_img2]
    title_set = [ 'im1','im2', 'and', 'or']
    plot_img(img_set,title_set)

def plot_img(img_set, title_set):
    n=len(img_set)
    plt.figure(figsize= (20,20))

    for i in range(n):
        img = img_set[i]
        ch = len(img.shape)
        plt.subplot(2,2,i+1)
        if(ch == 3):
            plt.imshow(img_set[i])
        else:
            plt.imshow(img_set[i], cmap = 'gray')
        plt.title(title_set[i])    
    plt.show()
  
    
    
if __name__== '__main__':
    main()