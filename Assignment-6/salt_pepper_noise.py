
import numpy as np
import cv2
import random
import matplotlib.pyplot as plt

def main():
    img_path = 'G:/4y1s/DIP_Lab/Assignment-6/xray.jpg'
    img = plt.imread(img_path)    
    gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    noisy_img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    r, c = gray.shape
    flag = 0
    for i in range (c*r//15):
        x = random.randint(0, r-1)
        y = random.randint(0, c-1)
        if flag == 0:
            noisy_img[x,y] = 0
            flag = 1
        else:
            noisy_img[x,y] = 255
            flag = 0

    avg_kernel = np.array( [[1, 1, 1], [1, 1, 1], [1, 1, 1]])*(1/9)
    gaussian_kernel = np.array( [[1, 2, 1], [2, 4, 2], [1, 2, 1]])*(1/16)
    gray_filtered = cv2.filter2D(gray,-1,avg_kernel)
    avg_filtered = cv2.filter2D(noisy_img,-1,avg_kernel)
    gaussian_filtered = cv2.filter2D(noisy_img,-1,gaussian_kernel)
    figure_size = 3
    median_filtered = cv2.medianBlur(noisy_img, figure_size)

    img_set = [gray,gray_filtered, noisy_img, avg_filtered, gaussian_filtered, median_filtered]
    title_set = ['Gray','Filtered Image(Avaraging)', 'Noisy Image','Avarage filtered image', 'Gaussian Filtered Image', 'Median Filtered Image']
    plot_img(img_set,title_set)

def plot_img(img_set, title_set):
    n=len(img_set)
    plt.figure(figsize= (20,20))

    for i in range(n):
        img = img_set[i]
        ch = len(img.shape)
        plt.subplot(2,3,i+1)
        if(ch == 3):
            plt.imshow(img_set[i])
        else:
            plt.imshow(img_set[i], cmap = 'gray')
        plt.title(title_set[i])    
    plt.savefig('salt_pepper_noise.jpg')
    plt.show()
  
if __name__== '__main__':
    main()