import matplotlib.pyplot as plt
import cv2

import numpy as np

def main():
    img_path = 'C:/Users/Mayabee/Documents/Python/Assignment2/hydra.jpg'
    img_rgb = plt.imread(img_path)
    print(img_rgb.shape)
    r = cv2.cvtColor(img_rgb,cv2.COLOR_RGB2GRAY)
    print(r.shape)

    T1 = 30
    T2 = 110
    c = 2
    p = 4
    epsilon = 0.00000000000001
    
    m,n = r.shape
    s1 = np.ones(m,n)
    s2 = np.ones(m,n)
    s3 = np.ones(m,n)
    s4 = np.ones(m,n)

    for i in range(m):
        for j in range(n):
            if(r[i,j]>= T1 & r[i][j]<=T2):
                s1[i][j] = 100
                s2[i][j] = 10
            elif(r[i,j]< T1 & r[i][j]>T2):
                s1[i][j] = 10
                s2[i][j] = r[i][j]
            s3[i][j] = c*(np.log(i+r[i][j]))
            s4[i][j] = c * pow((r + epsilon),p)
    img_set = [img_rgb, r, s1, s2, s3, s4]
    title_set = ['RGB Image', 'Grayscale', 'Transformation-1', 'Transformation-2', 'Transformation-3', 'Transformation-4']
    plot_img(img_set,title_set)

def plot_img(img_set, title_set):
    n = len(img_set)
    plt.figure(figsize = (20, 20))
    for i in range(n):
        img = img_set[i]
        ch = len(img.shape)

        plt.subplot( 2, 2, i + 1)
        if (ch == 3):
            plt.imshow(img_set[i])
        else:
            plt.imshow(img_set[i], cmap = 'gray')
        plt.title(title_set[i])
    plt.show()		


if __name__ == '__main__':
    main()