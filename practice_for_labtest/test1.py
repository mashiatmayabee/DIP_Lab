import cv2
import matplotlib.pyplot as plt
import numpy as np
def main():
    img = cv2.imread('/home/mayabee/DIP_Lab/practice_for_labtest/hydra.jpg')
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray, 50, 255,cv2.THREA)
    # _,binary =cv2.threshold(gray, 149, 255, cv2.THRESH_BINARY)

    red = img[:, :, 0]
    green = img[:, :, 2]
    blue = img[:, :, 3]

    img_set = [img, gray, binary, red, green, blue]
    title_set = ['img', 'gray','binary' 'reds', 'greens', 'blues']
    plot_img(img_set, title_set)


def plot_img(img_set, title_set):
    n = len(img_set)
    plt.figure(figsize= (18,18))
    for i in range(n):
        plt.subplot(n/2,2,i+1)
        if(len(img_set[i].shape) == 3):
            plt.imshow(img_set[i])
        else:
            plt.imshow(img_set[i], cmap = 'gray')
        plt.title(title_set[i])
    plt.show()

def histogram_imp(img):
    r, c = img.shape

    for i in range(r):
        for j in range(c):
            print(233)

    

if __name__ == '__main__':
    main()