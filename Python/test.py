from random import randrange
import matplotlib.pyplot as plt
import cv2
def main():
    img  = plt.imread('G:/4y1s/DIP_Lab/Assignment-5/pea.jpg')
    red = img[ :, :, 0]
    green = img[ :, :, 1]
    blue = img[ :, :, 2]
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
    print("Shape of red\n",red.shape, red.max(), red.min())
    img_set = [img , gray, red, green, blue, binary]
    title_set = ['Original', 'Gray', 'Red', 'Green', 'Blue', 'Binary']
    plot_image(img_set,title_set)
    
def plot_image(img_set, title_set):
    n = len(img_set)
    plt.figure(figsize=(18, 18))
    for i in range(n):
        plt.subplot(2,3,i+1)
        plt.title(title_set[i])
        if len(img_set[i].shape)==3:
            plt.imshow(img_set[i])
        else:
            plt.imshow(img_set[i], cmap='gray')
    plt.figure(figsize=(19,19))
    for i in range(n):
        plt.subplot(3,2,i+1)
        plt.title(title_set[i])
        plt.hist(img_set[i].ravel(), 256,[0, 256])        

    plt.show()
if __name__ == '__main__':
    main()