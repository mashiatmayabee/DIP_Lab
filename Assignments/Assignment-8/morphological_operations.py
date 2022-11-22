import matplotlib.pyplot as plt
import numpy as np
import cv2

def main():
    img_path = 'G:/4y1s/DIP_Lab/Assignment-8/damaged.jpg'
    img_original = plt.imread(img_path)
    gray =cv2.cvtColor(img_original,cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
    kernel = np.ones((3,3), np.uint8)
    image_er = cv2.erode(binary, kernel)
    image_dil = cv2.dilate(binary, kernel)

    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN,kernel, iterations=1) 
    closing = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)

    img_set = [img_original, binary, image_er, image_dil, opening, closing]
    title_set = [ 'Original','binary', 'eroded', 'dilated', 'opening ', 'closing']
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
    plt.savefig('morph.jpg')
    plt.show()

if __name__ == '__main__':
    main()
