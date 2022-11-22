
import cv2
import matplotlib.pyplot as plt
import numpy as np

def main():
    path = '/home/mayabee/Documents/DIP_Lab/Assignments/Assignment-8/damaged.jpg'
    imgo = plt.imread(path)
    
    r,c,l = imgo.shape
    structuring_element = np.ones((5,5), dtype = np.uint8)
    gray = cv2.cvtColor(imgo, cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
    
    
    er_img = mor_erode(binary, structuring_element);
    dil_img = mor_dilate(binary, structuring_element)
    op = opening(binary, structuring_element)
    cl = closing(binary, structuring_element)
    
    image_er = cv2.erode(binary, structuring_element)
    image_dil = cv2.dilate(binary,structuring_element)

    openin = cv2.morphologyEx(binary, cv2.MORPH_OPEN,structuring_element, iterations=1) 
    closin = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, structuring_element, iterations=1)
    img_set = [ imgo,binary, er_img,image_er, dil_img,image_dil, op,openin, cl, closin]
    title_set = ['Original','Binary ', 'Erotion with implemented','Erotion with builtin', 'Dilation with implemented','Dilation with built-in' ,'Opening with implemented', 'Opening with built-in','closing with implemented', 'closing with builtin']
    plot_img(img_set,title_set)


def mor_dilate(img, element):
    r,c = img.shape
    m,n = element.shape
    p = (m//2)
    proc = np.zeros((r,c), dtype = np.int8)
    img = np.pad(img, p, constant_values = 0)
    
    for i in range(r):
        for j in range(c):
            res = np.sum(img[i:i+m, j:j+n] * element)
            if res > 0:
                proc[i,j] = 255
    return proc

def mor_erode(img, element):
    r,c = img.shape
    m,n = element.shape
    p = (m//2)
    proc = np.zeros((r,c), dtype = np.int8)
    img = np.pad(img, p, constant_values = 0)
    
    for i in range(r):
        for j in range(c):
            res = np.sum(img[i:i+m, j:j+n] * element)
            if res  == 255*m*n:
                proc[i,j] = 255
    return proc

def opening(img, element):
    r,c = img.shape
    img1 = mor_erode(img, element)
    img2 = mor_dilate(img1, element)
    return img2
def closing(img, element):
    r,c = img.shape
    img1 = mor_dilate(img, element)
    img2 = mor_erode(img1, element)
    return img2
    
def plot_img(img_set, title_set):
    n=len(img_set)
    plt.figure(figsize= (20,20))

    for i in range(n):
        img = img_set[i]
        ch = len(img.shape)
        plt.subplot(5,2,i+1)
        if(ch == 3):
            plt.imshow(img_set[i])
        else:
            plt.imshow(img_set[i], cmap = 'gray')
        plt.title(title_set[i])    
    plt.savefig('binary_masking.jpg')
    plt.show()
  

    
if __name__ == '__main__':
    main()