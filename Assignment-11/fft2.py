
import numpy as np
import cv2
import matplotlib.pyplot as plt

def main():
    img_path = 'G:/4y1s/DIP_Lab/Assignment-11/len.png'
    img = plt.imread(img_path)    
    gray =cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    domainFilter = cv2.edgePreservingFilter(gray, flags=1, sigma_s=60, sigma_r=0.6)
    gaussBlur = cv2.GaussianBlur(gray,(5,5),cv2.BORDER_DEFAULT)
    fft = np.fft.fftshift(np.fft.fft2(gray))
    magnitude_spectrum = 20*np.log(np.abs(fft))
    
    img_set = [ img, gray, domainFilter, gaussBlur, magnitude_spectrum]
    title_set = [ 'Image1','gray','domainfiltered', 'GaussBlurr', 'fft','ug']
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
    plt.savefig('binary_masking.jpg')
    plt.show()
    plt.savefig('G:/4y1s/DIP_Lab/Assignment-11/fft.png')
    
if __name__== '__main__':
    main()