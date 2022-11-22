import matplotlib.pyplot as plt
import numpy as np
import cv2

def main():
    img_path = '/home/mayabee/Documents/DIP_Lab/Assignments/Assignment-4/hydra.jpg'
    img_original = plt.imread(img_path)
    # plt.subplot(1,2,1)
    # plt.imshow(img_original)

    gray =cv2.cvtColor(img_original,cv2.COLOR_RGB2GRAY)
    # gray = np.zeros((5,5), dtype=np.uint8)
    # gray+= np.arange(5, dtype= np.uint8)
    kernel =np.array()
    """ Zero Padding"""
    r,c = gray.shape
    kr,kc = kernel.shape
    m = r + kr // 2
    n = c + kc // 2
    padded_img = np.zeros((m,n),dtype = np.uint8)
    for i in range(r):
        for j in range(c):
            padded_img[i+kr//2,j+kc//2] = gray[i,j]
    print('Padded',padded_img)
    # pr, pc = padded_img.shape
    processed_img2 = np.zeros((r,c),dtype = np.uint8)
    
    processed_img = np.zeros((r,c),dtype = np.uint8)   
    
    for i in range(kc//2 , m-(kc//2)-1):
        for j in range(kc//2, n- (kc//2)-1):
            processed_img2[i,j] = kernel[1,1]*padded_img[i,j] + kernel[0,0]*padded_img[i-1,j-1] +kernel[0,1]*padded_img[i-1,j] 
            + kernel[1,0]*padded_img[i,j-1] + kernel[2,2]*padded_img[i+1,j+1] + kernel[1,2]*padded_img[i,j+1] 
            + kernel[2,1]*padded_img[i+1,j] + kernel[0,2]*padded_img[i-1,j+1] + kernel[2,0]*padded_img[i+1,j-1]     
            if(processed_img2[i,j] >255):
                processed_img2[i,j] = 255
            
    processed_img = cv2.filter2D(gray,-1,kernel)
    print(processed_img2)
    print(processed_img)
    img_set = [img_original, gray,processed_img, processed_img2]
    title_set = [ 'Original','Gray', 'Builtin', 'Implemented']
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
    plt.savefig('Convolution.jpg')
    plt.show()

if __name__ == '__main__':
    main()
