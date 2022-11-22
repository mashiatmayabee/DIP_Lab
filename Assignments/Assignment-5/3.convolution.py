import matplotlib.pyplot as plt
import numpy as np
import cv2

def main():
    img_path = '/home/mayabee/Documents/DIP_Lab/Assignments/Assignment-5/pea.jpg'
    img_original = plt.imread(img_path)
    gray =cv2.cvtColor(img_original,cv2.COLOR_RGB2GRAY)
    
    kernel_laplacian =np.array([[0,1,0], [1,-4,1], [0,1,0]], dtype=np.int8)
    kernel_sobelX =np.array([[-1, 0, 1],[-2, 0, 2],[-1, 0, 1]], dtype=np.int8)
    kernel_sobelY =np.array([[-1, -2, -1],[0, 0, 0],[1, 2, 1]], dtype=np.int8)
    

    r,c = gray.shape
    kr,kc = kernel_laplacian.shape
    m = r + kr - 1
    n = c + kc - 1
    padded_img = np.zeros((m,n),dtype = np.uint8)
    for i in range(r):
        for j in range(c):
            padded_img[i+kr//2,j+kc//2] = gray[i,j]
    print('Padded',padded_img)
    # pr, pc = padded_img.shape
    
    processed_img1 = np.zeros((r,c),dtype = np.uint8)   
    processed_img2 = np.zeros((r,c),dtype = np.uint8)
    processed_img3 = np.zeros((r,c),dtype = np.uint8)
    
    for i in range(kc//2 , m-(kc//2)-1):
        for j in range(kc//2, n- (kc//2)-1):
            processed_img1[i,j] = kernel_laplacian[1,1]*padded_img[i,j] + kernel_laplacian[0,0]*padded_img[i-1,j-1] +kernel_laplacian[0,1]*padded_img[i-1,j] 
            + kernel_laplacian[1,0]*padded_img[i,j-1] + kernel_laplacian[2,2]*padded_img[i+1,j+1] + kernel_laplacian[1,2]*padded_img[i,j+1] 
            + kernel_laplacian[2,1]*padded_img[i+1,j] + kernel_laplacian[0,2]*padded_img[i-1,j+1] + kernel_laplacian[2,0]*padded_img[i+1,j-1]     
            if(processed_img1[i,j] >255):
                processed_img1[i,j] = 255
            
    for i in range(kc//2 , m-(kc//2)-1):
        for j in range(kc//2, n- (kc//2)-1):
            processed_img2[i,j] = kernel_sobelX[1,1]*padded_img[i,j] + kernel_sobelX[0,0]*padded_img[i-1,j-1] +kernel_sobelX[0,1]*padded_img[i-1,j] 
            + kernel_sobelX[1,0]*padded_img[i,j-1] + kernel_sobelX[2,2]*padded_img[i+1,j+1] + kernel_sobelX[1,2]*padded_img[i,j+1] 
            + kernel_sobelX[2,1]*padded_img[i+1,j] + kernel_sobelX[0,2]*padded_img[i-1,j+1] + kernel_sobelX[2,0]*padded_img[i+1,j-1]     
            if(processed_img2[i,j] >255):
                processed_img2[i,j] = 255

    for i in range(kc//2 , m-(kc//2)-1):
        for j in range(kc//2, n- (kc//2)-1):
            processed_img3[i,j] = kernel_sobelY[1,1]*padded_img[i,j] + kernel_sobelY[0,0]*padded_img[i-1,j-1] +kernel_sobelY[0,1]*padded_img[i-1,j] 
            + kernel_sobelY[1,0]*padded_img[i,j-1] + kernel_sobelY[2,2]*padded_img[i+1,j+1] + kernel_sobelY[1,2]*padded_img[i,j+1] 
            + kernel_sobelY[2,1]*padded_img[i+1,j] + kernel_sobelY[0,2]*padded_img[i-1,j+1] + kernel_sobelY[2,0]*padded_img[i+1,j-1]     
            if(processed_img3[i,j] >255):
                processed_img3[i,j] = 255
            
    processed_img = cv2.filter2D(gray,-1,kernel_laplacian)
    print(processed_img2)
    print(processed_img)
    img_set = [ gray, processed_img1,processed_img2, processed_img3]
    title_set = [ 'Grayscale Image','Laplacian', 'SobelX', 'SobelY']
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
