import cv2
import matplotlib.pyplot as plt
import numpy as np

def main():
    img_path = '/home/mayabee/Documents/DIP_Lab/Assignments/Assignment-2/hydra.jpg'
    rgb_img = plt.imread(img_path)
    print(rgb_img.shape, rgb_img.max(), rgb_img.min())

    grayscale = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY)
    print(grayscale.shape)

    kernel1 = np.array([[-1,-1,-1], [-1,8,-1], [-1,-1,-1]])
    print('kernel1: {}'.format(kernel1))
    processed_img1 = cv2.filter2D(grayscale,-1,kernel1)
    kernel2 = np.array([[-1,0,-1], [-1,8,-1], [-1,0,-1]])
    print('kernel2: {}'.format(kernel2))
    processed_img2 = cv2.filter2D(grayscale,-1,kernel2)
    
    kernel3 = np.array([[0,0,0], [0,1,0], [0,0,0]]) 
    print('kernel3: {}'.format(kernel3))
    processed_img3 = cv2.filter2D(grayscale,-1,kernel3)
    
    kernel4 = np.array([[0,-1,0], [-1,5,-1], [0,-1,0]])
    print('kernel4: {}'.format(kernel4))
    processed_img4 = cv2.filter2D(grayscale,-1,kernel4)
    
    kernel5 = np.array([[1,1,1], [1,1,1], [1,1,1]])*1/9
    print('kernel5: {}'.format(kernel5))
    processed_img5 = cv2.filter2D(grayscale,-1,kernel5)
    
    kernel6 = np.array([[1,1,1], [1,1,1], [1,1,1]])
    print('kernel6: {}'.format(kernel6))
    processed_img6 = cv2.filter2D(grayscale,-1,kernel6)
    
    img_set = [rgb_img,grayscale,processed_img1,processed_img2,processed_img3, processed_img4,processed_img5, processed_img6]
    title_set = ['RGB image', 'Grayscale', 'Kernel1','Kernel2', 'Kernel3', 'Kernel4', 'Kernel5', 'Kernel6']

    plot_img(img_set,title_set)

def plot_img(img_set, title_set):
    n=len(img_set)
    plt.figure(figsize= (18,18))

    for i in range(n):
        img = img_set[i]
        ch = len(img.shape)
        plt.subplot(2,4,i+1)
        if(ch == 4):
            plt.imshow(img_set[i])
        else:
            plt.imshow(img_set[i], cmap = 'gray')
        plt.title(title_set[i])
    plt.savefig("a3.jpg")
    plt.show()

if __name__=='__main__':
    main()
    
