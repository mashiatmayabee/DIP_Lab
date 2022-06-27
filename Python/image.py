import matplotlib.pyplot as plt
import cv2
def main():
    img_path = 'H:\download.png' 
    
   
    print(img_path)
    img = plt.imread(img_path)
    print(img)
    # grayscale_img= cv2.cvtColor(img,cv2.)
    print(img.shape)
    print(img)
    plt.figure(figsize = (20,20))
    plt.subplot(2,2,1)
    plt.imshow(img[:, :,0])
    grayscale_img= cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    


    # plt.figure(figsize = (20,20))
    plt.subplot(2,2,2)
    plt.imshow(img[:, :,1],cmap='gray')
    # plt.imshow(img)
    # plt.figure(figsize = (20,20))
    plt.subplot(2,2,3)
    plt.imshow(img[:, :,2],cmap='gray')
    # plt.imshow(img)
    # plt.figure(figsize = (20,20))
    plt.subplot(2,2,4)
    plt.imshow(img[:, :,3],cmap='gray')
    # plt.imshow(img)
    plt.imshow(grayscale_img,cmap='gray')

    plt.show()
if __name__ == '__main__':
    main()