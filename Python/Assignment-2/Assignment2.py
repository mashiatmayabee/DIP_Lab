import matplotlib.pyplot as plt
import cv2
import numpy as np

def main():

    img_path = 'C:/Users/Mayabee/Documents/Python/Assignment-2/pea.jpg'
    img_rgb = plt.imread(img_path)
    print(img_rgb.shape)
    r = cv2.cvtColor(img_rgb,cv2.COLOR_RGB2GRAY)
    print(r.shape,r.max(),r.min())
   
    #constants
    T1 = 30
    T2 = 50
    c = 3
    p = 1 #I have taken this value as 1 otherwise all the value in s4 becomes 255 and the image becomes all white for the condition I have given
    epsilon = 0.00000000000001

    #Initialization of 4 matrix with value zero
    m,n = r.shape
    s1 = np.zeros((m,n), dtype = np.uint8)  #To use in first transformation
    s2 = np.zeros((m,n),dtype = np.uint8)   #To use in second transformation
    s3 = np.zeros((m,n),dtype = np.uint8)   #To use in third transformation
    s4 = np.zeros((m,n),dtype = np.uint8)   #To use in fourth transformation
    

    for i in range(m):
        for j in range(n):
            if(r[i,j]>= T1 and r[i][j]<=T2):
                s1[i][j] = 100
                s2[i][j] = 10
            
            else:
                s1[i][j] = 10
                s2[i][j] = r[i][j]

            s3[i][j]= c*(np.log(1+r[i][j]))
            s4[i][j] = c*(pow((r[i][j] + epsilon),p))

            if((c*(np.log(1+r[i][j])))>255):
                s3[i][j] = 255
            if((c*(pow((r[i][j] + epsilon),p)))>255):
                s4[i][j] = 255
            
    img_set = [img_rgb, r, s1, s2, s3, s4]
    title_set = ['RGB Image', 'Grayscale', 'Transformation-1', 'Transformation-2', 'Transformation-3', 'Transformation-4']
    plot_img(img_set, title_set)

def plot_img(img_set, title_set):
    n = len(img_set)
    plt.figure(figsize = (20, 20))
    for i in range(n):
        img = img_set[i]
        ch = len(img.shape)

        plt.subplot( 2, 3, i + 1)
        if (ch == 3):
            plt.imshow(img_set[i])
        else:
            plt.imshow(img_set[i], cmap = 'gray')
        plt.title(title_set[i])
    plt.show()		

if __name__=='__main__':
    main()