import numpy as np
import matplotlib.pyplot as plt
import cv2

def main():
    img_path = 'G:/4y1s/DIP_Lab/Assignment-4/hydra.jpg'
    img = plt.imread(img_path)
    img =cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    # psnr=10*np.log10((255*255)/(1/(225*225)*np.sum(img)*np.sum(img)))
    # print('PSNR is: ',psnr)
    fig = plt.figure()
    fig.set_figheight(15)
    fig.set_figwidth(15)

    for i in range(1,9):
        fig.add_subplot(4,2,i)
        plt.imshow(discriminate_bit(i-1,img), cmap='gray')


    plt.show(block=True)
def cov_binary(num):
    return "{0:b}".format(int(num))
def conv_decimal(listt):
    x = 0
    for i in range(8):
        x = x + int(listt[i])*(2**(7-i))
    return x
def discriminate_bit(bit,img):
    z = np.zeros([225,225])
    for i in range(225):
        for j in range(225):
            x = cov_binary(img[i][j])
            for k in range(8):
                if k == bit:
                    x[k] = x[k]
                else:
                    x[k] = 0
            x1 = conv_decimal(x)
            z[i][j] = x1
    return z

    
if __name__ == '__main__':
    main()