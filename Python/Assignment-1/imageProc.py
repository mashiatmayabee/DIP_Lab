import matplotlib.pyplot as plt
import cv2

def main():

    img_path = 'G:/4y1s/DIP_Lab/Assignment-4/pea.jpg'
    print(img_path)
    rgb = plt.imread(img_path)
    print("Rgb",rgb.shape, rgb.max(), rgb.min())

    red = rgb[:, :, 0]
    green = rgb[:, :, 1]
    blue = rgb[:, :, 2]
    print(red.shape, red.max(), red.min())

    grayscale = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    print(grayscale.shape, grayscale.max(), grayscale.min())
    _, binary = cv2.threshold(grayscale, 50, 255, cv2.THRESH_BINARY)

    img_set = [rgb, red, green, blue, grayscale, binary]
    title_set = ['RGB', 'Red', 'Green', 'Blue', 'Grayscale', 'Binary']
    plt.figure(figsize = (10, 10))
    for i in range(6):
        plt.subplot(2, 3,  i + 1)
        plt.title(title_set[i])
        ch = len(img_set[i].shape)
        if (ch == 3):
            plt.imshow(img_set[i])
        else:
            plt.imshow(img_set[i], cmap = 'gray')			
    plt.show()
    print(rgb.max())
    plt.figure(figsize = (10, 10))
    for i in range(6):
        plt.subplot(2, 3,i + 1)
        plt.title(title_set[i])
        plt.hist(img_set[i].ravel(),256 ,[0,256]);
        # plt.hist(img.ravel(), 256, [0,356])
    plt.show()
    
if __name__ == '__main__':
    main()