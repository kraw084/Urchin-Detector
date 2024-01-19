import cv2
import numpy as np
import matplotlib.pyplot as plt

#Image processing functions
def grey_world_equalisation(c):
    h, w = c.shape
    channel_average = c.sum() / (255 * h * w)
    theta = 0.5 / channel_average
    normalized_c = theta * c
    return np.clip(np.round(normalized_c).astype(np.uint8), 0, 255)


def histogram_stretch(c, old_min, old_max, new_min, new_max):
    streched_c = (c - old_min) * ((new_max - new_min)/(old_max - old_min)) + new_min
    return np.clip(np.round(streched_c).astype(np.uint8), 0, 255)

#Debugging functions
def plot_hist(b, g, r, title = "RGB Histogram"):
    for c, col in zip((b, g, r), ("blue", "green", "red")): 
        histogram = cv2.calcHist([c], [0], None, [256], [0, 256])
        plt.plot(histogram, color=col)

    plt.title(title)
    plt.xlabel("Pixel Value")
    plt.ylabel("Frequency")

    plt.show()

def show_channels(b, g, r):
    for c, col in zip((b, g, r), ("blue", "green", "red")):
        cv2.imshow(col, c)






def enchance_underwater_image(im_path):
    im = cv2.imread(im_path)
    h, w, _ = im.shape

    im = cv2.resize(im, (w//4, h//4)) #FOR DISPLAY PURPOSES, remove later
    #cv2.imshow("og im", im)

    b, g, r = cv2.split(im)

    #b and g channel equilisation
    b = grey_world_equalisation(b)
    g = grey_world_equalisation(g)


    im = cv2.merge((b, g, r))
    #cv2.imshow("image", im)
    cv2.waitKey(0)




if __name__ == "__main__":
    enchance_underwater_image("data/images/im0.JPG")