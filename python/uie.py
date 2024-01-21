import cv2
import numpy as np
import matplotlib.pyplot as plt

#Image processing functions
def grey_world_equalisation(c):
    h, w = c.shape
    channel_average = c.sum() / (255 * h * w)
    theta = 0.5 / channel_average
    normalized_c = theta * c

    return np.round(np.clip(normalized_c, 0, 255)).astype(np.uint8)


def histogram_stretch(c, old_min, old_max, new_min, new_max):
    c = c.astype(np.float32)
    streched_c = (c - old_min) * ((new_max - new_min)/(old_max - old_min)) + new_min
    return np.round(np.clip(streched_c, 0, 255)).astype(np.uint8)


def histogram_parameter_estimation(c, col):
    c_flat = np.sort(c.flatten())

    val, counts = np.unique(c_flat, return_counts=True)
    mode = val[np.argmax(counts)]
    mode_index = np.where(c_flat == mode)[0][0]

    i_min = int(c_flat[int(mode_index * 0.001)])
    i_max = int(c_flat[int(-(len(c_flat) - mode_index) * 0.001)])
    
    sd = 0.655 * mode
    o_min = int(mode - sd)

    if col == "r":
        k = 1.1
        nrer = 0.83
    elif col == "g":
        k = 0.9
        nrer = 0.55
    elif col == "b":
        k = 0.9
        nrer = 0.97

    d = 3
    m = 1 #for simplification mu is set to 1 insteading of solving an inequality

    o_max = int((mode + m)/(k * (nrer**d)))

    if o_max < i_max or o_max > 255: o_max = 255
    if o_min > i_min or o_min < 0: o_min = 0

    o_min = 0
    o_max = 255

    #print(f"Channel: {col} I_min: {i_min} I_max: {i_max} O_min: {o_min} O_max: {o_max}")
    return i_min, i_max, o_min, o_max 


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


def RGHS_enchancement(im_path):
    im = cv2.imread(im_path)
    h, w, _ = im.shape

    im = cv2.resize(im, (w//4, h//4)) #FOR DISPLAY PURPOSES, remove later
    #cv2.imshow("og im", im)

    b, g, r = cv2.split(im)

    #b and g channel equilisation
    b = grey_world_equalisation(b)
    g = grey_world_equalisation(g)

    #Relative global histogram stretching
    b = histogram_stretch(b, *histogram_parameter_estimation(b, "b")) 
    g = histogram_stretch(g, *histogram_parameter_estimation(g, "g"))
    r = histogram_stretch(r, *histogram_parameter_estimation(r, "r"))

    im = cv2.merge((b, g, r))
    cv2.imshow("image", im)
    cv2.waitKey(0)




if __name__ == "__main__":
    RGHS_enchancement("data/images/im0.JPG")
