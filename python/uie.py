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
    return np.round(np.clip(streched_c, new_min, new_max)).astype(np.uint8)


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


def lab_colour_stretch(im):
    #covert to CIE-Lab
    im_lab = cv2.cvtColor(im, cv2.COLOR_BGR2Lab)
    im_lab = im_lab.astype(np.int32)
    l, a, b = cv2.split(im_lab)
    l = np.round(l * 100/255).astype(np.int32)
    a = a - 128
    b = b - 128

    #linear slide stretching L
    l_values = np.sort(l.flatten())
    val, counts = np.unique(l_values, return_counts=True)
    mode = val[np.argmax(counts)]
    mode_index = np.where(l_values == mode)[0][0]

    i_min = int(l_values[int(mode_index * 0.001)])
    i_max = int(l_values[int(-(len(l_values) - mode_index) * 0.001)])

    l = histogram_stretch(l, i_min, i_max, 0, 100)

    #stretch a and b
    s_curve = np.vectorize(lambda x: x * (1.3 ** (1 - abs(x/128))))
    a = s_curve(a)
    b = s_curve(b)

    #covert back to BGR
    l = (l.astype(np.float32) * 255/100).astype(np.uint8)
    a = np.round(a + 128).astype(np.uint8)
    b = np.round(b + 128).astype(np.uint8)

    im_lab = cv2.merge((l, a, b))
    im = cv2.cvtColor(im_lab, cv2.COLOR_Lab2BGR)
    return im


def RGHS_enchancement(im_path):
    im = cv2.imread(im_path)
    h, w, _ = im.shape
    im = cv2.resize(im, (w//4, h//4)) #FOR DISPLAY PURPOSES, remove later

    b, g, r = cv2.split(im)

    #b and g channel equilisation
    b = grey_world_equalisation(b)
    g = grey_world_equalisation(g)

    #Relative global histogram stretching
    b = histogram_stretch(b, *histogram_parameter_estimation(b, "b")) 
    g = histogram_stretch(g, *histogram_parameter_estimation(g, "g"))
    r = histogram_stretch(r, *histogram_parameter_estimation(r, "r"))
    
    im = cv2.merge((b, g, r))
    im = lab_colour_stretch(im)
    return im

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


def compare_enhancement(im_path):
    fig, axes = plt.subplots(1, 2, figsize = (14, 6))

    im = cv2.imread(im_path)
    h, w, _ = im.shape
    im = cv2.resize(im, (w//4, h//4)) #FOR DISPLAY PURPOSES, remove later
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    axes[0].imshow(im)

    enhanced_im = RGHS_enchancement(im_path)
    enhanced_im = cv2.cvtColor(enhanced_im, cv2.COLOR_BGR2RGB)
    axes[1].imshow(enhanced_im)

    plt.show()


if __name__ == "__main__":
    im = RGHS_enchancement("data/images/im0.JPG")
    cv2.imshow("im", im)
    cv2.waitKey(0)