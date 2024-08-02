import cv2
import os
from tqdm import tqdm
import random
import matplotlib.pyplot as plt
import multiprocessing as mp
import functools


def enhance_images(enhancment_func, input_dir, output_dir, limit=None):
    count = 0
    im_names = list(os.listdir(input_dir))
    random.shuffle(im_names)

    #empty folder first
    for im_name in os.listdir(output_dir):
        os.remove(output_dir + "/" + im_name)

    for im_name in tqdm(im_names, bar_format="{l_bar}{bar:30}{r_bar}"):
        im = cv2.imread(input_dir + "/" + im_name)
        im = enhancment_func(im)
        cv2.imwrite(output_dir + "/" + im_name, im)

        count += 1
        if limit and count >= limit: break

def process_im(im_name, enhancment_func, input_dir, output_dir):
    im = cv2.imread(input_dir + "/" + im_name)
    im = enhancment_func(im)
    cv2.imwrite(output_dir + "/" + im_name, im)

def enhance_images_mp(enhancment_func, input_dir, output_dir):
    im_names = list(os.listdir(input_dir))

    partial = functools.partial(process_im, enhancment_func=enhancment_func, input_dir=input_dir, output_dir=output_dir)
    pool = mp.Pool(mp.cpu_count())
    for _ in tqdm(pool.imap_unordered(partial, im_names), total=len(im_names), bar_format="{l_bar}{bar:30}{r_bar}"):
        pass

    pool.close()
    pool.join()



def uie_comparison(orig_dir, enhanced_dir):
    im_names = list(os.listdir(enhanced_dir))
    random.shuffle(im_names)

    for im_name in im_names:
        fig, axes = plt.subplots(1, 2, figsize = (20, 10))

        orig_im = cv2.imread(orig_dir + "/" + im_name)
        enha_im = cv2.imread(enhanced_dir + "/" + im_name)

        orig_im = cv2.cvtColor(orig_im, cv2.COLOR_BGR2RGB)
        enha_im = cv2.cvtColor(enha_im, cv2.COLOR_BGR2RGB)

        #plot original image
        ax = axes[0]
        ax.set_title(f"Original")
        ax.imshow(orig_im)
        ax.set_xticks([])
        ax.set_yticks([])
            
        #plot enhanced image
        ax = axes[1]
        ax.set_title(f"Enhanced")
        ax.imshow(enha_im)
        ax.set_xticks([])
        ax.set_yticks([])

        plt.show()



clahe = cv2.createCLAHE(clipLimit=1.75, tileGridSize=(16, 16))

def apply_clahe_rgb(im):
    b, g, r =cv2.split(im)
    new_r, new_g, new_b = map(clahe.apply, [r, g, b])
    new_im = cv2.merge([new_b, new_g, new_r])
    return new_im

def apply_clahe_hls(im):
    im = cv2.cvtColor(im, cv2.COLOR_BGR2HLS)
    h, l, s =cv2.split(im)
    new_im = cv2.merge([h, clahe.apply(l), s])
    new_im = cv2.cvtColor(new_im, cv2.COLOR_HLS2BGR)
    return new_im

if __name__ == "__main__":
    image_dir = "data/images"
    output = "data/images_clahe_mp"

    enhance_images_mp(apply_clahe_hls, image_dir, output)
    #uie_comparison(image_dir, output)