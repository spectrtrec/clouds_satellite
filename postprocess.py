import pandas as pd
import numpy as np
import cv2
from tqdm import tqdm
import os
class_names = ["Fish", "Flower", "Sugar", "Gravel"]


def rle_decode(mask_rle: str = "", shape=(1400, 2100)):
    """
    Decode rle encoded mask.
    
    :param mask_rle: run-length as string formatted (start length)
    :param shape: (height, width) of array to return 
    Returns numpy array, 1 - mask, 0 - background
    """
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1

    return img.reshape(shape, order="F")


def mask2rle(img):
    """
    Convert mask to rle.
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    """
    pixels = img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)


def draw_convex_hull(mask, mode="convex"):

    img = np.zeros(mask.shape)
    contours, hier = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for c in contours:
        if mode == "rect":  # simple rectangle
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), -1)
        elif mode == "convex":  # minimum convex hull
            hull = cv2.convexHull(c)
            cv2.drawContours(img, [hull], 0, (255, 255, 255), -1)
        elif mode == "approx":
            epsilon = 0.02 * cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, epsilon, True)
            cv2.drawContours(img, [approx], 0, (255, 255, 255), -1)
        else:  # minimum area rectangle
            rect = cv2.minAreaRect(c)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            cv2.drawContours(img, [box], 0, (255, 255, 255), -1)
    return img / 255.0


def make_mask(df, image_label, shape=(1400, 2100), cv_shape=(525, 350), debug=False):
    """
    Create mask based on df, image name and shape.
    """
    if debug:
        print(shape, cv_shape)
    df = df.set_index("Image_Label")
    encoded_mask = df.loc[image_label, "EncodedPixels"]
    #     print('encode: ',encoded_mask[:10])
    mask = np.zeros((shape[0], shape[1]), dtype=np.float32)
    if encoded_mask is not np.nan:
        mask = rle_decode(encoded_mask, shape=shape)  # original size

    return cv2.resize(mask, cv_shape)


min_size = [10000, 10000, 10000, 10000]


def post_process_minsize(mask, min_size):
    """
    Post processing of each predicted mask, components with lesser number of pixels
    than `min_size` are ignored
    """

    num_component, component = cv2.connectedComponents(mask.astype(np.uint8))
    predictions = np.zeros(mask.shape)
    num = 0
    for c in range(1, num_component):
        p = component == c
        if p.sum() > min_size:
            predictions[p] = 1
            num += 1
    return predictions  # , num


if __name__ == "__main__":
    model_class_names = ["Fish", "Flower", "Gravel", "Sugar"]
    mode = "convex"  # choose from 'rect', 'min', 'convex' and 'approx'
    sub = pd.read_csv('sumbissions/submission.csv')
    img_label_list = []
    enc_pixels_list = []
    test_imgs = os.listdir('clouds_resized/test_images_525')
    for test_img_i, test_img in enumerate(tqdm(test_imgs)):
        for class_i, class_name in enumerate(model_class_names):

            path = os.path.join('clouds_resized/test_images_525', test_img)
            img = cv2.imread(path).astype(
                np.float32
            )  # use already-resized ryches' dataset
            img = img / 255.0
            img2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            img_label_list.append(f"{test_img}_{class_name}")

            mask = make_mask(sub, test_img + "_" + class_name, shape=(350, 525))
            if True:
                # if class_name == 'Flower' or class_name =='Sugar': # you can decide to post-process for some certain classes
                mask = draw_convex_hull(mask.astype(np.uint8), mode=mode)
            mask[img2 <= 2 / 255.0] = 0
            mask = post_process_minsize(mask, min_size[class_i])

            if mask.sum() == 0:
                enc_pixels_list.append(np.nan)
            else:
                mask = np.where(mask > 0.5, 1.0, 0.0)
                enc_pixels_list.append(mask2rle(mask))
    submission_df = pd.DataFrame(
        {"Image_Label": img_label_list, "EncodedPixels": enc_pixels_list}
    )
    submission_df.to_csv("sub_convex.csv", index=None)

