import numpy as np
import pandas as pd

import argparse
import pickle
from tqdm import tqdm
from pathlib import Path

import cv2
import os
import numpy as np
import pandas as pd
from collections import defaultdict
from utils.utils import (
    load_yaml,
    mask2rle,
    sigmoid,
    draw_convex_hull,
    new_make_mask,
    post_process_minsize,
)

min_size = [10000 ,10000, 10000, 10000]
def post_process(probability, threshold, min_size):
    """
    Post processing of each predicted mask, components with lesser number of pixels
    than `min_size` are ignored
    """
    mask = cv2.threshold(probability, threshold, 1, cv2.THRESH_BINARY)[1]
    num_component, component = cv2.connectedComponents(mask.astype(np.uint8))
    predictions = np.zeros((350, 525), np.float32)
    num = 0
    for c in range(1, num_component):
        p = component == c
        if p.sum() > min_size:
            predictions[p] = 1
            num += 1
    return predictions, num


def build_rle_dict(mask_dict):
    encoded_pixels = []
    for name, mask in tqdm(mask_dict.items()):
        if mask.shape != (350, 525):
            mask = cv2.resize(mask, dsize=(525, 350), interpolation=cv2.INTER_LINEAR)
        predict, num_predict = post_process(mask, 0.45, 10000)
        if num_predict == 0:
            encoded_pixels.append("")
        else:
            r = mask2rle(predict)
            encoded_pixels.append(r)
    return encoded_pixels


def buid_submission(sub, encoded_pixels):
    sub["EncodedPixels"] = pd.Series(encoded_pixels)
    sub.to_csv(
        "sumbissions/submission.csv",
        columns=["Image_Label", "EncodedPixels"],
        index=False,
    )
    return sub


def post_process_new(sub):
    model_class_names = ["Fish", "Flower", "Gravel", "Sugar"]
    mode = "convex"

    img_label_list = []
    enc_pixels_list = []
    test_imgs = os.listdir("cloudsimg/test_images")
    for test_img_i, test_img in enumerate(tqdm(test_imgs)):
        for class_i, class_name in enumerate(model_class_names):

            path = os.path.join("cloudsimg/test_images", test_img)
            img = cv2.imread(path).astype(np.float32)
            img = cv2.resize(img, dsize=(525, 350), interpolation=cv2.INTER_LINEAR)
            img = img / 255.0
            img2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            img_label_list.append(f"{test_img}_{class_name}")

            mask = new_make_mask(sub, test_img + "_" + class_name, shape=(350, 525))
            if True:
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


def load_mask_dict(cfg):
    reshape_mode = cfg.get("RESHAPE_MODE", False)
    if "MASK_DICT" in cfg:
        result_path = Path(cfg["MASK_DICT"])
        with open(result_path, "rb") as handle:
            mask_dict = pickle.load(handle)
    return mask_dict


def main():
    config_path = Path(
        "configs/efficientnet-b3/submission_efficientnet-b3.yaml".strip("/")
    )
    sub_config = load_yaml(config_path)

    sample_sub = pd.read_csv(sub_config["SAMPLE_SUB"])
    print("start loading mask results....")
    mask_dict = load_mask_dict(sub_config)
    rle_dict = build_rle_dict(mask_dict)
    buid_submission(sample_sub, rle_dict)


if __name__ == "__main__":
    main()
    #sub = pd.read_csv('sumbissions/submission_new.csv')
    #post_process_new(sub)
