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
from utils.utils import load_yaml, mask2rle, sigmoid


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)
    return parser.parse_args()


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
        predict, num_predict = post_process(mask, 0.7, 10000)
        if num_predict == 0:
            encoded_pixels.append("")
        else:
            r = mask2rle(predict)
            encoded_pixels.append(r)
    return encoded_pixels


def buid_submission(sub, encoded_pixels, sub_name):
    sub["EncodedPixels"] = pd.Series(encoded_pixels)
    sub.to_csv(
        f"sumbissions/{sub_name}", columns=["Image_Label", "EncodedPixels"], index=False
    )
    return sub


def load_mask_dict(cfg):
    reshape_mode = cfg.get("RESHAPE_MODE", False)
    if "MASK_DICT" in cfg:
        result_path = Path(cfg["MASK_DICT"])
        with open(result_path, "rb") as handle:
            mask_dict = pickle.load(handle)
    return mask_dict


def main():
    args = parse_args()
    config_path = Path(args.config.strip("/")) 
    sub_config = load_yaml(config_path)
    sample_sub = pd.read_csv(sub_config["SAMPLE_SUB"])
    mask_dict = load_mask_dict(sub_config)
    rle_dict = build_rle_dict(mask_dict)
    buid_submission(sample_sub, rle_dict, sub_config["SUB_FILE"])


if __name__ == "__main__":
    main()
