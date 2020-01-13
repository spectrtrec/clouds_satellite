import albumentations as albu
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from albumentations import torch as AT
from torch.optim import lr_scheduler
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, StepLR
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torch.utils.data.sampler import SubsetRandomSampler

from utils.utils import *


class CloudDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        img_path: str,
        img_ids: np.array,
        mode,
        transforms=albu.Compose([albu.HorizontalFlip(), AT.ToTensor()]),
        preprocessing=None,
    ):
        self.df = df
        self.data_folder = img_path
        self.img_ids = img_ids
        self.transforms = transforms
        self.preprocessing = preprocessing
        self.mode = mode

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        if self.mode in ["train", "validation"]:
            image_name = self.img_ids[idx]
            mask = make_mask(self.df, image_name)
            img = cv2.cvtColor(
                cv2.imread(os.path.join(self.data_folder, image_name)),
                cv2.COLOR_BGR2RGB,
            )
            augmented = self.transforms(image=img, mask=mask)
            img = augmented["image"]
            mask = augmented["mask"]
            if self.preprocessing:
                preprocessed = self.preprocessing(image=img, mask=mask)
                img = preprocessed["image"]
                mask = preprocessed["mask"]
            return img, mask, image_name
        else:
            image_name = self.img_ids[idx]
            img = cv2.cvtColor(
                cv2.imread(os.path.join(self.data_folder, image_name)),
                cv2.COLOR_BGR2RGB,
            )
            augmented = self.transforms(image=img)
            img = augmented["image"]
            if self.preprocessing:
                preprocessed = self.preprocessing(image=img)
                img = preprocessed["image"]
            return img, image_name
