import torchvision
import torchvision.transforms as transforms
import torch
from torch.utils.data import TensorDataset, DataLoader, Dataset
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data.sampler import SubsetRandomSampler
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, CosineAnnealingLR

import albumentations as albu
from albumentations import torch as AT
import numpy as np
import pandas as pd
import cv2
from utils.utils import *


class CloudDataset(Dataset):
    def __init__(self, df: pd.DataFrame, img_path: str, datatype: str, img_ids: np.array,
                 transforms=albu.Compose(
                     [albu.HorizontalFlip(), AT.ToTensor()]),
                 preprocessing=None):
        self.df = df
        if datatype != 'test':
            self.data_folder = img_path
        else:
            self.data_folder = img_path
        self.img_ids = img_ids
        self.transforms = transforms
        self.preprocessing = preprocessing
    
    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        image_name = self.img_ids[idx]
        mask = make_mask(self.df, image_name)
        image_path = os.path.join(self.data_folder, image_name)
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        augmented = self.transforms(image=img, mask=mask)
        img = augmented['image']
        mask = augmented['mask']
        if self.preprocessing:
            preprocessed = self.preprocessing(image=img, mask=mask)
            img = preprocessed['image']
            mask = preprocessed['mask']
        return img, mask, image_name

