import os
import pydoc

import pandas as pd

from torchmethods.dataloader import make_data
from torchmethods.metrics import Metrics
from utils.utils import get_augmentation


class DataFactory:
    def __init__(
        self,
        train_df,
        preprocessing,
        params,
    ):
        self.train_df = train_df
        self.preprocessing = preprocessing
        self.params = params

    def make_train_loader(self, fold=None, mode="train"):
        img_ids = self.make_filenames(mode, fold)
        train_transform = get_augmentation(mode)
        return make_data(
            self.train_df,
            img_ids,
            mode,
            train_transform,
            self.preprocessing,
            **self.params
        )

    def make_val_loader(self, fold, mode="validation"):
        img_ids = self.make_filenames(mode, fold)
        transform_val = get_augmentation(mode)
        return make_data(
            self.train_df,
            img_ids,
            mode,
            transform_val,
            self.preprocessing,
            **self.params
        )

    def make_test_loader(self, fold=None, mode="test"):
        img_ids = self.make_filenames(mode, fold)
        transform_test = get_augmentation(mode)
        return make_data(
            self.train_df,
            img_ids,
            mode,
            transform_test,
            self.preprocessing,
            **self.params
        )

    def make_filenames(self, mode, fold):
        if mode == "test":
            df_img = pd.read_csv('folds/test_id.csv')
        if mode == 'train':
            df_img = pd.read_csv(
                f"folds/train_fold_{fold}.csv", names=["im_id"], header=0
            )
        elif mode == 'validation':
            df_img = pd.read_csv(
                f"folds/validation_fold_{fold}.csv", names=["im_id"], header=0
            )
        return df_img['im_id'].values


class MetricFactory:
    def __init__(self, params: dict):
        self.params = params

    @staticmethod
    def get_metric_name(metric):
        return metric.split('.')[-1]

    def make_metrics(self) -> Metrics:
        return Metrics(
            {
                self.get_metric_name(metric): pydoc.locate(metric)(**params)
                for metric, params in self.params['metrics'].items()
            }
        )
