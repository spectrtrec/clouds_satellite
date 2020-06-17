import pydoc

from torchmethods.dataloader import make_data
from torchmethods.metrics import Metrics


class DataFactory:
    def __init__(
        self,
        train_df,
        img_id_train,
        img_id_val,
        transform_train,
        transform_val,
        preprocessing,
        params,
    ):
        self.train_df = train_df
        self.img_id_train = img_id_train
        self.img_id_val = img_id_val
        self.transform_train = transform_train
        self.transform_val = transform_val
        self.preprocessing = preprocessing
        self.params = params

    def make_train_loader(self, mode="train"):
        return make_data(
            self.train_df,
            self.img_id_train,
            mode,
            self.transform_train,
            self.preprocessing,
            **self.params
        )

    def make_val_loader(self, mode="validation"):
        return make_data(
            self.train_df,
            self.img_id_val,
            mode,
            self.transform_val,
            self.preprocessing,
            **self.params
        )


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
