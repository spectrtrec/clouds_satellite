import logging
import os

import torch
from tensorboardX import SummaryWriter

from callbacks.metacallback import Callback, Callbacks


class TensorBoard(Callback):
    def __init__(self, log_dir: str, optimizer):
        super().__init__()
        self.log_dir = log_dir
        self.optimizer = optimizer
        self.writer = None

    def on_train_begin(self, fold):
        os.makedirs(self.log_dir, exist_ok=True)
        self.writer = SummaryWriter(self.log_dir)

    def on_epoch_begin(self, epoch):
        pass

    def on_epoch_end(self, epoch, train_metrics, val_metrics):
        for k, v in train_metrics.items():
            self.writer.add_scalar(f"train/{k}", float(v), global_step=epoch)

        for k, v in val_metrics.items():
            self.writer.add_scalar(f"val/{k}", float(v), global_step=epoch)

        for idx, param_group in enumerate(self.optimizer.param_groups):
            lr = param_group["lr"]
            self.writer.add_scalar(f"group{idx}/lr", float(lr), global_step=epoch)

    def on_train_end(self):
        self.writer.close()


class Logger(Callback):
    def __init__(self, log_dir, optimizer):
        super().__init__()
        self.log_dir = log_dir
        self.optimizer = optimizer
        self.logger = None

    def on_train_begin(self, fold):
        os.makedirs(self.log_dir, exist_ok=True)
        self.logger = self._get_logger(str(self.log_dir / f"logs_fold_{fold}.txt"))
        self.logger.info(f"Starting training fold: {fold} \n")

    def on_train_end(self):
        pass

    def on_epoch_begin(self, epoch):
        self.logger.info(
            f"Epoch {epoch} | "
            f'optimizer "{self.optimizer.__class__.__name__}" | '
            f"lr {self.current_lr}"
        )

    def on_epoch_end(self, epoch, train_metrics, val_metrics):
        self.logger.info("Train metrics: " + self._get_metrics_string(train_metrics))
        self.logger.info(
            "Valid metrics: " + self._get_metrics_string(val_metrics) + "\n"
        )

    @staticmethod
    def _get_logger(log_path):
        logger = logging.getLogger(log_path)
        logger.setLevel(logging.DEBUG)
        fh = logging.FileHandler(log_path)
        fh.setLevel(logging.INFO)
        formatter = logging.Formatter("[%(asctime)s] %(message)s")
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        return logger

    @property
    def current_lr(self):
        res = []
        for param_group in self.optimizer.param_groups:
            res.append(param_group["lr"])
        if len(res) == 1:
            return res[0]
        return res

    @staticmethod
    def _get_metrics_string(metrics):
        return " | ".join("{}: {:.5f}".format(k, v) for k, v in metrics.items())
