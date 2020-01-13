from abc import ABCMeta, abstractmethod


class Callback(metaclass=ABCMeta):
    """
    Abstract base class used to build new callbacks.
    """

    def __init__(self):
        self.runner = None
        self.metrics = None

    @abstractmethod
    def on_epoch_begin(self, epoch):
        pass

    @abstractmethod
    def on_epoch_end(self, epoch, train_metrics, val_metrics):
        pass

    @abstractmethod
    def on_train_begin(self, fold):
        pass

    @abstractmethod
    def on_train_end(self):
        pass

class Callbacks(Callback):
    def __init__(self, callbacks):
        super().__init__()
        if isinstance(callbacks, Callbacks):
            self.callbacks = callbacks.callbacks
        if isinstance(callbacks, list):
            self.callbacks = callbacks
        else:
            self.callbacks = []

    def on_epoch_begin(self, epoch):
        for callback in self.callbacks:
            callback.on_epoch_begin(epoch)

    def on_epoch_end(self, epoch, train_metrics, val_metrics):
        for callback in self.callbacks:
            callback.on_epoch_end(epoch, train_metrics, val_metrics)

    def on_train_begin(self, fold):
        for callback in self.callbacks:
            callback.on_train_begin(fold)

    def on_train_end(self):
        for callback in self.callbacks:
            callback.on_train_end()
