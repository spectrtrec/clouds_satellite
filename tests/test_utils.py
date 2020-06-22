import os
from unittest import TestCase

import numpy as np
import pandas as pd

from utils.utils import prepare_train, get_project_root, DataframeError, get_pkl_file_name


class TestUtils(TestCase):
    def test_prepare_train(self):
        train_df, submission = prepare_train(
            os.path.join(get_project_root(), "cloudsimg"))
        self.assertIsInstance(train_df, pd.DataFrame)
        self.assertIsInstance(submission, pd.DataFrame)

    def test_prepare_train_raise(self):
        self.assertRaises(DataframeError,
                          lambda: prepare_train(os.path.join(get_project_root(), "")))

    def test_pkl_file_name(self):
        pth_path = 'configs/efficientnet-b3/checkpoints/fold0/efficientnet-b3_fold0_epoch1.pth'
        path = get_pkl_file_name(pth_path)
        self.assertIsInstance(path, str)
        self.assertEqual('efficientnet-b3_fold0_epoch1.pkl', path)

    def test_project_root(self):
        pass
