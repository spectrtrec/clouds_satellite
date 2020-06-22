import os
from unittest import TestCase
import numpy as np
import pandas as pd
from test import build_masks_list, build_checkpoints_list, build_masks_list
from utils.utils import load_yaml, get_project_root
from pathlib import Path

class TestInference(TestCase):
    def setUp(self):
        self.inference_yaml = load_yaml(os.path.join(
            get_project_root(), "tests/test_inference.yaml"))

    def test_build_checkpoints_list(self):
        checkpoints = build_checkpoints_list(self.inference_yaml)
        self.assertIsInstance(checkpoints, list)
        self.assertEqual(len(checkpoints), 1)

    def test_best_checkpoints_list(self):
        self.inference_yaml["SUBMIT_BEST"] = True
        checkpoints = build_checkpoints_list(self.inference_yaml)
        self.assertIsInstance(checkpoints, list)
        self.assertEqual(len(checkpoints), 1)
        self.assertEqual(
            str(checkpoints[0]), 'tests/checkpoints/best/efficientnet-b3_fold0.pth')

    def test_build_masks_list(self):
        experiment_folder = Path('tests/test_inference.yaml').parents[0]
        dict_dir = Path(experiment_folder, self.inference_yaml["DICT_FOLDER"])
        masks_list = build_masks_list(dict_dir)
        self.assertEqual(len(masks_list), 1)
        self.assertEqual(
            str(masks_list[0]), 'tests/dict_efficientnet-b3/efficientnet-b3_fold0_epoch0.pkl')
