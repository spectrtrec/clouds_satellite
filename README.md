# Clouds Satellite
Classify cloud structures from satellites.
## How to run?
```
bash sh scripts/train.sh
bash sh scripts/test.sh
bash sh scripts/submit.sh
```
## How to run tensorboard?
```
tensorboard --logdir=configs/`config_folder`/log/
```
## References
* Folder `segmentation_models_pytorch` was coped from * https://github.com/qubvel/segmentation_models.pytorch
* Based on code of https://github.com/amirassov/kaggle-pneumothorax
* Based on code of https://github.com/sneddy/pneumothorax-segmentation
* https://github.com/albu/albumentations
