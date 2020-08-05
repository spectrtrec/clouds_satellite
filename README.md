# Clouds Satellite
Classify cloud structures from satellites.
## How to run:
```
bash sh scripts/train.sh
bash sh scripts/test.sh
bash sh scripts/submit.sh
```
## tensorboard:
```
tensorboard --logdir=configs/`config_folder`/log/
```
## docker
```
make build
make run
make exec
```
## References
* https://github.com/albu/albumentations
* Based on code of https://github.com/amirassov/kaggle-pneumothorax
* Based on code of https://github.com/sneddy/pneumothorax-segmentation