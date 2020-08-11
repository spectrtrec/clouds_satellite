# Clouds Satellite
Classify cloud structures from satellites.
## How to run:
```
bash sh scripts/train.sh
bash sh scripts/test.sh
bash sh scripts/submit.sh
```
## Tensorboard:
```
tensorboard --logdir=configs/`config_folder`/log/
```
## Docker
```
make build
make run
make exec
```
## Configs file structure
    ├── configs
    │   ├── config_folder
    │   │   |
    │   │   │── config_name.yaml
    │   │   │── inference_config_name.yaml
    │   │   │── submit_config_name.yaml
    │   │   ├── checkpoints
    │   │   │      ├── foldi
    │   │   │      │   ├── topk_checkpoint_from_fold_i_epoch_k.pth 
    │   │   │      ├── best
    │   │   │      |   ├── best_checkpoint_foldi.pth
    │   │   ├── log
    |   |   |      ├── logs_fold_i.txt
    |   |   ├── dict_efficientnet-b3
    |   |   |       ├── experiment_foldi_epochi.pkl
## Install
```bash
pip install -r requirements.txt
```
## References
* https://github.com/albu/albumentations
* https://github.com/qubvel/segmentation_models.pytorch
* https://github.com/amirassov/kaggle-pneumothorax
* https://github.com/sneddy/pneumothorax-segmentation
