set -e
PYTHONPATH="${PROJECT_ROOT}" \
python train.py --config="configs/resnet34/resnet34.yaml"
