set -e
PYTHONPATH="${PROJECT_ROOT}" \
python train.py --config="configs/efficientnet-b3/efficientnet-b3.yaml"
