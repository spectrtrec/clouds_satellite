set -e
PYTHONPATH="${PROJECT_ROOT}" \
python createsub.py --config="configs/resnet34/resnet34.yaml"
