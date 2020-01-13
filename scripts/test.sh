set -e
PYTHONPATH="${PROJECT_ROOT}" \
python test.py --config="configs/resnet34/resnet34.yaml"
