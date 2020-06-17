set -e
PYTHONPATH="${PROJECT_ROOT}" \
python test.py --config="configs/efficientnet-b3/inference_efficientnet-b3.yaml"
