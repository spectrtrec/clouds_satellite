set -e
PYTHONPATH="${PROJECT_ROOT}" \
python createsub.py --config="configs/efficientnet-b3/submission_efficientnet-b3.yaml"
