import os
import sys

import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold

sys.path.insert(0, "..")

opj = os.path.join
ope = os.path.exists


def create_random_split(train_meta, mask_count, n_splits=4):
    train_meta = train_meta.copy()
    split_dir = opj(DATA_DIR, "split_stage1", "random_folds%d" % n_splits)
    os.makedirs(split_dir, exist_ok=True)

    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=100)
    for idx, (train_indices, valid_indices) in enumerate(kf.split(train_meta)):
        train_split_df = train_meta.loc[train_indices]
        valid_split_df = train_meta.loc[valid_indices]

        fname = opj(split_dir, "random_train_cv%d.csv" % idx)
        print(
            "train: create split file: %s; samples: %s"
            % (fname, train_split_df.shape[0])
        )
        train_split_df.to_csv(fname, index=False)

        fname = opj(split_dir, "random_valid_cv%d.csv" % idx)
        print(
            "valid: create split file: %s; samples: %s"
            % (fname, valid_split_df.shape[0])
        )
        valid_split_df.to_csv(fname, index=False)


if __name__ == "__main__":
    print("%s: calling main function ... " % os.path.basename(__file__))
    train = pd.read_csv(opj("cloudsimg", "train.csv"))
    train["label"] = train["Image_Label"].apply(lambda x: x.split("_")[1])
    train["im_id"] = train["Image_Label"].apply(lambda x: x.split("_")[0])

    sub = pd.read_csv(opj("cloudsimg", "sample_submission.csv"))
    sub["label"] = sub["Image_Label"].apply(lambda x: x.split("_")[1])
    sub["im_id"] = sub["Image_Label"].apply(lambda x: x.split("_")[0])

    # split data
    id_mask_count = (
        train.loc[train["EncodedPixels"].isnull() == False, "Image_Label"]
        .apply(lambda x: x.split("_")[0])
        .value_counts()
        .sort_index()
        .reset_index()
        .rename(columns={"index": "img_id", "Image_Label": "count"})
    )
    ids = id_mask_count["img_id"].values
    create_random_split(ids,, n_splits=4)
