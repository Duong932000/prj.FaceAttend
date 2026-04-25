#########################################################
#             .',;::::;,'.                 
#          .';:cccccccccccc:;,.              
#       .;cccccccccccccccccccccc;           --------------
#     .:cccccccccccccccccccccccccc:.        Project name :      prj.FaceAttend
#   .;ccccccccccccc;.:dddl:.;ccccccc;.      Author       :      Nguyen Dac Duong
#  .:ccccccccccccc;OWMKOOXMWd;ccccccc:.     File name    :      fpreprocess.py
# .:ccccccccccccc;KMMc;cc;xMMc;ccccccc:.    Description  :      
# ,cccccccccccccc;MMM.;cc;;WW:;cccccccc,    --------------
# :cccccccccccccc;MMM.;cccccccccccccccc:
# :ccccccc;oxOOOo;MMM000k.;cccccccccccc:
# cccccc;0MMKxdd:;MMMkddc.;cccccccccccc;
# ccccc;XMO';cccc;MMM.;cccccccccccccccc'
# ccccc;MMo;ccccc;MMW.;ccccccccccccccc;
# ccccc;0MNc.ccc.xMMd;ccccccccccccccc;
# cccccc;dNMWXXXWM0:;cccccccccccccc:,
# cccccccc;.:odl:.;cccccccccccccc:,.
# ccccccccccccccccccccccccccccc:'.
# :ccccccccccccccccccccccc:;,..
#  ':cccccccccccccccc::;,.
#########################################################


import os
import shutil
import pandas as pd
import random

# =========================
# CONFIG
# =========================
CSV_PATH = "/home/dacduong/ml-data/kaggle/anti-spoofing-live/real_30.csv"
IMG_DIR  = "/home/dacduong/ml-data/kaggle/anti-spoofing-live/sample"

OUT_DIR  = "/tmp/liveness_dataset_processed"

TRAIN_SPLIT = 0.8

# =========================
# SETUP
# =========================
def create_dirs():
    for split in ["train", "val"]:
        for cls in ["real", "fake"]:
            os.makedirs(os.path.join(OUT_DIR, split, cls), exist_ok=True)


def main():
    df = pd.read_csv(CSV_PATH)

    # shuffle dataset
    df = df.sample(frac=1).reset_index(drop=True)

    split_idx = int(len(df) * TRAIN_SPLIT)

    train_df = df[:split_idx]
    val_df   = df[split_idx:]

    process_split(train_df, "train")
    process_split(val_df, "val")

    print("Preprocessing done!")


def process_split(df, split):
    for _, row in df.iterrows():

        img_name = row["image"]
        label    = row["label"]

        src = os.path.join(IMG_DIR, img_name)

        if not os.path.exists(src):
            continue

        cls = "real" if label == 1 else "fake"

        dst = os.path.join(OUT_DIR, split, cls, img_name)
        if not os.path.exists(dst):
            os.symlink(src, dst)


if __name__ == "__main__":
    create_dirs()
    main()