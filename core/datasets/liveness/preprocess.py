#########################################################
#             .',;::::;,'.                 
#          .';:cccccccccccc:;,.              
#       .;cccccccccccccccccccccc;           --------------
#     .:cccccccccccccccccccccccccc:.        Project name :      prj.FaceAttend
#   .;ccccccccccccc;.:dddl:.;ccccccc;.      Author       :      Nguyen Dac Duong
#  .:ccccccccccccc;OWMKOOXMWd;ccccccc:.     File name    :      preprocess.py
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
import random
from pathlib import Path


class FaceMaskDatasetPreprocessor:
    def __init__(self, raw_dataset_dir, output_dir, train_split=0.8, seed=42):
        self.raw_dataset = Path(raw_dataset_dir).expanduser()
        self.output_dir = Path(output_dir).expanduser()
        self.train_split = train_split
        self.seed= seed

        self.with_mask_dir = self.raw_dataset / "with_mask"
        self.without_mask_dir = self.raw_dataset / "without_mask"

        self.class_map = {
            "no_mask": self.without_mask_dir,
            "mask": self.with_mask_dir / "Mask",
            "mask_chin": self.with_mask_dir / "Mask_Chin",
            "mask_mouth_chin": self.with_mask_dir / "Mask_Mouth_Chin",
            "mask_nose_mouth": self.with_mask_dir / "Mask_Nose_Mouth",
        }

    def create_dir(self):

        for split in ["train", "val"]:
            for cls in self.class_map.keys():
                (self.output_dir / split / cls).mkdir(parents=True, exist_ok=True)

    def split_and_copy(self, files, class_name):

        random.shuffle(files)

        split_idx = int(len(files) * self.train_split)
        train_files = files[:split_idx]
        val_files = files[split_idx:]

        for f in train_files:
            dest_train = self.output_dir / "train" / class_name / f.name
            shutil.copy(f, dest_train)

        for f in val_files:
            dest_val = self.output_dir / "val" / class_name / f.name
            shutil.copy(f, dest_val)

    def process_class(self, class_name, path):

        if not path.exists():
            print(f"Skip missing folder: {path}")
            return
        
        files = list(path.glob("*.*"))
        print(f"{class_name}: {len(files)} images")

        self.split_and_copy(files, class_name)

    def run(self):

        random.seed(self.seed)

        self.create_dir()
        
        for class_name, path in self.class_map.items():
            self.process_class(class_name, path)
        
        print("FaceMask Detection preprocessing completed!")


if __name__ == "__main__":

    RAW_DATASET = Path("~/ml-dataset/facemask-detection").expanduser()
    OUTPUT_DIR = Path("~/ml-dataset/processed-face-mask").expanduser()

    try:
        processor = FaceMaskDatasetPreprocessor(
            raw_dataset_dir=RAW_DATASET,
            output_dir=OUTPUT_DIR,
        )
        processor.run()

    except Exception as e:
        print(f"Error during dataset preprocessing: {e}")
