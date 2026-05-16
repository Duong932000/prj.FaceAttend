#########################################################
#             .',;::::;,'.                 
#          .';:cccccccccccc:;,.              
#       .;cccccccccccccccccccccc;           --------------
#     .:cccccccccccccccccccccccccc:.        Project name :      prj.FaceAttend
#   .;ccccccccccccc;.:dddl:.;ccccccc;.      Author       :      Nguyen Dac Duong
#  .:ccccccccccccc;OWMKOOXMWd;ccccccc:.     File name    :      download_model.py
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
import zipfile
import requests
from tqdm import tqdm
from pathlib import Path
from core.face_recognition.utils.config import load_common_config

class ModelDownloader:

    def __init__(self):

        root_dir, config = load_common_config()
        self.cfg = config["face_recognition"]
        
        # model parameters
        self.model_name = self.cfg["model"]["name"]
        self.download_url = self.cfg["model"]["download_url"]

        # paths
        self.model_root_dir = Path(root_dir / "mdl" / "face_recognition")
        self.model_root_dir.mkdir(parents=True, exist_ok=True)

        self.model_dir = (self.model_root_dir / self.model_name)
        self.zip_path = (self.model_root_dir / f"{self.model_name}.zip")

    def download_file(self):

        if self.model_dir.exists():

            print("[INFO] Model already exists:")
            print(f"{self.model_dir}")

            return

        print(f"[INFO] Downloading {self.model_name}....")

        response = requests.get(self.download_url, stream=True)

        response.raise_for_status()

        total_size = int(response.headers.get( "content-length", 0))

        with open(self.zip_path, "wb") as file:
            with tqdm(total=total_size, unit="B", unit_scale=True, desc="Downloading") as progress_bar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        file.write(chunk)
                        progress_bar.update(len(chunk))

        print("[INFO] Download completed")

    def extract_zip(self):

        print("[INFO] Extracting model...")

        self.model_dir.mkdir(parents=True, exist_ok=True)

        with zipfile.ZipFile(self.zip_path, "r",
        ) as zip_ref:

            zip_ref.extractall(
                self.model_dir
            )

        print(
            "[INFO] Extraction completed"
        )

    def cleanup(self):

        if self.zip_path.exists():
            self.zip_path.unlink()
            print("[INFO] Removed zip file")

    def verify_model(self):

        required_files = ["det_10g.onnx", "2d106det.onnx", "genderage.onnx", "w600k_r50.onnx"]

        missing_files = []

        for file_name in required_files:
            file_path = (self.model_dir / file_name)
            if not file_path.exists():
                missing_files.append(file_name)

        if len(missing_files) > 0:
            raise RuntimeError(f"Missing model files: {missing_files}")

        print("[INFO] Model verification successful")

    def run(self):

        self.download_file()

        if not self.model_dir.exists():
            self.extract_zip()

        self.verify_model()

        self.cleanup()

        print("\n=================================")
        print("[INFO] BUFFALO_L DOWNLOAD FINISHED")
        print("=================================")
        print(f"[MODEL PATH] {self.model_dir}")

if __name__ == "__main__":

    ModelDownloader().run()