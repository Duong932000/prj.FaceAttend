#########################################################
#             .',;::::;,'.                 
#          .';:cccccccccccc:;,.              
#       .;cccccccccccccccccccccc;           --------------
#     .:cccccccccccccccccccccccccc:.        Project name :      prj.FaceAttend
#   .;ccccccccccccc;.:dddl:.;ccccccc;.      Author       :      Nguyen Dac Duong
#  .:ccccccccccccc;OWMKOOXMWd;ccccccc:.     File name    :      align_face.py
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


import cv2
from pathlib import Path
import onnxruntime as ort
from insightface.app import FaceAnalysis
from insightface.utils import face_align
from core.face_recognition.utils.config import load_config


class FaceAlignment:
    def __init__(self):
        
        # check onnxruntime available or not
        print(f"[INFO] Check onnxruntime: {ort.get_available_providers()}")

        root_dir, config = load_config()
        self.cfg = config["face_recognition"]

        # path
        self.raw_dir = Path(root_dir / self.cfg["paths"]["raw_dataset"])
        self.aligned_dir = Path(root_dir / self.cfg["paths"]["aligned_faces"])
        self.aligned_dir.mkdir(parents=True, exist_ok=True)

        # model config
        self.model_name = self.cfg["model"]["name"]
        self.det_size = tuple(self.cfg["model"]["det_size"])
        self.ctx_id = self.cfg["model"]["ctx_id"]

        self.image_size = self.cfg["alignment"]["image_size"]

        # load model
        self.loading_model = self.load_model()

    def load_model(self):

        print("[INFO] Loading InsightFace...")

        app = FaceAnalysis(name=self.model_name, allowed_modules=["detection"])
        app.prepare(ctx_id=self.ctx_id, det_size=self.det_size)

        print("[INFO] InsightFace loaded")

        return app

    def get_largest_face(self, faces):

        return max(faces, key=lambda f: ((f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])))

    def align_face(self, image, face):

        aligned_face = face_align.norm_crop(image, landmark=face.kps, image_size=self.image_size)

        return aligned_face

    def process_image(self, image_path, output_path):

        image = cv2.imread(str(image_path))

        if image is None:
            print(f"[WARNING] Cannot read image: {image_path}")

            return False

        faces = self.loading_model.get(image)

        if len(faces) == 0:
            print(f"[WARNING] No face detected: {image_path.name}")
            return False

        face = self.get_largest_face(faces)

        aligned_face = self.align_face(image, face)

        if aligned_face is None:
            print(f"[WARNING] Alignment failed: {image_path.name}")
            return False

        if aligned_face.size == 0:
            print(f"[WARNING] Empty aligned face: {image_path.name}")
            return False

        success = cv2.imwrite(str(output_path),aligned_face)
        if not success:
            print(f"[WARNING] Failed saving: {output_path}")
            return False

        print(f"[SAVE] {output_path}")

        return True

    def process_dataset(self):

        total_images = 0
        success_images = 0
        failed_images = 0

        print(f"[INFO] RAW_DIR: {self.raw_dir}")

        print(f"[INFO] ALIGNED_DIR: {self.aligned_dir}")

        for person_dir in self.raw_dir.iterdir():
            if not person_dir.is_dir():
                continue

            person_name = person_dir.name

            print(f"\n[INFO] Processing person: {person_name}")
            for pose_dir in person_dir.iterdir():
                if not pose_dir.is_dir():
                    continue

                pose_name = pose_dir.name
                print(f"[INFO] Pose: {pose_name}")

                output_dir = (self.aligned_dir / person_name / pose_name)
                output_dir.mkdir(parents=True, exist_ok=True)

                for image_path in pose_dir.glob("*.jpg"):
                    total_images += 1
                    output_path = (output_dir / image_path.name)
                    success = self.process_image(image_path=image_path, output_path=output_path)
                    if success:
                        success_images += 1
                    else:
                        failed_images += 1

        print("\n========================================")
        print("[INFO] ALIGNMENT FINISHED")
        print("========================================")

        print(f"Total Images   : {total_images}")
        print(f"Success Images : {success_images}")
        print(f"Failed Images  : {failed_images}")

if __name__ == "__main__":

    FaceAlignment().process_dataset()
