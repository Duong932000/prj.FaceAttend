#########################################################
#             .',;::::;,'.                 
#          .';:cccccccccccc:;,.              
#       .;cccccccccccccccccccccc;           --------------
#     .:cccccccccccccccccccccccccc:.        Project name :      prj.FaceAttend
#   .;ccccccccccccc;.:dddl:.;ccccccc;.      Author       :      Nguyen Dac Duong
#  .:ccccccccccccc;OWMKOOXMWd;ccccccc:.     File name    :      build_faiss_indexes.py
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


import faiss
import numpy as np
from pathlib import Path
from core.face_recognition.utils.config import load_common_config

class FaissIndexesBuilder:

    def __init__(self):

        # load config
        root_dir, config = load_common_config()
        self.cfg = config["face_recognition"]

        # paths
        self.embedding_dir = Path(root_dir / self.cfg["paths"]["embeddings"])
        self.index_dir = Path(root_dir / self.cfg["paths"]["indexes"])
        self.index_dir.mkdir(parents=True, exist_ok=True)

        self.embedding_path = (self.embedding_dir / "face_embeddings.npy")
        self.index_path = (self.index_dir / "face_recognition.index")


        self.embeddings = None
        self.index = None
        self.embedding_dim = None
        self.num_embeddings = None

    def load_embeddings(self):

        print(f"[INFO] EMBEDDING_DIR: {self.embedding_dir}")

        print(f"[INFO] INDEX_DIR: {self.index_dir}")

        if not self.embedding_path.exists():
            raise FileNotFoundError(f"Embedding file not found: {self.embedding_path}")

        self.embeddings = np.load(self.embedding_path)

        print(f"[INFO] Loaded embeddings shape: {self.embeddings.shape}")

    def validate_embeddings(self):

        if len(self.embeddings.shape) != 2:
            raise RuntimeError("Embeddings must be 2D")

        (self.num_embeddings, self.embedding_dim) = self.embeddings.shape
        if self.num_embeddings == 0:
            raise RuntimeError("No embeddings found")

        print(f"[INFO] Num embeddings : {self.num_embeddings}")

        print(f"[INFO] Embedding dim  : {self.embedding_dim}")

    def normalize_embeddings(self):

        self.embeddings = self.embeddings.astype(np.float32)

        faiss.normalize_L2(self.embeddings)

        print("[INFO] Embeddings normalized")

    def build_index(self):

        print("[INFO] Building FAISS index...")

        self.index = faiss.IndexFlatIP(self.embedding_dim)

        self.index.add(self.embeddings)

        print(f"[INFO] Total vectors in index: {self.index.ntotal}")

    def save_index(self):

        faiss.write_index(self.index, str(self.index_path))

        print(f"[SAVED] {self.index_path}")

    def verify_index(self):

        loaded_index = faiss.read_index(str(self.index_path))

        print(f"[INFO] Verified index size: {loaded_index.ntotal}")

    def process(self):

        self.load_embeddings()

        self.validate_embeddings()

        self.normalize_embeddings()

        self.build_index()

        self.save_index()

        self.verify_index()

        print("\n========================================")
        print("[INFO] FAISS INDEX BUILD FINISHED")
        print("========================================")

if __name__ == "__main__":

    FaissIndexesBuilder().process()