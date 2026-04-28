from models import load_model
from inference.embedding import extract_embeddings
from inference.matcher import FaceMatcher

class FacePipeline:
    def __init__(self, config):
        self.app = load_model(config["model_name"], config["device"])

        db = config["database"]
        self.matcher = FaceMatcher(
            db["embedding_path"],
            db["metadata_path"],
            db["use_faiss"]
        )

        self.threshold = config["similarity_threshold"]

    def run(self, img):
        embeddings, faces = extract_embeddings(self.app, img)

        results = []
        for emb, face in zip(embeddings, faces):
            name, score = self.matcher.match(emb, self.threshold)
            results.append({
                "name": name,
                "score": score,
                "bbox": face.bbox.astype(int)
            })

            return results
