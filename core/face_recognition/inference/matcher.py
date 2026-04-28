
import json
import faiss
import numpy

class FaceMatcher:
    def __init__(self, emb_path, meta_path, use_faiss=True):
        self.embeddings = numpy.load(emb_path)
        with open(meta_path, "r") as f:
            self.metadata = json.load(f)

        self.use_faiss = use_faiss

        if use_faiss:
            dim = self.embeddings.shape[1]
            self.index = faiss.IndexFlatIP(dim)
            self.index.add(self.embeddings)

    def match(self, query, threshold=0.5):
        if self.use_faiss:
            D, I = self.index.search(query.reshape(1, -1), 1)
            score = float(D[0][0])
            idx = int(I[0][0])
        else:
            sims = numpy.dot(self.embeddings, query)
            idx = numpy.argmax(sims)
            score = sims[idx]
        
        if score < threshold:
            return "unknow", score
        
        return self.metadata[str(idx)], score
    