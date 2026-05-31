from insightface.app import FaceAnalysis


class SCRFDDetector:

    def __init__(self, det_size=(640, 640)):

        self.app = FaceAnalysis(
            name="buffalo_l",
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        )

        self.app.prepare(ctx_id=0, det_size=det_size)

    def detect(self, frame):

        faces = self.app.get(frame)

        results = []

        for face in faces:

            bbox = face.bbox.astype(int)

            x1, y1, x2, y2 = bbox

            results.append({
                "bbox": (x1, y1, x2, y2),
                "kps": face.kps
            })

        return results