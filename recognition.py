
import cv2
import numpy
from insightface.app import FaceAnalysis

# init InsightFace
app = FaceAnalysis(name="buffalo_l")
app.prepare(ctx_id=0)

# load known embeddings
known_emdeddings = []
known_names = []

def cosine_simimarity(a, b):

    return numpy.dot(a, b) / (numpy.linalg.norm(a) * numpy.linalg.norm(b))

def recognize(face_embedding, threshold=0.5):

    if len(known_emdeddings) == 0:
        return "Unknown"
    
    sims = [cosine_simimarity(face_embedding, emb) for emb in known_emdeddings]
    best_idx = numpy.argmax(sims)

    if sims[best_idx] > threshold:
        return known_names[best_idx]
    
    return "Unknown"

if __name__ == "__main__":

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        faces = app.get(frame)

        for face in faces:
            bbox = face.bbox.astype(int)
            embedding = face.embedding

            name = recognize(embedding)

            # draw bbow
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
            cv2.putText(frame, name, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

        cv2.imshow("Face Recognition", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
