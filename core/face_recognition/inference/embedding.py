
def extract_embeddings(app,img):
    faces = app.get(img)
    if len(faces) == 0:
        return []
    
    return [f.embedding for f in faces], faces
