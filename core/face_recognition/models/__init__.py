import numpy

def cosine_similarity(a, b):

    return numpy.dot(a, b) / (numpy.linalg.norm(a) * numpy.linalg.norm(b))
