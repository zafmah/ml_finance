import numpy as np


def compute_similarity(f1,f2):
    dx_f1 = f1[1:]-f1[:-1]
    d1 = np.sqrt(np.sum(dx_f1*dx_f1))
    dx_f2 = f2[1:]-f2[:-1]
    d2 = np.sqrt(np.sum(dx_f2*dx_f2))
    if d1 < 0.00001 or d2 < 0.00001:
        if d1 > 0.0001 or d1 > 0.0001:
            # no similarity: Constant and non-constant
            return 0.0
        else:
            # very similar since both seem to be constants
            return 0.98
    return np.minimum(np.dot(dx_f1, dx_f2)/(d1*d2), 1.0)

def compute_similarity_dist(f1,f2):
    dx_f1 = f1[1:]-f1[:-1]
    d1 = np.sqrt(np.sum(dx_f1*dx_f1))
    dx_f2 = f2[1:]-f2[:-1]
    d2 = np.sqrt(np.sum(dx_f2*dx_f2))
    if d1 < 0.00001 or d2 < 0.00001:
        if d1 > 0.0001 or d1 > 0.0001:
            # no similarity: Constant and non-constant
            return 0.0
        else:
            # very similar since both seem to be constants
            return 0.98
    return 1.0-compute_similarity(f1,f2)

def compute_l2_distance(f1,f2):
    return np.linalg.norm(f1-f2, ord=2)/float(f1.shape[0])

def compute_h1_semi_distance(f1,f2):
    return np.linalg.norm(np.diff(f1-f2, n=1), ord=2)/float(f1.shape[0])

def compute_h2_semi_distance(f1,f2):
    return np.linalg.norm(np.diff(f1-f2, n=2), ord=2)/float(f1.shape[0])

def compute_peer_distance(f1,f2):
    return (1.5-compute_similarity(f1,f2))*compute_h1_semi_distance(f1,f2)

measures = {
'similarity': compute_similarity,
'similarity_dist': compute_similarity_dist,
'l2' : compute_l2_distance,
'h1': compute_h1_semi_distance,
'h2': compute_h2_semi_distance,
'peer':  compute_peer_distance
}

def compute(g1,g2,dist, matrix = False):
    if len(g1.shape) > len(g2.shape):
        f1 = g2
        f2 = g1
    else:
        f1 = g1
        f2 = g2
    if isinstance(dist, str): # from callable to string
        dist = measures[dist]
    if matrix:
        if len(f1.shape) != len(f2.shape):
            raise Exception('g1 and g2 must have same shape for matrix computation')
        result = np.empty((f1.shape[0], f1.shape[0],))
        for i in range(f1.shape[0]):
            for j in range(i, f1.shape[0]):
                result[i,j] = dist(f1[i,:], f2[j,:])
                result[j,i] = result[i,j]
    else:
        result = np.empty((f2.shape[0],))
        if len(f1.shape)>1:
            for i in range(f2.shape[0]):
                result[i] = dist(f1[i,:], f2[i,:])
        else:
            for i in range(f2.shape[0]):
                result[i] = dist(f1[:], f2[i,:])
    return result