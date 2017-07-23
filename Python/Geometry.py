import numpy as np
from numpy.linalg import norm


def Normal2D(pointA, pointB):                  
    diff = pointA - pointB                          
    len = norm(diff)                                
    return np.array([diff[1]/len, -diff[0]/len])    

def Normal3D(a, b, c):
    ab = b - a
    ac = c - a
    normal = np.cross(ab, ac)
    normal /= norm(normal)
    return normal
    
