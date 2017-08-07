import numpy as np

def readVertexArray(fileName):
    file = open(fileName + '.1.node', 'r')
    strs = file.readline().split()
    nVertices = int(strs[0])
    nDimensions = int(strs[1])
    nAttributes = int(strs[2])
    bBoundaryMarkers = int(strs[3]) == 1
    aVertices = np.empty([nVertices, nDimensions], dtype=np.float32)
    aAttributes = np.empty([nVertices, nAttributes], dtype=np.float32)
    aBoundaryMarkers = np.empty([nVertices], dtype=np.int32)
    nodeIndex = 1
    for l in file:
        strs = l.split()
        if (strs[0] != '#'):
            assert nodeIndex ==  int(strs[0]), \
                'nodeIndex ({}) doesn\'t match file ({})'.format(nodeIndex, int(strs[0]))
            for dim in range(nDimensions):
                aVertices[nodeIndex-1, dim] = float(strs[1 + dim])
            for attr in range(nAttributes):
                aAttributes[nodeIndex-1, dim] = float(strs[1 + nDimensions + attr])
            if bBoundaryMarkers:
                aBoundaryMarkers[nodeIndex-1] = int(strs[1 + nDimensions + nAttributes])
            nodeIndex += 1
    file.close()
    return aVertices, aAttributes, aBoundaryMarkers

def readTriangleArray(fileName):
    file = open(fileName + '.1.face', 'r')
    strs = file.readline().split()
    nTriangles = int(strs[0])
    bBoundaryMarkers = int(strs[1]) == 1
    aTriangles = np.empty([nTriangles, 3], dtype=np.int32)
    aBoundaryMarkers = np.zeros([nTriangles], dtype=np.int32)
    triangleIndex = 1
    for l in file:
        strs = l.split()
        if (strs[0] != '#'):
            assert triangleIndex == int(strs[0]), \
                'triangleIndex ({}) doent\'t match file ({})'.format(triangleIndex, int(strs[0]))
            for vert in range(3):
                aTriangles[triangleIndex-1, vert] = int(strs[1 + vert]) - 1
            if bBoundaryMarkers:
                aBoundaryMarkers[triangleIndex-1] = int(strs[4])
                triangleIndex += 1
    file.close()
    return aTriangles, aBoundaryMarkers

def writeSTL(fileName, aVertices, aTriangles, aBoundaryMarker = None, boundaryMarker = 0):
    file = open(fileName + '.stl', 'w')
    file.write('solid ' + fileName + '\n')
    for i in range(aTriangles.shape[0]):
        if aBoundaryMarker[i] == boundaryMarker:
            p1 = aVertices[aTriangles[i][0], :]
            p2 = aVertices[aTriangles[i][1], :]
            p3 = aVertices[aTriangles[i][2], :]
            n = np.cross(p2 - p1, p3 - p1)
            n /= np.linalg.norm(n) # normalize
            file.write('facet normal ' + str(n[0]) + ' ' +  str(n[1]) + ' ' + str(n[2]) + '\n')
            file.write('  outer loop\n')
            file.write('    vertex ' + str(p1[0]) + ' ' + str(p1[1]) + ' ' + str(p1[2]) + '\n')
            file.write('    vertex ' + str(p2[0]) + ' ' + str(p2[1]) + ' ' + str(p2[2]) + '\n')
            file.write('    vertex ' + str(p3[0]) + ' ' + str(p3[1]) + ' ' + str(p3[2]) + '\n')
            file.write('  endloop\n')
            file.write('endfacet\n')
    file.close()

def flipOrientation(aTriangles):
    for i in range(aTriangles.shape[0]):
        aTriangles[i, 1], aTriangles[i, 2] = aTriangles[i, 2], aTriangles[i, 1]
        
