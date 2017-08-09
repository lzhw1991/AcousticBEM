bOptimized = True
if bOptimized:
    from HelmholtzIntegrals3D_C import *
else:
    from HelmholtzIntegrals3D import *        
from InteriorHelmholtzSolver import *
from Geometry import *

class InteriorHelmholtzSolver3D(InteriorHelmholtzSolver):
    
    def __init__(self, *args, **kwargs):
        super(InteriorHelmholtzSolver3D, self).__init__(*args, **kwargs)

    def __repr__(self):
        result = "InteriorHelmholtzProblem3D("
        result += "aVertex = "   + repr(self.aVertex) + ", "
        result += "aTriangle = " + repr(self.aTriangle) + ", "
        result += "c = "         + repr(self.c) + ", "
        result += "rho = "       + repr(self.rho) + ")"
        return result

    def computeBoundaryMatrices(self, k, mu):
        A = np.empty((self.aElement.shape[0], self.aElement.shape[0]), dtype=complex)
        B = np.empty(A.shape, dtype=complex)

        for i in range(self.aElement.shape[0]):
            pa = self.aVertex[self.aElement[i, 0]]
            pb = self.aVertex[self.aElement[i, 1]]
            pc = self.aVertex[self.aElement[i, 2]]
            p = (pa + pb + pc) / 3.0
            centerNormal = Normal3D(pa, pb, pc)
            for j in range(self.aElement.shape[0]):
                qa = self.aVertex[self.aElement[j, 0]]
                qb = self.aVertex[self.aElement[j, 1]]
                qc = self.aVertex[self.aElement[j, 2]]

                elementL  = ComputeL(k, p, qa, qb, qc, i==j)
                elementM  = ComputeM(k, p, qa, qb, qc, i==j)
                elementMt = ComputeMt(k, p, centerNormal, qa, qb, qc, i==j)
                elementN  = ComputeN(k, p, centerNormal, qa, qb, qc, i==j)

#                print "N[{}, {}] = {:1.7e}".format(i, j, elementN)
                
                A[i, j] = elementL + mu * elementMt
                B[i, j] = elementM + mu * elementN

            A[i,i] -= 0.5 * mu
            B[i,i] += 0.5

        return A, B

    def solveInterior(self, solution, aIncidentInteriorPhi, aInteriorPoints):
        assert aIncidentInteriorPhi.shape == aInteriorPoints.shape[:-1], \
            "Incident phi vector and interior points vector must match"

        aResult = np.empty(aInteriorPoints.shape[0], dtype=complex)

        for i in range(aIncidentInteriorPhi.size):
            p  = aInteriorPoints[i]
            sum = aIncidentInteriorPhi[i]
            for j in range(solution.aPhi.size):
                qa = self.aVertex[self.aElement[j, 0]]
                qb = self.aVertex[self.aElement[j, 1]]
                qc = self.aVertex[self.aElement[j, 2]]

                elementL  = ComputeL(solution.k, p, qa, qb, qc, False)
                elementM  = ComputeM(solution.k, p, qa, qb, qc, False)

                sum += elementL * solution.aV[j] - elementM * solution.aPhi[j]
            aResult[i] = sum
        return aResult

