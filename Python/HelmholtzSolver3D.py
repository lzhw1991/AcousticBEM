from HelmholtzSolver import *
from Geometry import *

bOptimized = True
if bOptimized:
    from HelmholtzIntegrals3D_C import *
else:
    from HelmholtzIntegrals3D import *        


    
class HelmholtzSolver3D(HelmholtzSolver):
    def __init__(self, *args, **kwargs):
        super(HelmholtzSolver3D, self).__init__(*args, **kwargs)
        self.aCenters = 1.0/3.0 * (self.aVertex[self.aElement[:, 0]] +\
                                   self.aVertex[self.aElement[:, 1]] +\
                                   self.aVertex[self.aElement[:, 2]])
        # area of the boundary alements
        self.aArea = np.empty(self.aElement.shape[0], dtype=np.float32)
        for i in range(self.aArea.size):
            a = self.aVertex[self.aElement[i, 0], :]
            b = self.aVertex[self.aElement[i, 1], :]
            c = self.aVertex[self.aElement[i, 2], :]
            self.aArea[i] = 0.5 * np.linalg.norm(np.cross(b-a, c-a))
                                  
    def computeBoundaryMatrices(self, k, mu, orientation):
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

                A[i, j] = elementL + mu * elementMt
                B[i, j] = elementM + mu * elementN

            if orientation == 'interior':
                # interior variant, signs are reversed for exterior
                A[i,i] -= 0.5 * mu
                B[i,i] += 0.5
            elif orientation == 'exterior':
                A[i,i] += 0.5 * mu
                B[i,i] -= 0.5
            else:
                assert False, 'Invalid orientation: {}'.format(orientation)

        return A, B

    def computeBoundaryMatricesInterior(self, k, mu):
        return self.computeBoundaryMatrices(k, mu, 'interior')
    
    def computeBoundaryMatricesExterior(self, k, mu):
        return self.computeBoundaryMatrices(k, mu, 'exterior')
    
    def solveSamples(self, solution, aIncidentPhi, aSamples, orientation):
        assert aIncidentPhi.shape == aSamples.shape[:-1], \
            "Incident phi vector and sample points vector must match"

        aResult = np.empty(aSamples.shape[0], dtype=complex)

        for i in range(aIncidentPhi.size):
            p  = aSamples[i]
            sum = aIncidentPhi[i]
            for j in range(solution.aPhi.size):
                qa = self.aVertex[self.aElement[j, 0]]
                qb = self.aVertex[self.aElement[j, 1]]
                qc = self.aVertex[self.aElement[j, 2]]

                elementL  = ComputeL(solution.k, p, qa, qb, qc, False)
                elementM  = ComputeM(solution.k, p, qa, qb, qc, False)
                if orientation == 'interior':
                    sum += elementL * solution.aV[j] - elementM * solution.aPhi[j]
                elif orientation == 'exterior':
                    sum -= elementL * solution.aV[j] - elementM * solution.aPhi[j]
                else:
                    assert False, 'Invalid orientation: {}'.format(orientation)
            aResult[i] = sum
        return aResult

    def solveInterior(self, solution, aIncidentInteriorPhi, aInteriorPoints):
        return self.solveSamples(solution, aIncidentInteriorPhi, aInteriorPoints, 'interior')
    
    def solveExterior(self, solution, aIncidentExteriorPhi, aExteriorPoints):
        return self.solveSamples(solution, aIncidentExteriorPhi, aExteriorPoints, 'exterior')
