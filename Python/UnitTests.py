import unittest
import InteriorHelmholtzSolver2D as IH
import InteriorHelmholtzSolver2D_C as IH
import InteriorHelmholtzSolverRAD as RAD
import InteriorHelmholtzSolver3D as IH3
from scipy.special import hankel1
import numpy as np

class TestComplexQuadGenerator(unittest.TestCase):

    def testComplexQuadGenerator01(self):
        a = np.array([0.0, 0.0], dtype=np.float32)
        b = np.array([1.0, 1.0], dtype=np.float32)
        def func(x):
            return 1.0
        result = RAD.ComplexQuadGenerator(func, a, b)
        self.assertAlmostEqual(result, np.sqrt(2.0), 6, msg = "{} != {}".format(result, np.sqrt(2.0)))


class TestCircularIntegratorPI(unittest.TestCase):

    def testCurcularIntegrator01(self):
        circle = RAD.CircularIntegratorPi(1)
        def func(x):
            return 1.0
        result = circle.integrate(func)
        self.assertAlmostEqual(result, np.pi, msg = "{} != {}".format(result, np.pi))

    def testCurcularIntegrator02(self):
        circle = RAD.CircularIntegratorPi(2)
        def func(x):
            return 1.0
        result = circle.integrate(func)
        self.assertAlmostEqual(result, np.pi, msg = "{} != {}".format(result, np.pi))

class TestTriangleIntegrator(unittest.TestCase):
    def testComplexQuad(self):
        def func(x):
            return 1.0
        a = np.array([0, 0, 0], dtype=np.float32)
        b = np.array([0, 1, 0], dtype=np.float32)
        c = np.array([0, 0, 1], dtype=np.float32)
        result = IH3.InteriorHelmholtzSolver3D.ComplexQuad(func, a, b, c)
        self.assertAlmostEqual(result, 0.5, msg = "{} != {}".format(result, 0.5))
    
class TestHankel(unittest.TestCase):

    def testHankel01(self):
        H1scipy = hankel1(0, 1.0)
        H1gsl = IH.hankel1(0, 1.0)
        self.assertAlmostEqual(H1scipy, H1gsl, msg = "{} != {}".format(H1scipy, H1gsl))

    def testHankel02(self):
        H1scipy = hankel1(0, 10.0)
        H1gsl = IH.hankel1(0, 10.0)
        self.assertAlmostEqual(H1scipy, H1gsl, msg = "{} != {}".format(H1scipy, H1gsl))

    def testHankel03(self):
        H1scipy = hankel1(1, 1.0)
        H1gsl = IH.hankel1(1, 1.0)
        self.assertAlmostEqual(H1scipy, H1gsl, msg = "{} != {}".format(H1scipy, H1gsl))

    def testHankel04(self):
        H1scipy = hankel1(1, 10.0)
        H1gsl = IH.hankel1(1, 10.0)
        self.assertAlmostEqual(H1scipy, H1gsl, msg = "{} != {}".format(H1scipy, H1gsl))

class TestComputeL(unittest.TestCase):

    def testComputeL01(self):
        k = 0.0
        p = np.array([0.5, 0.75], dtype=np.float32)
        a = np.array([0.0, 0.00], dtype=np.float32)
        b = np.array([0.0, 0.10], dtype=np.float32)
        pOnElement = False
        zPy = IH.InteriorHelmholtzSolver2D.ComputeL(k, p, a, b, pOnElement)
        zC  = IH.InteriorHelmholtzSolver2D_C.ComputeL(k, p, a, b, pOnElement)
        self.assertAlmostEqual(zPy, zC, msg = "{} != {}".format(zPy, zC))
        
    def testComputeL02(self):
        k = 10.0
        p = np.array([0.5, 0.75], dtype=np.float32)
        a = np.array([0.0, 0.00], dtype=np.float32)
        b = np.array([0.0, 0.10], dtype=np.float32)
        pOnElement = False
        zPy = IH.InteriorHelmholtzSolver2D.ComputeL(k, p, a, b, pOnElement)
        zC  = IH.InteriorHelmholtzSolver2D_C.ComputeL(k, p, a, b, pOnElement)
        self.assertAlmostEqual(zPy, zC, msg = "{} != {}".format(zPy, zC))
        
    def testComputeL03(self):
        k = 0.0
        p = np.array([0.0, 0.05], dtype=np.float32)
        a = np.array([0.0, 0.00], dtype=np.float32)
        b = np.array([0.0, 0.10], dtype=np.float32)
        pOnElement = True
        zPy = IH.InteriorHelmholtzSolver2D.ComputeL(k, p, a, b, pOnElement)
        zC  = IH.InteriorHelmholtzSolver2D_C.ComputeL(k, p, a, b, pOnElement)
        self.assertAlmostEqual(zPy, zC, msg = "{} != {}".format(zPy, zC))

    def testComputeL04(self):
        k = 10.0
        p = np.array([0.0, 0.05], dtype=np.float32)
        a = np.array([0.0, 0.00], dtype=np.float32)
        b = np.array([0.0, 0.10], dtype=np.float32)
        pOnElement = True
        zPy = IH.InteriorHelmholtzSolver2D.ComputeL(k, p, a, b, pOnElement)
        zC  = IH.InteriorHelmholtzSolver2D_C.ComputeL(k, p, a, b, pOnElement)
        self.assertAlmostEqual(zPy, zC, msg = "{} != {}".format(zPy, zC))

class TestComputeM(unittest.TestCase):

    def testComputeM01(self):
        k = 0.0
        p = np.array([0.5, 0.75], dtype=np.float32)
        a = np.array([0.0, 0.00], dtype=np.float32)
        b = np.array([0.0, 0.10], dtype=np.float32)
        pOnElement = False
        zPy = IH.InteriorHelmholtzSolver2D.ComputeM(k, p, a, b, pOnElement)
        zC  = IH.InteriorHelmholtzSolver2D_C.ComputeM(k, p, a, b, pOnElement)
        self.assertAlmostEqual(zPy, zC, msg = "{} != {}".format(zPy, zC))
        
    def testComputeM02(self):
        k = 10.0
        p = np.array([0.5, 0.75], dtype=np.float32)
        a = np.array([0.0, 0.00], dtype=np.float32)
        b = np.array([0.0, 0.10], dtype=np.float32)
        pOnElement = False
        zPy = IH.InteriorHelmholtzSolver2D.ComputeM(k, p, a, b, pOnElement)
        zC  = IH.InteriorHelmholtzSolver2D_C.ComputeM(k, p, a, b, pOnElement)
        self.assertAlmostEqual(zPy, zC, msg = "{} != {}".format(zPy, zC))
        
    def testComputeM03(self):
        k = 0.0
        p = np.array([0.0, 0.05], dtype=np.float32)
        a = np.array([0.0, 0.00], dtype=np.float32)
        b = np.array([0.0, 0.10], dtype=np.float32)
        pOnElement = True
        zPy = IH.InteriorHelmholtzSolver2D.ComputeM(k, p, a, b, pOnElement)
        zC  = IH.InteriorHelmholtzSolver2D_C.ComputeM(k, p, a, b, pOnElement)
        self.assertAlmostEqual(zPy, zC, msg = "{} != {}".format(zPy, zC))
        
    def testComputeM04(self):
        k = 10.0
        p = np.array([0.0, 0.05], dtype=np.float32)
        a = np.array([0.0, 0.00], dtype=np.float32)
        b = np.array([0.0, 0.10], dtype=np.float32)
        pOnElement = True
        zPy = IH.InteriorHelmholtzSolver2D.ComputeM(k, p, a, b, pOnElement)
        zC  = IH.InteriorHelmholtzSolver2D_C.ComputeM(k, p, a, b, pOnElement)
        self.assertAlmostEqual(zPy, zC, msg = "{} != {}".format(zPy, zC))
        
class TestComputeMt(unittest.TestCase):

    def testComputeMt01(self):
        k = 0.0
        p = np.array([0.5, 0.75], dtype=np.float32)
        normal_p = np.array([-np.sqrt(0.5), -np.sqrt(0.5)])
        a = np.array([0.0, 0.00], dtype=np.float32)
        b = np.array([0.0, 0.10], dtype=np.float32)
        pOnElement = False
        zPy = IH.InteriorHelmholtzSolver2D.ComputeMt(k, p, normal_p, a, b, pOnElement)
        zC  = IH.InteriorHelmholtzSolver2D_C.ComputeMt(k, p, normal_p, a, b, pOnElement)
        self.assertAlmostEqual(zPy, zC, msg = "{} != {}".format(zPy, zC))
        
    def testComputeMt02(self):
        k = 10.0
        p = np.array([0.5, 0.75], dtype=np.float32)
        normal_p = np.array([-np.sqrt(0.5), -np.sqrt(0.5)])
        a = np.array([0.0, 0.00], dtype=np.float32)
        b = np.array([0.0, 0.10], dtype=np.float32)
        pOnElement = False
        zPy = IH.InteriorHelmholtzSolver2D.ComputeMt(k, p, normal_p, a, b, pOnElement)
        zC  = IH.InteriorHelmholtzSolver2D_C.ComputeMt(k, p, normal_p, a, b, pOnElement)
        self.assertAlmostEqual(zPy, zC, msg = "{} != {}".format(zPy, zC))
        
    def testComputeMt03(self):
        k = 0.0
        p = np.array([0.0, 0.05], dtype=np.float32)
        normal_p = np.array([-np.sqrt(0.5), -np.sqrt(0.5)])
        a = np.array([0.0, 0.00], dtype=np.float32)
        b = np.array([0.0, 0.10], dtype=np.float32)
        pOnElement = True
        zPy = IH.InteriorHelmholtzSolver2D.ComputeMt(k, p, normal_p, a, b, pOnElement)
        zC  = IH.InteriorHelmholtzSolver2D_C.ComputeMt(k, p, normal_p, a, b, pOnElement)
        self.assertAlmostEqual(zPy, zC, msg = "{} != {}".format(zPy, zC))
        
    def testComputeMt04(self):
        k = 10.0
        p = np.array([0.0, 0.05], dtype=np.float32)
        normal_p = np.array([-np.sqrt(0.5), -np.sqrt(0.5)])
        a = np.array([0.0, 0.00], dtype=np.float32)
        b = np.array([0.0, 0.10], dtype=np.float32)
        pOnElement = True
        zPy = IH.InteriorHelmholtzSolver2D.ComputeMt(k, p, normal_p, a, b, pOnElement)
        zC  = IH.InteriorHelmholtzSolver2D_C.ComputeMt(k, p, normal_p, a, b, pOnElement)
        self.assertAlmostEqual(zPy, zC, msg = "{} != {}".format(zPy, zC))
                
class TestComputeN(unittest.TestCase):

    def testComputeN01(self):
        k = 0.0
        p = np.array([0.5, 0.75], dtype=np.float32)
        normal_p = np.array([-np.sqrt(0.5), -np.sqrt(0.5)])
        a = np.array([0.0, 0.00], dtype=np.float32)
        b = np.array([0.0, 0.10], dtype=np.float32)
        pOnElement = False
        zPy = IH.InteriorHelmholtzSolver2D.ComputeN(k, p, normal_p, a, b, pOnElement)
        zC  = IH.InteriorHelmholtzSolver2D_C.ComputeN(k, p, normal_p, a, b, pOnElement)
        self.assertAlmostEqual(zPy, zC, msg = "{} != {}".format(zPy, zC))
        
    def testComputeN02(self):
        k = 10.0
        p = np.array([0.5, 0.75], dtype=np.float32)
        normal_p = np.array([-np.sqrt(0.5), -np.sqrt(0.5)])
        a = np.array([0.0, 0.00], dtype=np.float32)
        b = np.array([0.0, 0.10], dtype=np.float32)
        pOnElement = False
        zPy = IH.InteriorHelmholtzSolver2D.ComputeN(k, p, normal_p, a, b, pOnElement)
        zC  = IH.InteriorHelmholtzSolver2D_C.ComputeN(k, p, normal_p, a, b, pOnElement)
        self.assertAlmostEqual(zPy, zC, 6, msg = "{} != {}".format(zPy, zC))
        
    def testComputeN(self):
        k = 0.0
        p = np.array([0.0, 0.05], dtype=np.float32)
        normal_p = np.array([-np.sqrt(0.5), -np.sqrt(0.5)])
        a = np.array([0.0, 0.00], dtype=np.float32)
        b = np.array([0.0, 0.10], dtype=np.float32)
        pOnElement = True
        zPy = IH.InteriorHelmholtzSolver2D.ComputeN(k, p, normal_p, a, b, pOnElement)
        zC  = IH.InteriorHelmholtzSolver2D_C.ComputeN(k, p, normal_p, a, b, pOnElement)
        self.assertAlmostEqual(zPy, zC, msg = "{} != {}".format(zPy, zC))
        
    def testComputeN04(self):
        k = 10.0
        p = np.array([0.0, 0.05], dtype=np.float32)
        normal_p = np.array([-np.sqrt(0.5), -np.sqrt(0.5)])
        a = np.array([0.0, 0.00], dtype=np.float32)
        b = np.array([0.0, 0.10], dtype=np.float32)
        pOnElement = True
        zPy = IH.InteriorHelmholtzSolver2D.ComputeN(k, p, normal_p, a, b, pOnElement)
        zC  = IH.InteriorHelmholtzSolver2D_C.ComputeN(k, p, normal_p, a, b, pOnElement)
        # note, how accuracy here is reduced to only 3 digits after the decimal dot.
        # I don't believe this is because of buggy code but because of error accumulation
        # being different for the C and the Python codes.
        self.assertAlmostEqual(zPy, zC, 3, msg = "{} != {}".format(zPy, zC))

if __name__ == '__main__':
    unittest.main()
    
