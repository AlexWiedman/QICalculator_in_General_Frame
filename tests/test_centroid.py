import numpy as np
from QIC.centroidFrame import get_FS_frame, get_Centroid_frame, get_kappa1_kappa2, centroid, get_kappa3
import matplotlib.pyplot as plt
import unittest

@unittest.skip("")
class CentroidTest(unittest.TestCase):
    def test_centroid(self):
        exCurve = np.load('./input/curve1_10xPoints.npy')
        h = 1 / len(exCurve[0])
        t, n, b, curvature, torsion = get_FS_frame(exCurve[1], exCurve[2], exCurve[3])

        tangent = []
        normal = []
        binormal = []
        curvatureJ = []
        centr = []

        #finite difference
        dtdpFinite = (np.roll(t, -1, axis=0) - np.roll(t, 1, axis=0)) / (2 * h)

        #analytic
        dldp = np.linalg.norm(exCurve[1], axis=1)
        dtdp = dldp[:, None] * curvature[:, None] * n
        bprime = (np.roll(b, -1, axis=0) - np.roll(b, 1, axis=0))/(2 * h * dldp[:, None])
        torsionAlt=np.empty(torsion.shape)
        for idx, _ in enumerate(n):
            torsionAlt[idx] = -np.dot(bprime[idx],n[idx])

        rtol = 1e-13
        atol = 1e-13
        #np.testing.assert_allclose(t, tangent, rtol, atol)
        #np.testing.assert_allclose(n, normal, rtol, atol)
        #np.testing.assert_allclose(b, binormal, rtol, atol)
        #np.testing.assert_allclose(curvature, curvatureJ, rtol, atol)
        #np.testing.assert_allclose(torsion, torsionAlt, rtol, atol)


        c = centroid(exCurve[0], exCurve[1])

        #np.testing.assert_allclose(c, centr)

        p_vec_J = []
        q_vec_J = []
        dpdphi_J = []
        dqdphi_J = []
        
        
        p, q, dpdphi, dqdphi = get_Centroid_frame(c, exCurve[0], t, exCurve[1], dtdp)

        #np.testing.assert_allclose(p, p_vec_J, rtol, atol)
        #np.testing.assert_allclose(q, q_vec_J, rtol, atol)
        #np.testing.assert_allclose(dpdphi, dpdphi_J, rtol, atol)
        #np.testing.assert_allclose(dqdphi, dqdphi_J, rtol, atol)



        pprime = (np.roll(p, -1, axis=0) - np.roll(p, 1, axis=0))/(2 * h * dldp[:, None])
        qprime = (np.roll(q, -1, axis=0) - np.roll(q, 1, axis=0))/(2 * h * dldp[:, None])

        np.testing.assert_allclose(dqdphi / dldp[:, None], qprime)


        k1, k2 = get_kappa1_kappa2(p, q, n, curvature)

        np.testing.assert_allclose((k1**2+k2**2), curvature, rtol, atol)

        k3 = get_kappa3(dpdphi, dqdphi, q, p, dldp)


        k1p = (np.roll(k1, -1) - np.roll(k1, 1))/(2 * h * dldp)
        k2p = (np.roll(k2, -1) - np.roll(k2, 1))/(2 * h * dldp)
        k3alt = np.empty(k1.shape)
        for idx in range(len(k1)):
            k3alt[idx] = torsion[idx] - (k1[idx] * k2p[idx] - k1p[idx] * k2[idx]) / (curvature[idx]*curvature[idx])



        def get_kappa1_kappa2_alt(dpdphi, dqdphi, t, lp):
        
            k1 = np.empty(t[:,0].shape)
            k2 = np.empty(t[:,0].shape)

            for idx, _ in enumerate(t):
                k1[idx] = -np.dot(dpdphi[idx], t[idx]) / lp[idx]
                k2[idx] = -np.dot(dqdphi[idx], t[idx]) / lp[idx]

            return k1, k2

        k1analytic, k2analytic = get_kappa1_kappa2_alt(dpdphi, dqdphi, t, dldp)

        np.testing.assert_allclose(k1, k1analytic, rtol, atol)
        np.testing.assert_allclose(k3, k3alt)


if __name__ == "__main__":
    unittest.main()