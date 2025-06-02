"""
Compare output from pyQIC (Name already taken, must rename) to pyQSC.
"""

import unittest
import numpy as np
import logging
from QIC.qic import QIC
from QIC.centroidFrame import get_centroid_X1_Y1, get_centroid_X2_Y2
from QIC.util import cylindrical_to_centroid
from QIC.init_axis import init_axis
from qsc.qsc import Qsc


#simpler test? create axisymmetric (circular magnetic axis) shape

nphi = 21

rc = np.zeros(5)
zc = np.zeros(5)
rs = np.zeros(5)
zs = np.zeros(5)
X1c = np.ones(nphi)
Y1c = np.ones(nphi)
B0 = np.ones(nphi)

rc[0] = 1
rc[2] = 0.34
zs[2] = 0.34

test = QIC(rc, zs, X1c=X1c, Y1c=Y1c, nphi=nphi, B0=B0, order='r2', B2c=np.ones(nphi))

test.calculate()



atol=1e-3

class QIC_v_QSC(unittest.TestCase):
    def test_sec_5_1(self):
            nphi = 63
            stel_qsc = Qsc.from_paper('r2 section 5.1', nphi=nphi)
            X1cfs = stel_qsc.X1c_untwisted
            Y1cfs = stel_qsc.Y1c_untwisted
            X1sfs = stel_qsc.X1s_untwisted
            Y1sfs = stel_qsc.Y1s_untwisted
            
            X20fs = stel_qsc.X20_untwisted
            Y20fs = stel_qsc.Y20_untwisted
            X2cfs = stel_qsc.X2c_untwisted
            Y2cfs = stel_qsc.Y2c_untwisted
            X2sfs = stel_qsc.X2s_untwisted
            Y2sfs = stel_qsc.Y2s_untwisted

            B2s = np.full(nphi, stel_qsc.B2s)
            B2c = np.full(nphi, stel_qsc.B2c)
            p2 = stel_qsc.p2
            I2 = stel_qsc.I2
            G2 = stel_qsc.G2


            rc=stel_qsc.rc
            zs=stel_qsc.zs

            nfp = stel_qsc.nfp
            stel_qic = QIC(rc, zs, nfp = nfp, nphi=nphi)
            p = stel_qic.frame_p_cartesian
            q = stel_qic.frame_q_cartesian
            n = stel_qic.normal_cartesian
            b = stel_qic.binormal_cartesian

            X1c, Y1c, X1s, Y1s = get_centroid_X1_Y1(p, q, n, b, [X1cfs, X1sfs], [Y1cfs, Y1sfs])
            X20, Y20, X2c, Y2c, X2s, Y2s = get_centroid_X2_Y2(p, q, n, b, [X20fs, X2cfs, X2sfs], [Y20fs, Y2cfs, Y2sfs])

            stel_qic = QIC(rc, zs, X1c=X1c, Y1c=Y1c, nfp = nfp, nphi = nphi, sigma0=stel_qsc.sigma0, order='r2', B2s=B2s, B2c=B2c, p2=p2, I2=I2)

            B0 = stel_qic.B0
            G0 = stel_qic.G0
            dldp = stel_qic.d_l_d_varphi
            k1 = stel_qic.k1
            k2 = stel_qic.k2
            iota = stel_qic.iota
            d_d_varphi = stel_qic.d_d_varphi
            Z20 = stel_qic.Z20
            B1c = stel_qic.B1c
            B1s = stel_qic.B1s
            iota_N = stel_qic.iotaN
            k3 = stel_qic.k3
            #G2 = stel_qic.G2

            qs_ = np.matmul(d_d_varphi,X1s) - iota_N * X1c - Y1s * k3 * dldp
            qc_ = np.matmul(d_d_varphi,X1c) + iota_N * X1s - Y1c * k3 * dldp
            rs_ = np.matmul(d_d_varphi,Y1s) - iota_N * Y1c + X1s * k3 * dldp
            rc_ = np.matmul(d_d_varphi,Y1c) + iota_N * Y1s + X1c * k3 * dldp
            
            
            np.testing.assert_allclose(Z20, stel_qic.Z20)
            np.testing.assert_allclose(X1s, stel_qic.X1s)
            np.testing.assert_allclose(Y1s, stel_qic.Y1s)
            np.testing.assert_allclose(stel_qsc.iota, stel_qic.iota)
            np.testing.assert_allclose(stel_qsc.G2, stel_qic.G2)
            np.testing.assert_allclose(X20, stel_qic.X20)
            np.testing.assert_allclose(stel_qic.B20, stel_qsc.B20)
            np.testing.assert_allclose(stel_qic.B2s, B2s)
            np.testing.assert_allclose(stel_qic.B2c, B2c)
            
    @unittest.skip("Not stellarator symmetric")
    def test_sec_5_2(self):
            nphi = 63
            stel_qsc = Qsc.from_paper('r2 section 5.2', nphi=nphi)
            X1cfs = stel_qsc.X1c_untwisted
            Y1cfs = stel_qsc.Y1c_untwisted
            X1sfs = stel_qsc.X1s_untwisted
            Y1sfs = stel_qsc.Y1s_untwisted
            B2s = np.full(nphi, stel_qsc.B2s)
            B2c = np.full(nphi, stel_qsc.B2c)
            p2 = stel_qsc.p2
            I2 = stel_qsc.I2

            rc=stel_qsc.rc
            zs=stel_qsc.zs

            nfp = stel_qsc.nfp
            stel_qic = QIC(rc, zs, nfp = nfp, nphi=nphi)
            p = stel_qic.frame_p_cartesian
            q = stel_qic.frame_q_cartesian
            n = stel_qic.normal_cartesian
            b = stel_qic.binormal_cartesian

            X1c, Y1c, X1s, Y1s = get_centroid_X1_Y1(p, q, n, b, [X1cfs, X1sfs], [Y1cfs, Y1sfs])

            stel_qic = QIC(rc, zs, X1c=X1c, Y1c=Y1c, nfp = nfp, nphi = nphi, sigma0=stel_qsc.sigma0, order='r2', B2s=B2s, B2c=B2c, p2=p2, I2=I2)

            np.testing.assert_allclose(X1s, stel_qic.X1s, atol=atol)
            np.testing.assert_allclose(Y1s, stel_qic.Y1s, atol=atol)
            np.testing.assert_allclose(stel_qsc.iota, stel_qic.iota, atol=atol)
            np.testing.assert_allclose(stel_qic.B20, stel_qsc.B20, atol=atol)
    
    @unittest.skip("Not stellarator symmetric")
    def test_sec_5_3(self):
            nphi = 63
            stel_qsc = Qsc.from_paper('r2 section 5.3', nphi=nphi)
            X1cfs = stel_qsc.X1c_untwisted
            Y1cfs = stel_qsc.Y1c_untwisted
            X1sfs = stel_qsc.X1s_untwisted
            Y1sfs = stel_qsc.Y1s_untwisted

            rc=stel_qsc.rc
            zs=stel_qsc.zs

            nfp = stel_qsc.nfp
            stel_qic = QIC(rc, zs, nfp = nfp, nphi=nphi)
            p = stel_qic.frame_p_cartesian
            q = stel_qic.frame_q_cartesian
            n = stel_qic.normal_cartesian
            b = stel_qic.binormal_cartesian

            X1c, Y1c, X1s, Y1s = get_centroid_X1_Y1(p, q, n, b, [X1cfs, X1sfs], [Y1cfs, Y1sfs])

            stel_qic = QIC(rc, zs, X1c=X1c, Y1c=Y1c, nfp = nfp, nphi = nphi, sigma0=stel_qsc.sigma0)

            np.testing.assert_allclose(X1s, stel_qic.X1s)
            np.testing.assert_allclose(Y1s, stel_qic.Y1s)
            np.testing.assert_allclose(stel_qsc.iota, stel_qic.iota)

if __name__ == "__main__":
    unittest.main()