# Just a quick QIC test

from QIC.qic import QIC
import numpy as np
import logging
from QIC.plot import plot, B_fieldline, plot_boundary, plot_axis
from qsc.qsc import Qsc
import matplotlib.pyplot as plt
import unittest
from QIC.centroidFrame import get_centroid_X1_Y1

#logging.basicConfig(level=logging.DEBUG)

def r1sec5_1():
    nphi = 63
    stel_qsc = Qsc.from_paper('r1 section 5.1', nphi=nphi)
    X1cfs = stel_qsc.X1c
    Y1cfs = stel_qsc.Y1c
    X1sfs = stel_qsc.X1s
    Y1sfs = stel_qsc.Y1s

    rc=stel_qsc.rc
    zs=stel_qsc.zs

    nfp = stel_qsc.nfp
    stel_qic = QIC(rc, zs, nfp = nfp, nphi=nphi)
    p = stel_qic.frame_p
    q = stel_qic.frame_q
    n = stel_qic.normal
    b = stel_qic.binormal

    X1c, Y1c, X1s, Y1s = get_centroid_X1_Y1(p, q, n, b, [X1cfs, X1sfs], [Y1cfs, Y1sfs])

    stel_qic = QIC(rc, zs, X1c=X1c, Y1c=Y1c, nfp = nfp, nphi = nphi, sigma0=0)
    return stel_qic

class Plotting_Test(unittest.TestCase):
    def test_plot(self):
        test = r1sec5_1()
        plot(test, show=False)
        plt.close()

    def test_B_fieldline(self):
        test = r1sec5_1()
        nphi = test.nphi
        B_fieldline(test, nphi=nphi, show=False)
        plt.close()

    def test_plot_boundary(self):
        test = r1sec5_1()
        nphi = test.nphi
        plot_boundary(test, nphi=nphi, show=False)
        plt.close()
    
    def test_plot_axis(self):
        test = r1sec5_1()
        nphi = test.nphi
        plot_axis(test, nphi=nphi, show=False)
        plt.close()


if __name__ == "__main__":
    unittest.main()