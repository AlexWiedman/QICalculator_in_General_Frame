# Just a quick QIC test

from QIC.qic import QIC
import numpy as np
import logging
from QIC.plot import plot, B_fieldline, plot_boundary, plot_axis
from qsc.qsc import Qsc
import matplotlib.pyplot as plt
import unittest

#logging.basicConfig(level=logging.DEBUG)

class Plotting_Test(unittest.TestCase):
    def test_plot(self):
        test = Qsc.from_paper("r1 section 5.1", nphi=63)
        plot(test, show=False)
        plt.close()

    def test_B_fieldline(self):
        test = Qsc.from_paper("r1 section 5.1", nphi=63)
        nphi = test.nphi
        B_fieldline(test, nphi=nphi, show=False)
        plt.close()

    def test_plot_boundary(self):
        test = Qsc.from_paper("r1 section 5.1", nphi=63)
        nphi = test.nphi
        plot_boundary(test, nphi=nphi, show=False)
        plt.close()
    
    def test_plot_axis(self):
        test = Qsc.from_paper("r1 section 5.1", nphi=63)
        nphi = test.nphi
        plot_axis(test, nphi=nphi, show=False)
        plt.close()


if __name__ == "__main__":
    unittest.main()