"""
Compare output from pyQIC (Name already taken, must rename) to pyQSC.
"""

import unittest
import numpy as np
import logging
from QIC.qic import QIC

logger = logging.getLogger(__name__)

def compare_to_qsc_r1(filename):
    filler = np.load(filename) # placeholder for the time being, until I actually get a file to compare to

    dat = filler.x
    nphi = dat.nphi
    nfp = dat.nfp
    QSx1c = dat.x1c
    QSx1s = dat.x1s
    QSy1c = dat.y1c
    QSy1s = dat.y1s
    QSB0 = np.full(nphi, dat.B0)
    QSB1s = np.full(nphi, dat.B1s)
    QSB1c = np.full(nphi, dat.B1c)
    QSrc = dat.rc
    QSzs = dat.zs
    sigma0 = dat.sigma0
    sG = dat.sG
    spsi = dat.spsi

    def compare_field(qsc_field, qic_field, rtol=1e-9, atol=1e-9):
        logger('max difference: {}'.format(np.max(np.abs(qsc_field - qic_field))))
        np.testing.assert_allclose(qsc_field, qic_field, rtol=rtol, atol=atol)

    iso = QIC(QSrc, QSzs, nfp = nfp, sigma0 = sigma0, X1c = QSx1c, Y1c=QSy1c, B0=QSB0, nphi=nphi, sG=sG, spsi=spsi)
    iso.calculate()

    compare_field(QSx1s, iso.X1s)
    compare_field(QSy1s, iso.y1s)
    compare_field(QSB1s, iso.B1s)
    compare_field(QSB1c, iso.B1c)
    compare_field(dat.iota, iso.iota)
    compare_field(dat.torsion, iso.torsion)
    compare_field(dat.curvature, iso.curvature)
    compare_field(dat.sigma, iso.sigma)


    


