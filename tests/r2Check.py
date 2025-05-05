import numpy as np
import logging
from QIC.qic import QIC
from QIC.centroidFrame import get_centroid_X1_Y1, get_centroid_X2_Y2
from QIC.util import cylindrical_to_centroid
from QIC.init_axis import init_axis
from qsc.qsc import Qsc


qsc_stel = Qsc.from_paper("r1 section 5.1")
rc = qsc_stel.rc
zs = qsc_stel.zs
B0 = np.full(qsc_stel.nphi, qsc_stel.etabar)
X1c = qsc_stel.X1c
Y1c = qsc_stel.Y1c

qic_stel = QIC(rc, zs, X1c=X1c, Y1c=Y1c, B0=B0, order="r1", frame="FS")

print(qic_stel.X1s)
print(qsc_stel.X1s)