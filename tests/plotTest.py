# Just a quick QIC test

from QIC.qic import QIC
import numpy as np
import logging
from QIC.plot import plot, B_fieldline, plot_boundary, plot_axis
from qsc.qsc import Qsc

#logging.basicConfig(level=logging.DEBUG)

rc = [1, -0.3]
zs = [0, -0.25]

# Change X1c Y1c inputs from the FS frame X1c Y1c to Centroid X1c Y1c (some rotation)
#test = QIC(rc, zs)

test = Qsc.from_paper("r1 section 5.1", nphi=63)
nphi = test.nphi

#print(test.k1)
#print(test.k2)
#print(test.k3)


#plot(test) #works fine (could use some tuning in some places though)
#B_fieldline(test, nphi=nphi) #This is a little more jagged then it probably should be, not sure why
plot_boundary(test, nphi=nphi, nsections=8) #Definitely some problem with the plotting here, but this is a pretty complex function and I haven't found the issue yet
#plot_axis(test, nphi=nphi) #works fine
