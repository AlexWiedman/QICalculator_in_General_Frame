import numpy as np

class StellConfig():
    def __init__(self, rc = None, zs = None, x1c_fourier = None, y1c_fourier = None, angle_x = None, epsilon = None):
        self.rc = rc
        self.zs = zs
        self.x1c_fourier = x1c_fourier
        self.y1c_fourier = y1c_fourier
        self.angle_x = angle_x
        self.epsilon = epsilon
        self.in_vals = np.array([rc, zs, x1c_fourier, y1c_fourier, angle_x, epsilon])
    
    def get_dofs(self):
        dofs = np.empty()
        for n in self.in_vals:
            if n != None:
                dofs = np.append(dofs, n, axis = 0)
        return dofs
    
    def set_dofs(self, rc = None, zs = None, x1c_fourier = None, y1c_fourier = None, angle_x = None, epsilon = None):
        self.rc = rc
        self.zs = zs
        self.x1c_fourier = x1c_fourier
        self.y1c_fourier = y1c_fourier
        self.angle_x = angle_x
        self.epsilon = epsilon
        self.in_vals = np.array([rc, zs, x1c_fourier, y1c_fourier, angle_x, epsilon])

    def x1c_y1c_fourier_to_array(self, x1c_fourier, y1c_fourier, nphi, nfp):
        assert len(x1c_fourier) == len(y1c_fourier), "x1c_fourier and y1c_fourier have different lengths"
        
        phi = np.linspace(0, 2*np.pi/nfp, nphi)
        nfourier = len(x1c_fourier)
        X1c = np.zeros(nphi)
        Y1c = np.zeros(nphi)
        for jn in range(0, nfourier):
            n = jn * nfp
            sinangle = np.sin(n * phi)
            cosangle = np.cos(n * phi)
            X1c += x1c_fourier[jn] * cosangle
            Y1c += y1c_fourier[jn] * sinangle
        
        return X1c, Y1c