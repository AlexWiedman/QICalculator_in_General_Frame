import numpy as np
from .util import mu0

def r2calc(self):

    #shorthand terms
    nphi = self.nphi
    B0_over_abs_G0 = self.B0 / np.abs(self.G0)
    abs_G0_over_B0 = 1 / B0_over_abs_G0
    X1c = self.X1c
    X1s = self.X1s
    Y1s = self.Y1s
    Y1c = self.Y1c
    sigma = self.sigma
    d_d_varphi = self.d_d_varphi
    iota_N = self.iotaN
    iota = self.iota
    k1 = self.k1
    k2 = self.k2
    k3 = self.k3
    B1s = self.B1s
    B1c = self.B1c
    B0 = self.B0
    G0 = self.G0
    I2 = self.I2
    B2s = self.B2s
    B2c = self.B2c
    p2 = self.p2
    sG = self.sG
    spsi = self.spsi
    I2_over_B0 = self.I2 / self.B0
    Bbar = self.Bbar
    dldp = self.d_l_d_varphi
    #beta0 is omitted, for what reason?
    beta0=0

    V1 = X1c * X1c + X1s * X1s + Y1c * Y1c + Y1s * Y1s
    V2 = 2 * (X1s * X1c + Y1s * Y1c)
    V3 = X1c * X1c - X1s * X1s + Y1c * Y1c - Y1s * Y1s

    factor = - B0_over_abs_G0 / 8;
    Z20 = factor*np.matmul(d_d_varphi,V1)
    Z2s = factor*(np.matmul(d_d_varphi,V2) - 2 * iota_N * V3)
    Z2c = factor*(np.matmul(d_d_varphi,V3) + 2 * iota_N * V2)

    # Big issues start here. Have to work out the linear algebra to solve for X2 and Y2 terms since unlike in the original they both appear in A34-A36.

    # Need to double check the math here
    beta_1s = -4 * spsi * sG * mu0 * p2 * B1c * abs_G0_over_B0 / (iota_N * B0 * B0)
    beta_1c = -4 * spsi * sG * mu0 * p2 * B1s * abs_G0_over_B0 / (iota_N * B0 * B0)

    qs = np.matmul(d_d_varphi,X1s) - iota_N * X1c - Y1s * k3 * abs_G0_over_B0
    qc = np.matmul(d_d_varphi,X1c) + iota_N * X1s - Y1c * k3 * abs_G0_over_B0
    rs = np.matmul(d_d_varphi,Y1s) - iota_N * Y1c + X1s * k3 * abs_G0_over_B0
    rc = np.matmul(d_d_varphi,Y1c) + iota_N * Y1s + X1c * k3 * abs_G0_over_B0

    eqA32_inhomogeneous = -sG * Bbar / (2 * B0) * (X1s*k1 + Y1s*k2)
    eqA33_inhomogeneous = -sG * Bbar / (2 * B0) * (X1c*k1 + Y1c*k2)
    #eqA34_inhomogeneous = (1 / dldp) * (np.matmul(d_d_varphi, Z20) - (1 / dldp) * (abs_G0_over_B0*abs_G0_over_B0*))
    eqA35_inhomogeneous = (1 / dldp) * (np.matmul(d_d_varphi, Z2s) - 2 * iota_N * Z2c \
                                        - (1 / dldp) * (-abs_G0_over_B0*abs_G0_over_B0*B2s/B0) + (3/2)*(abs_G0_over_B0*abs_G0_over_B0*B1c*B1s/(B0*B0) \
                                        - (X1c*X1s*k1*k1 + X1c*Y1s*k1*k2 + X1s*Y1c*k1*k2+Y1c*Y1s*k2*k2)/4 * dldp*dldp - (qc*qs+rc*rs)/2))
    eqA36_inhomogeneous = (1 / dldp) * (np.matmul(d_d_varphi, Z2c) - 2 * iota_N * Z2s \
                                        - (1 / dldp) * (-abs_G0_over_B0*abs_G0_over_B0*B2c/B0) + (3/4)*(abs_G0_over_B0*abs_G0_over_B0*(B1c*B1c-B1s*B1s)/(B0*B0) \
                                        - ((X1c*X1c - X1s * X1s)*k1*k1 + 2*(X1c*Y1c-X1s*Y1c)*k1*k2 + (Y1c*Y1c - Y1s*Y1s)*k2*k2)/4 * dldp*dldp - (qc*qc - qs*qs + rc*rc - rs*rs)/2))

    fx0_inhomogeneous = -(I2/Bbar) * 0.5 * (k1*(X1c*Y1c+X1s*Y1s) + k2*(Y1c*Y1c+Y1s*Y1s)) * dldp - 0.5*beta0*k1*dldp*(X1s*Y1c - X1c*Y1s) - 0.5*dldp*(beta_1c*Y1s - beta_1s*Y1c) + k1*dldp*Z20
    fxs_inhomogeneous = -(I2/Bbar) * 0.5 * (k1*(X1s*Y1c+X1c*Y1s) + 2*k2*(Y1c*Y1s)) * dldp - 0.5*beta0*dldp*(k1*(X1c*Y1c - X1s*Y1s)-k2*(Y1s*Y1s-Y1c*Y1c)) - 0.5*dldp*(beta_1c*Y1s - beta_1s*Y1c) + k1*dldp*Z2s
    fxc_inhomogeneous = -(I2/Bbar) * 0.5 * (k1*(X1c*Y1c-X1s*Y1s) + k2*(Y1c*Y1c-Y1s*Y1s)) * dldp - 0.5*beta0*dldp*(-k1*(X1c*Y1s + X1s*Y1c)+2*k2*(Y1s*Y1c)) - 0.5*dldp*(beta_1c*Y1s - beta_1s*Y1c) + k1*dldp*Z2c
    fy0_inhomogeneous =  (I2/Bbar) * 0.5 * (k1*(X1s*X1s+X1c*X1c) + k2*(X1c*Y1c + X1s*Y1s)) * dldp - 0.5*beta0*dldp*(k2*(X1s*Y1c-X1c*Y1s)) - 0.5*dldp*(beta_1s*X1c - beta_1c*X1s) + k2*dldp*Z20
    fys_inhomogeneous =  (I2/Bbar) * 0.5 * (2*k1*(X1s*X1c) + k2*(X1s*Y1c + X1c*Y1s)) * dldp - 0.5*beta0*dldp*(k1*(X1s*X1s-X1c*X1c) - k2*(X1c*Y1c-X1s*Y1s)) - 0.5*dldp*(beta_1s*X1c - beta_1c*X1s) + k2*dldp*Z2s
    fyc_inhomogeneous =  (I2/Bbar) * 0.5 * (k1*(X1c*X1c - X1s*X1s) + k2*(X1c*Y1c - X1s*Y1s)) * dldp - 0.5*beta0*dldp*(2*k1*(X1s*X1c) + k2*(X1c*Y1s+X1s*Y1c)) - 0.5*dldp*(beta_1s*X1c - beta_1c*X1s) + k2*dldp*Z2c
    
    eqA41_inhomogeneous = -X1s*fx0_inhomogeneous + X1c*fxs_inhomogeneous - X1s*fxc_inhomogeneous - Y1s*fy0_inhomogeneous + Y1c*fys_inhomogeneous - Y1s*fyc_inhomogeneous
    eqA42_inhomogeneous = -X1c*fx0_inhomogeneous + X1s*fxs_inhomogeneous + X1c*fxc_inhomogeneous - Y1c*fy0_inhomogeneous + Y1s*fys_inhomogeneous + Y1c*fyc_inhomogeneous

    eqA32_X20 = -Y1c
    eqA32_X2s = Y1s
    eqA32_X2c = Y1c
    eqA32_Y20 = X1c
    eqA32_Y2s = -X1s
    eqA32_Y2c = -X1c

    eqA33_X20 = Y1s
    eqA33_X2s = -Y1c
    eqA33_X2c = Y1s
    eqA33_Y20 = -X1s
    eqA33_Y2s = X1c
    eqA33_Y2c = -X1s

    eqA35_X20 = 0
    eqA35_X2s = k1
    eqA35_X2c = 0
    eqA35_Y20 = 0
    eqA35_Y2s = k2
    eqA35_Y2c = 0

    eqA36_X20 = 0
    eqA36_X2s = 0
    eqA36_X2c = k1
    eqA36_Y20 = 0
    eqA36_Y2s = 0
    eqA36_Y2c = k2

    fx0_X20 = d_d_varphi
    fx0_X2s = 0
    fx0_X2c = 0
    fx0_Y20 = -k3*dldp + 2*I2*dldp/Bbar
    fx0_Y2s = -4*G0*Z2c/Bbar
    fx0_Y2c =  4*G0*Z2s/Bbar

    fxs_X20 = 0
    fxs_X2s = d_d_varphi
    fxs_X2c = -2*iota_N
    fxs_Y20 = 4*G0*Z2c/Bbar
    fxs_Y2s = -k3*dldp + 2*I2*dldp/Bbar
    fxs_Y2c = -4*G0*Z20/Bbar

    fxc_X20 = 0
    fxc_X2s = 2*iota_N
    fxc_X2c = d_d_varphi
    fxc_Y20 = -4*G0*Z2s/Bbar
    fxc_Y2s = 4*G0*Z20/Bbar
    fxc_Y2c = -k3*dldp + 2*I2*dldp/Bbar
    
    fy0_X20 = k3*dldp - 2*I2*dldp/Bbar
    fy0_X2s = -4*G0*Z2c/Bbar
    fy0_X2c = 4*G0*Z2s/Bbar
    fy0_Y20 = d_d_varphi
    fy0_Y2s = 0
    fy0_Y2c = 0
    
    fys_X20 = -4*G0*Z2c/Bbar
    fys_X2s = k3*dldp - 2*I2*dldp/Bbar
    fys_X2c = 4*G0*Z20/Bbar
    fys_Y20 = 0
    fys_Y2s = d_d_varphi
    fys_Y2c = -2*iota_N

    fyc_X20 = 4*G0*Z2s/Bbar
    fyc_X2s = -4*G0*Z20/Bbar
    fyc_X2c = k3*dldp - 2*I2*dldp/Bbar
    fyc_Y20 = 0
    fyc_Y2s = 2*iota_N
    fyc_Y2c = d_d_varphi

    eqA41_X20 = -X1s*fx0_X20 + X1c*fxs_X20 - X1s*fxc_X20 - Y1s*fy0_X20 + Y1c*fys_X20 - Y1s*fyc_X20
    eqA41_X2s = -X1s*fx0_X2s + X1c*fxs_X2s - X1s*fxc_X2s - Y1s*fy0_X2s + Y1c*fys_X2s - Y1s*fyc_X2s
    eqA41_X2c = -X1s*fx0_X2c + X1c*fxs_X2c - X1s*fxc_X2c - Y1s*fy0_X2c + Y1c*fys_X2c - Y1s*fyc_X2c
    eqA41_Y20 = -X1s*fx0_Y20 + X1c*fxs_Y20 - X1s*fxc_Y20 - Y1s*fy0_Y20 + Y1c*fys_Y20 - Y1s*fyc_Y20
    eqA41_Y2s = -X1s*fx0_Y2s + X1c*fxs_Y2s - X1s*fxc_Y2s - Y1s*fy0_Y2s + Y1c*fys_Y2s - Y1s*fyc_Y2s
    eqA41_Y2c = -X1s*fx0_Y2c + X1c*fxs_Y2c - X1s*fxc_Y2c - Y1s*fy0_Y2c + Y1c*fys_Y2c - Y1s*fyc_Y2c

    eqA42_X20 = -X1c*fx0_X20 + X1s*fxs_X20 + X1c*fxc_X20 - Y1c*fy0_X20 + Y1s*fys_X20 + Y1c*fyc_X20
    eqA42_X2s = -X1c*fx0_X2s + X1s*fxs_X2s + X1c*fxc_X2s - Y1c*fy0_X2s + Y1s*fys_X2s + Y1c*fyc_X2s
    eqA42_X2c = -X1c*fx0_X2c + X1s*fxs_X2c + X1c*fxc_X2c - Y1c*fy0_X2c + Y1s*fys_X2c + Y1c*fyc_X2c
    eqA42_Y20 = -X1c*fx0_Y20 + X1s*fxs_Y20 + X1c*fxc_Y20 - Y1c*fy0_Y20 + Y1s*fys_Y20 + Y1c*fyc_Y20
    eqA42_Y2s = -X1c*fx0_Y2s + X1s*fxs_Y2s + X1c*fxc_Y2s - Y1c*fy0_Y2s + Y1s*fys_Y2s + Y1c*fyc_Y2s
    eqA42_Y2c = -X1c*fx0_Y2c + X1s*fxs_Y2c + X1c*fxc_Y2c - Y1c*fy0_Y2c + Y1s*fys_Y2c + Y1c*fyc_Y2c

    matrix = np.zeros((6 * nphi, 6 * nphi))
    right_hand_side = np.zeros(6 * nphi)
    for j in range(nphi):
        #X20 terms
        matrix[j,               0:nphi      ] = eqA32_X20
        matrix[j+nphi,          0:nphi      ] = eqA33_X20
        matrix[j+2*nphi,        0:nphi      ] = eqA35_X20
        matrix[j+3*nphi,        0:nphi      ] = eqA36_X20 
        matrix[j+4*nphi,        0:nphi      ] = eqA41_X20
        matrix[j+5*nphi,        0:nphi      ] = eqA42_X20
        #X2c terms
        matrix[j,               nphi:2*nphi ] = eqA32_X2c
        matrix[j+nphi,          nphi:2*nphi ] = eqA33_X2c 
        matrix[j+2*nphi,        nphi:2*nphi ] = eqA35_X2c
        matrix[j+3*nphi,        nphi:2*nphi ] = eqA36_X2c 
        matrix[j+4*nphi,        nphi:2*nphi ] = eqA41_X2c
        matrix[j+5*nphi,        nphi:2*nphi ] = eqA42_X2c
        #X2s terms
        matrix[j,               2*nphi:3*nphi] = eqA32_X2s
        matrix[j+nphi,          2*nphi:3*nphi] = eqA33_X2s 
        matrix[j+2*nphi,        2*nphi:3*nphi] = eqA35_X2s
        matrix[j+3*nphi,        2*nphi:3*nphi] = eqA36_X2s 
        matrix[j+4*nphi,        2*nphi:3*nphi] = eqA41_X2s
        matrix[j+5*nphi,        2*nphi:3*nphi] = eqA42_X2s
        #Y20 terms
        matrix[j,               3*nphi:4*nphi] = eqA32_Y20
        matrix[j+nphi,          3*nphi:4*nphi] = eqA33_Y20
        matrix[j+2*nphi,        3*nphi:4*nphi] = eqA35_Y20
        matrix[j+3*nphi,        3*nphi:4*nphi] = eqA36_Y20
        matrix[j+4*nphi,        3*nphi:4*nphi] = eqA41_Y20
        matrix[j+5*nphi,        3*nphi:4*nphi] = eqA42_Y20
        #Y2c terms
        matrix[j,               4*nphi:5*nphi] = eqA32_Y2c
        matrix[j+nphi,          4*nphi:5*nphi] = eqA33_Y2c
        matrix[j+2*nphi,        4*nphi:5*nphi] = eqA35_Y2c
        matrix[j+3*nphi,        4*nphi:5*nphi] = eqA36_Y2c
        matrix[j+4*nphi,        4*nphi:5*nphi] = eqA41_Y2c
        matrix[j+5*nphi,        4*nphi:5*nphi] = eqA42_Y2c
        #Y2s terms
        matrix[j,               5*nphi:6*nphi] = eqA32_Y2s
        matrix[j+nphi,          5*nphi:6*nphi] = eqA33_Y2s
        matrix[j+2*nphi,        5*nphi:6*nphi] = eqA35_Y2s
        matrix[j+3*nphi,        5*nphi:6*nphi] = eqA36_Y2s
        matrix[j+4*nphi,        5*nphi:6*nphi] = eqA41_Y2s
        matrix[j+5*nphi,        5*nphi:6*nphi] = eqA42_Y2s


    #Inhomogeneous
    right_hand_side[0:nphi               ] = -(eqA32_inhomogeneous)
    right_hand_side[nphi:2*nphi          ] = -(eqA33_inhomogeneous)
    right_hand_side[2*nphi:3*nphi        ] = -(eqA35_inhomogeneous)
    right_hand_side[3*nphi:4*nphi        ] = -(eqA36_inhomogeneous)
    right_hand_side[4*nphi:5*nphi        ] = -(eqA41_inhomogeneous)
    right_hand_side[5*nphi:6*nphi        ] = -(eqA42_inhomogeneous)
    
    solution = np.linalg.solve(matrix, right_hand_side)

    X20 = solution[0:nphi]
    X2c = solution[nphi:2*nphi]
    X2s = solution[2*nphi:3*nphi]
    Y20 = solution[3*nphi:4*nphi]
    Y2c = solution[4*nphi:5*nphi]
    Y2s = solution[5*nphi:6*nphi]

    # Check to make sure the equations actually sum to 0

    #B20 = 