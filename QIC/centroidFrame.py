"""
Definition set for calculating the centroid frame from the Frenet-Serret Frame, from the CoilForces.jl julia package
"""
import numpy as np

def get_r_vector(R0, R0p, R0pp, R0ppp, Z0, Z0p, Z0pp, Z0ppp, nphi, nfp):

    phi = np.linspace(0, 2 * np.pi / nfp, nphi, endpoint=False)
    cosphi = np.cos(phi)
    sinphi = np.sin(phi)

    r = np.array([R0*cosphi, R0*sinphi, Z0]).transpose()
    rp = np.array([R0p*cosphi - R0*sinphi, R0p*sinphi + R0*cosphi, Z0p]).transpose()
    rpp = np.array([(R0pp - R0)*cosphi - 2*R0p*sinphi, (R0pp - R0)*sinphi + 2*R0p*cosphi, Z0pp]).transpose()
    rppp = np.array([(R0ppp - 3*R0p)*cosphi - (3*R0pp - R0)*sinphi, (R0ppp - 3*R0p)*sinphi + (3*R0pp - R0)*cosphi, Z0ppp]).transpose()


    return r, rp, rpp, rppp

def get_FS_frame(drdp, d2rdp2, d3rdp3):
    # takes r' and r'', r''' returns Frenet-Serret Frame
    # r is an n by 3 array, where n is the parametrized phi
    
    dldp = np.linalg.norm(drdp, axis=1)

    t = np.empty(drdp.shape)
    n = np.empty(drdp.shape)
    for idx, l in enumerate (dldp):
        t[idx] = np.divide(drdp[idx], l)
        n[idx] = np.divide(np.cross(drdp[idx,:], np.cross(d2rdp2[idx,:], drdp[idx])),(l * np.linalg.norm(np.cross(d2rdp2[idx], drdp[idx]))))
    b = np.cross(t, n, axis=1)

    curvature = np.divide(np.linalg.norm(np.cross(drdp, d2rdp2, axis=1),axis=1),(dldp*dldp*dldp))
    torsion = np.empty(curvature.shape)
    for idx, _ in enumerate(drdp):
        torsion[idx] = np.dot(drdp[idx], np.cross(d2rdp2[idx], d3rdp3[idx])) / np.linalg.norm(np.cross(drdp[idx], d2rdp2[idx]))**2

    return t, n, b, curvature, torsion



def centroid(r, drdp):
    # takes r and r' returns the centroid of the curve

    n = len(r)

    dldp = np.linalg.norm(drdp, axis=1)

    x = np.zeros(3)
    L = 0

    for i in range(n):
        x += r[i] * dldp[i]
        L += dldp[i]

    return (1/L) * x



def get_Centroid_frame(c, r, t, drdphi, dtdphi):
    # takes the centroid, r, and the tangent vector, returns p and q from the centroid grame
    
    #P0
    p = r - c
    dpdphi = drdphi.copy()
    for idx, tan in enumerate(t):
        #P1
        temp = np.dot(p[idx], tan)
        dtempdphi = np.dot(p[idx], dtdphi[idx]) + np.dot(dpdphi[idx], tan)

        p[idx] -= temp * tan
        dpdphi[idx] = dpdphi[idx] - dtempdphi * tan - temp * dtdphi[idx]

        #Pf
        temp2 = np.linalg.norm(p[idx])
        p[idx] = (1 / temp2) * p[idx]
        dpdphi[idx] =  (1 / temp2) * dpdphi[idx] - 1/(temp2) * p[idx] * np.dot(dpdphi[idx], p[idx])

        

    return p, np.cross(t, p, axis=1), dpdphi, (np.cross(t, dpdphi, axis=1) + np.cross(dtdphi, p, axis=1)) 



def get_kappa1_kappa2(p, q, n, curvature):
    # takes p and q from a centroid frame, n from a Frenet-Serret frame, and the curvature, returns the curvatures kappa1 and kappa2

    k1 = np.empty(n[:,0].shape)
    k2 = np.empty(n[:,0].shape)
    for idx, _ in enumerate(n):
        cos_a = np.dot(p[idx], n[idx])
        minus_sin_a = np.dot(q[idx], n[idx])
        k1[idx] = curvature[idx] * cos_a
        k2[idx] = curvature[idx] * minus_sin_a

    return k1, k2

def get_kappa3(dpdphi, dqdphi, q, p, lp, tol=1e-10):

    k3_0 = np.empty(p[:,0].shape)
    k3_1 = np.empty(p[:,0].shape)
    k3 = np.empty(p[:,0].shape)
    for idx, _ in enumerate(p):
        k3_0[idx] = np.dot(dpdphi[idx], q[idx]) / lp[idx]
        k3_1[idx] = -np.dot(dqdphi[idx], p[idx]) / lp[idx]
        k3[idx] = (k3_0[idx]+k3_1[idx]) / 2

    if np.average(np.abs(k3_0-k3_1)) > tol:
        print("Distance between methods is larger than tolerance")

    return k3

