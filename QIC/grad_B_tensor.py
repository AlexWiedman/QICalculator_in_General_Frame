import numpy as np
from .util import Struct

def calculate_grad_B_tensor(self):

    s = self

    tensor = Struct()

    factor = s.B0 * s.B0 / (s.Bbar * s.d_l_d_varphi)

    tensor.tp = s.sG * s.B0 * s.k1
    tensor.pt = tensor.tp
    tensor.tq = s.sG * s.B0 * s.k2
    tensor.qt = tensor.tq
    
    tensor.pp = factor * (s.d_X1c_d_varphi * s.Y1s - s.d_X1s_d_varphi * s.Y1c + s.iotaN * (s.X1s * s.Y1s + s.X1c * s.Y1c))
    tensor.qq = factor * (s.X1c * s.d_Y1s_d_varphi - s.X1s * s.d_Y1c_d_varphi - s.iotaN * (s.X1s * s.Y1s + s.X1c * s.Y1c))
    tensor.qp = factor * (s.X1c * s.d_X1s_d_varphi - s.X1s * s.d_X1c_d_varphi - s.sG * s.Bbar * s.k3 * s.d_l_d_varphi / s.B0 - s.iotaN * (s.X1s * s.X1s + s.X1c * s.X1c))
    tensor.pq = factor * (s.Y1s * s.d_Y1c_d_varphi - s.Y1c * s.d_Y1s_d_varphi + s.sG * s.Bbar * s.k3 * s.d_l_d_varphi / s.B0 + s.iotaN * (s.Y1s * s.Y1s + s.Y1c * s.Y1c))
    tensor.tt = s.sG * np.matmul(s.d_d_varphi, s.B0) / s.d_l_d_varphi

    s.grad_b_tensor = tensor

    t = s.tangent_cylindrical.transpose()
    p = s.p_cylindrical.transpose()
    q = s.q_cylindrical.transpose()
    self.grad_B_tensor_cylindrical = np.array([[
                              tensor.pp * p[i] * p[j] \
                            + tensor.qp * q[i] * p[j] + tensor.pq * p[i] * q[j] \
                            + tensor.qq * q[i] * q[j] \
                            + tensor.tp * t[i] * p[j] + tensor.pt * p[i] * t[j] \
                            + tensor.tq * t[i] * q[j] + tensor.qt * q[i] + t[j] \
                            + tensor.tt * t[i] * t[j]
                        for i in range(3)] for j in range(3)])
    
    self.grad_B_colon_grad_B = tensor.tn * tensor.tn + tensor.nt * tensor.nt \
        + tensor.bb * tensor.bb + tensor.nn * tensor.nn \
        + tensor.nb * tensor.nb + tensor.bn * tensor.bn \
        + tensor.tt * tensor.tt
    
    self.L_grad_B = s.B0 * np.sqrt(2 / self.grad_B_colon_grad_B)
    self.inv_L_grad_B = 1.0 / self.L_grad_B



