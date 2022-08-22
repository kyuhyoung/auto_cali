#!usr/bin/env/ python
# _*_ coding:utf-8 _*_

import numpy as np


# pq 위치에 해당하는 v 벡터를 반환
def create_v(p, q, H, s):
    H = H.reshape(3, 3)
    return np.array([
        s * (H[0, p] * H[0, q]),
        s * (H[0, p] * H[1, q] + H[1, p] * H[0, q]),
        s * (H[1, p] * H[1, q]),
        s * (H[2, p] * H[0, q] + H[0, p] * H[2, q]),
        s * (H[2, p] * H[1, q] + H[1, p] * H[2, q]),
        s * (H[2, p] * H[2, q])
    ])
    

def get_intrinsics_param(H):
    # V matrix
    # V = np.array([])
    V = []
    for i in range(len(H)):
        # V = np.append(V, np.array([create_v(0, 1, H[i]), create_v(0, 0 , H[i])- create_v(1, 1 , H[i])]))
        V.append(create_v(0, 1, H[i], 1))
        V.append(create_v(0, 0 , H[i], 1)- create_v(1, 1 , H[i], 1))     
    V = np.array(V)

    # V*b = 0
    U, S, VT = np.linalg.svd((np.array(V, dtype='float')).reshape((-1, 6)))
    b = VT[-1]

    w = b[0] * b[2] * b[5] - b[1] * b[1] * b[5] - b[0] * b[4] * b[4] + 2 * b[1] * b[3] * b[4] - b[2] * b[3] * b[3]
    d = b[0] * b[2] - b[1] * b[1]

    alpha = np.sqrt(w / (d * b[0]))
    beta = np.sqrt(w / d**2 * b[0])
    gamma = np.sqrt(w / (d**2 * b[0])) * b[1]
    uc = (b[1] * b[4] - b[2] * b[3]) / d
    vc = (b[1] * b[3] - b[0] * b[4]) / d

    return np.array([
        [alpha, gamma, uc],
        [0,     beta,  vc],
        [0,     0,      1]
    ])

def create_v_1_ABCDEF(h_s, h):
    h_1 = h[:, 0];  h_2 = h[:, 1]
    h_s_1 = h_s[:, 0];  h_s_2 = h_s[:, 1]
    sn_s_1 = np.dot(h_s_1, h_s_1);  sn_s_2 = np.dot(h_s_2, h_s_2);
    s1 = sn_s_1;    s2 = sn_s_2;
    h1x = h_1[0];   h1y = h_1[1];   h1z = h_1[2];   
    h2x = h_2[0];   h2y = h_2[1];   h2z = h_2[2]; 
    return [s2 * h1x * h1x - s1 * h2x * h2x, 
            2 * s2 * h1x * h1y - 2 * s1 * h2x * h2y,
            2 * s2 * h1x * h1z - 2 * s1 * h2x * h2z,
            s2 * h1y * h1y - s1 * h2y * h2y,
            2 * s2 * h1y * h1z - 2 * s1 * h2y * h2z,
            s2 * h1z * h1z - s1 * h2z * h2z]

def create_v_1_AD(h_s, h):
    h_1 = h[:, 0];  h_2 = h[:, 1]
    h_s_1 = h_s[:, 0];  h_s_2 = h_s[:, 1]
    sn_s_1 = np.dot(h_s_1, h_s_1);  sn_s_2 = np.dot(h_s_2, h_s_2);
    s1 = sn_s_1;    s2 = sn_s_2;
    h1x = h_1[0];   h1y = h_1[1];   h1z = h_1[2];   
    h2x = h_2[0];   h2y = h_2[1];   h2z = h_2[2]; 
    return [s2 * h1x * h1x - s1 * h2x * h2x, s2 * h1y * h1y - s1 * h2y * h2y], (s2 * h1z * h1z - s1 * h2z * h2z)


def create_v_2_ABCDEF(h_s, h):
    h_1 = h[:, 0];  h_2 = h[:, 1]
    h_s_1 = h_s[:, 0];  h_s_2 = h_s[:, 1]
    sn_s_1 = np.dot(h_s_1, h_s_1);  #sn_s_2 = np.dot(h_s_2, h_s_2);
    dp_h_s_1_h_s_2 = np.dot(h_s_1, h_s_2)
    s1 = sn_s_1;    s12 = dp_h_s_1_h_s_2;
    h1x = h_1[0];   h1y = h_1[1];   h1z = h_1[2];   
    h2x = h_2[0];   h2y = h_2[1];   h2z = h_2[2]; 
    return [s12 * h1x * h1x - s1 * h1x * h2x, 
            2 * s12 * h1x * h1y - s1 * h1x * h2y - s1 * h1y * h2x,
            2 * s12 * h1x * h1z - s1 * h1x * h2z - s1 * h1z * h2x,
            s12 * h1y * h1y - s1 * h1y * h2y,
            2 * s12 * h1y * h1z - s1 * h1z * h2y - s1 * h1y * h2z,
            s12 * h1z * h1z - s1 * h1z * h2z]



def create_v_2_AD(h_s, h):
    h_1 = h[:, 0];  h_2 = h[:, 1]
    h_s_1 = h_s[:, 0];  h_s_2 = h_s[:, 1]
    sn_s_1 = np.dot(h_s_1, h_s_1);  #sn_s_2 = np.dot(h_s_2, h_s_2);
    dp_h_s_1_h_s_2 = np.dot(h_s_1, h_s_2)
    s1 = sn_s_1;    s12 = dp_h_s_1_h_s_2;
    h1x = h_1[0];   h1y = h_1[1];   h1z = h_1[2];   
    h2x = h_2[0];   h2y = h_2[1];   h2z = h_2[2]; 
    return [s12 * h1x * h1x - s1 * h1x * h2x, s12 * h1y * h1y - s1 * h1y * h2y], (s12 * h1z * h1z - s1 * h1z * h2z)


def solve_ax_0(A, eps=1e-15):
    print('A :\n{}'.format(A))
    print('A.shape : {}'.format(A.shape))
    u, s, vh = np.linalg.svd(A)
    null_space = np.compress(s <= eps, vh, axis=0)
    return null_space.T


def get_intrinsics_param_vir_temp(H_v, H_c, intrinsic_vir, extrinsic_vir):
    V = []
    for i in range(len(H_v)):
        H_vi = H_v[i].reshape(3, 3)

        rt_vir_H = np.delete(extrinsic_vir[i], 2, axis = 1)
        vir_H = np.matmul(intrinsic_vir, rt_vir_H)
        # print(vir_H)

        H_v0 = np.array(vir_H[:,0])
        H_v1 = np.array(vir_H[:,1])
        
        v1 = create_v(0, 0, H_c[i], np.dot(H_v1, H_v1)) -  create_v(1, 1, H_c[i], np.dot(H_v0, H_v0))
        v2 = create_v(0, 0, H_c[i], np.dot(H_v0, H_v1)) -  create_v(1, 0, H_c[i], np.dot(H_v0, H_v0))
        V.append(v1)
        V.append(v2)
        
    V = np.array(V)
    
    # V*b = 0
    U, S, VT = np.linalg.svd((np.array(V, dtype='float')).reshape((-1, 6)))
    b = VT[-1]

    w = b[0] * b[2] * b[5] - b[1] * b[1] * b[5] - b[0] * b[4] * b[4] + 2 * b[1] * b[3] * b[4] - b[2] * b[3] * b[3]
    d = b[0] * b[2] - b[1] * b[1]

    alpha = np.sqrt(w / (d * b[0]))
    beta = np.sqrt(w / d**2 * b[0])
    gamma = np.sqrt(w / (d**2 * b[0])) * b[1]
    uc = (b[1] * b[4] - b[2] * b[3]) / d
    vc = (b[1] * b[3] - b[0] * b[4]) / d

    return np.array([
        [alpha, gamma, uc],
        [0,     beta,  vc],
        [0,     0,      1]
    ])

