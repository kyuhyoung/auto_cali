#!usr/bin/env/ python
# _*_ coding:utf-8 _*_

import cv2 as cv
import numpy as np
import numpy as geek
import os
from sympy import Matrix
#from scipy.linalg import svd
import scipy
#from step.intrinsics import get_intrinsics_param_vir


image_size = (5184, 3456)
sensor_size = (22.3, 14.9)
image_size_vir = (3840, 2160)
sensor_size_vir = (42.67, 24)

def null(A, eps=1e-12):
    u, s, vh = scipy.linalg.svd(A)
    padding = max(0,np.shape(A)[1]-np.shape(s)[0])
    null_mask = np.concatenate(((s <= eps), np.ones((padding,),dtype=bool)),axis=0)
    null_space = scipy.compress(null_mask, vh, axis=0)
    return scipy.transpose(null_space)

def null_space(A, rcond=None):
    u, s, vh = scipy.linalg.svd(A, full_matrices=True)
    #u, s, vh = svd(A, full_matrices=False)
    M, N = u.shape[0], vh.shape[1]
    if rcond is None:
        rcond = np.finfo(s.dtype).eps * max(M, N)
    tol = np.amax(s) * rcond
    num = np.sum(s > tol, dtype=int)
    Q = vh[num:,:].T.conj()
    return Q


def solve_ax_0(a):
    
    t0 = np.linalg.lstsq(a, 0.000000001 * np.ones((a.shape[0], 1)))[0]
    t0 = t0.reshape((-1, ))
    print('t0.shape : {}'.format(t0.shape));    #exit(0)
    print('t0 : \n{}'.format(t0));    #exit(0)
    t0 /= np.linalg.norm(t0)
    if t0[-1] < 0:
        t0 *= -1
    return t0
    #t2 = null(a)
    #print('t2 : \n{}'.format(t2));  exit(0);
    #t1 = null_space(a)
    #print('t1 : \n{}'.format(t1));  exit(0);

    '''
    A = Matrix(a)
    t0 = A.nullspace()
    print('t0 : \n{}'.format(t0))
    print('len(t0) : {}'.format(len(t0)))
    print('type(t0) : {}'.format(type(t0))); exit(0)    #   list
    '''
    #print('a :\n{}'.format(a));
    b = a[:, 0].copy()
    #print('b :\n{}'.format(b));
    # we impose a condition that first term be 1, 
    t0 = a[:, 1:]
    #print('t0 :\n{}'.format(t0));
    #x = np.linalg.lstsq(a[:, 1:], -b)[0]
    x, t1, t2, t3 = np.linalg.lstsq(t0, -b)
    #print('t1 :\n{}'.format(t1));
    #print('x b4 :\n{}'.format(x));
    x = np.r_[1, x]
    #print('x after :\n{}'.format(x));   
    x /= np.linalg.norm(x)
    #print('x after 2 :\n{}'.format(x));   
    #print('a.dot(x) :\n{}'.format(a.dot(x)))
    #exit(0)
    return x


def create_v_1_ACDEF(h_s, h):
    h_1 = h[:, 0];  h_2 = h[:, 1]
    h_s_1 = h_s[:, 0];  h_s_2 = h_s[:, 1]
    sn_s_1 = np.dot(h_s_1, h_s_1);  sn_s_2 = np.dot(h_s_2, h_s_2);
    s1 = sn_s_1;    s2 = sn_s_2;
    h1x = h_1[0];   h1y = h_1[1];   h1z = h_1[2];   
    h2x = h_2[0];   h2y = h_2[1];   h2z = h_2[2]; 
    return [s2 * h1x * h1x - s1 * h2x * h2x, 
            #2 * s2 * h1x * h1y - 2 * s1 * h2x * h2y,
            2 * s2 * h1x * h1z - 2 * s1 * h2x * h2z,
            s2 * h1y * h1y - s1 * h2y * h2y,
            2 * s2 * h1y * h1z - 2 * s1 * h2y * h2z,
            s2 * h1z * h1z - s1 * h2z * h2z]

def create_v_2_ACDEF(h_s, h):
    h_1 = h[:, 0];  h_2 = h[:, 1]
    h_s_1 = h_s[:, 0];  h_s_2 = h_s[:, 1]
    sn_s_1 = np.dot(h_s_1, h_s_1);  #sn_s_2 = np.dot(h_s_2, h_s_2);
    dp_h_s_1_h_s_2 = np.dot(h_s_1, h_s_2)
    s1 = sn_s_1;    s12 = dp_h_s_1_h_s_2;
    h1x = h_1[0];   h1y = h_1[1];   h1z = h_1[2];   
    h2x = h_2[0];   h2y = h_2[1];   h2z = h_2[2]; 
    return [s12 * h1x * h1x - s1 * h1x * h2x, 
            #2 * s12 * h1x * h1y - s1 * h1x * h2y - s1 * h1y * h2x,
            2 * s12 * h1x * h1z - s1 * h1x * h2z - s1 * h1z * h2x,
            s12 * h1y * h1y - s1 * h1y * h2y,
            2 * s12 * h1y * h1z - s1 * h1z * h2y - s1 * h1y * h2z,
            s12 * h1z * h1z - s1 * h1z * h2z]




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



def get_intrinsics_cam(H_v_s, H_s_c, is_ABCDEF):
    # V matrix
    # V = np.array([])
    V = []
    #A = []; B = []
    for i in range(len(H_v_s)):
        # print(H_v[i])
        #H_vi = H_v[i].reshape(3, 3)
        #H_v0 = np.array(H_vi[:,0])
        #H_v1 = np.array(H_vi[:,1])
        
        #v1 = create_v(0, 0, H_c[i], np.dot(H_v1, H_v1)) -  create_v(1, 1, H_c[i], np.dot(H_v0, H_v0))
        #v2 = create_v(0, 0, H_c[i], np.dot(H_v0, H_v1)) -  create_v(1, 0, H_c[i], np.dot(H_v0, H_v0))
        #'''
        if is_ABCDEF:
            v1 = create_v_1_ABCDEF(H_v_s[i], H_s_c[i])
            v2 = create_v_2_ABCDEF(H_v_s[i], H_s_c[i]) #0, 0, H_c[i], np.dot(H_v0, H_v1)) -  create_v(1, 0, H_c[i], np.dot(H_v0, H_v0))
        else:      
            v1 = create_v_1_ACDEF(H_v_s[i], H_s_c[i])
            v2 = create_v_2_ACDEF(H_v_s[i], H_s_c[i]) #0, 0, H_c[i], np.dot(H_v0, H_v1)) -  create_v(1, 0, H_c[i], np.dot(H_v0, H_v0))
       
        V.append(v1);   V.append(v2)
        #'''
        '''
        a1, b1 = create_v_1_AD(H_v_s[i], H_s_c[i])
        a2, b2 = create_v_2_AD(H_v_s[i], H_s_c[i]) #0, 0, H_c[i], np.dot(H_v0, H_v1)) -  create_v(1, 0, H_c[i], np.dot(H_v0, H_v0))
        A.append(a1);   A.append(a2);        B.append(b1);   B.append(b2);   
        '''
        
    V = np.array(V)
    
    if is_ABCDEF:
        ABCDEF = solve_ax_0(V)
        #print('ABCDEF :\n{}'.format(ABCDEF));   #exit(0)
        B11 = ABCDEF[0];  B12 = ABCDEF[1];  B13 = ABCDEF[2];  B22 = ABCDEF[3];  B23 = ABCDEF[4];  B33 = ABCDEF[5];
    else:       
        ACDEF = solve_ax_0(V)
        #print('ACDEF :\n{}'.format(ACDEF));   #exit(0)
        B11 = ACDEF[0];  B12  = 0;     B13 = ACDEF[1];  B22 = ACDEF[2];  B23 = ACDEF[3];  B33 = ACDEF[4];
    '''
    U, S, VT = np.linalg.svd(V)
    print('U :\n{}'.format(U))
    print('S :\n{}'.format(S))
    #ABCDEF = solve_ax_0(V)
    print('VT :\n{}'.format(VT)) 
    w, VT2 = np.linalg.eig(V.transpose() @ V)
    print('w :\n{}'.format(w))
    print('VT2 :\n{}'.format(VT2)); #exit(0) 

    '''
    '''
    mat_A = np.array(A);    vec_b = np.array(B)
    print('mat_A :\n{}'.format(mat_A)); print('vec_b :\n{}'.format(vec_b));
    print('mat_A.shape : {}'.format(mat_A.shape)); print('vec_b.shape : {}'.format(vec_b.shape));
    vec_x, residual, rank, singular_value  = np.linalg.lstsq(mat_A, vec_b)
    
    print('vec_x : \n{}'.format(vec_x));    #exit(0) 
    '''    

    '''
    ABCDEF = VT2[-1]
    #U, S, VT = np.linalg.svd((np.array(V, dtype='float')).reshape((-1, 6)))
    '''
    v0 = (B12 * B13 - B11 * B23) / (B11 * B22 - B12 * B12)
    #print('v0 : {}'.format(v0));    #exit(0);
    mat_B = np.array([[B11, B12, B13], [B12, B22, B23], [B13, B23, B33]])
    print('mat_B :\n{}'.format(mat_B)); 
    print('np.linalg.eigvalsh(mat_B) : {}'.format(np.linalg.eigvalsh(mat_B)))
    K_inv_T_recon = np.linalg.cholesky(100 * mat_B)
    print('K_inv_T_recon :\n{}'.format(K_inv_T_recon))
    K_inv_recon = K_inv_T_recon.transpose()
    print('K_inv_recon :\n{}'.format(K_inv_recon))
    K_recon = np.linalg.inv(K_inv_recon)
    print('K_recon :\n{}'.format(K_recon)); exit(0);
 

    #'''
    '''
    AD = VT[-1]
    A = AD[0];  D = AD[1]
    mat_B = np.array([[A, 0, 0], [0, D, 0], [0, 0, 1]])
    '''


    K_inv = np.linalg.cholesky(mat_B)
    print('K_inv :\n {}'.format(K_inv));    exit(0)

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




def concatenate_images(li_im, is_horizontal):
    im_total = None
    w_total = 0;    h_total = 0
    if is_horizontal:
        for im in li_im:
            hei, wid, chn = im.shape
            w_total += wid
            if h_total < hei:
                h_total = hei
        im_total = np.ones((h_total, w_total, 3), np.uint8) * 0
        x_from = 0
        for im in li_im:
            hei, wid, chn = im.shape
            im_total[0 : hei, x_from : x_from + wid, :] = im
            x_from += wid
    else:
        for im in li_im:
            hei, wid, chn = im.shape
            h_total += hei
            if w_total < wid:
                w_total = wid
        im_total = np.ones((h_total, w_total, 3), np.uint8) * 0
        y_from = 0
        for im in li_im:
            hei, wid, chn = im.shape
            im_total[y_from : y_from + hei, 0 : wid, :] = im
            y_from += hei
    return im_total
    
def check_homography(h_first_2_second, first, second, wh, img_second, fn_img):
    
    w, h = wh;
    #print('h : {}, w : {}'.format(h, w));    #exit(0);
    #print('first : \n{}'.format(first));    #exit(0);
    ul_x = first[0, 0];             ul_y = first[0, 1]; 
    ur_x = first[w - 1, 0];         ur_y = first[w - 1, 1]; 
    ll_x = first[w * (h - 1),0];    ll_y = first[w * (h - 1), 1];
    lr_x = first[w * h - 1, 0];     lr_y = first[w * h - 1, 1];
    print('ul : ({}, {}), ur : ({}, {}), ll : ({}, {}), lr : ({}, {})'.format(ul_x, ul_y, ur_x, ur_y, ll_x, ll_y, lr_x, lr_y));  #exit(0)
    #corners1 = np.array([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2).astype(np.float32)
    corners1 = np.array([[ul_x, ul_y], [ll_x, ll_y], [lr_x, lr_y], [ur_x, ur_y]]).reshape(-1, 1, 2).astype(np.float32)
    corners2 = cv.perspectiveTransform(corners1, h_first_2_second)
    cv.polylines(img_second, [np.int32(corners2)], True, (0, 255, 0), 2, cv.LINE_AA)
    #print('corners2.shape : {}'.format(corners2.shape));
    ul_homo = (int(corners2[0, 0, 0]), int(corners2[0, 0, 1]))
    ll_homo = (int(corners2[1, 0, 0]), int(corners2[1, 0, 1]))
    lr_homo = (int(corners2[2, 0, 0]), int(corners2[2, 0, 1]))
    ur_homo = (int(corners2[3, 0, 0]), int(corners2[3, 0, 1]))
    cv.putText(img_second, 'ul', ul_homo, cv.FONT_HERSHEY_COMPLEX, 5, (0, 0, 255), 2) 
    cv.putText(img_second, 'll', ll_homo, cv.FONT_HERSHEY_COMPLEX, 5, (0, 0, 255), 2) 
    cv.putText(img_second, 'lr', lr_homo, cv.FONT_HERSHEY_COMPLEX, 5, (0, 0, 255), 2) 
    cv.putText(img_second, 'ur', ur_homo, cv.FONT_HERSHEY_COMPLEX, 5, (0, 0, 255), 2) 
    cv.imwrite(fn_img, img_second)
    print('homography check image is saved at {}'.format(fn_img));  #exit(0)

def compute_homography(vir_file_dir, cam_file_dir, save_result_image):
    vir_pic_name = os.listdir(vir_file_dir)
    cam_pic_name = os.listdir(cam_file_dir)
    
    cross_corners = [9, 6]
    #real_coor = np.zeros((cross_corners[0] * cross_corners[1], 3), np.float32)
    real_coor = np.zeros((cross_corners[0] * cross_corners[1], 2), np.float32)
    real_coor = np.mgrid[0:9, 0:6].T.reshape(-1, 2)
   
    print('real_coor : \n{}'.format(real_coor.transpose()));  #exit(0);

    vir_scr_homographies =[]
    scr_cam_homographies =[]
    
    real_points_x_y = []
    vir_pic_points = []
    cam_pic_points = []
    
    n_img = len(vir_pic_name)
    for i in range(n_img):
        print('===== {} / {} ====='.format(i, n_img))
        vir_pic_path = os.path.join(vir_file_dir, vir_pic_name[i])
        cam_pic_path = os.path.join(cam_file_dir, cam_pic_name[i])
        print('\tvir_pic_path : {}, cam_pic_path : {}'.format(vir_pic_path, cam_pic_path))
        
        vir_img = cv.imread(vir_pic_path)
        cam_img = cv.imread(cam_pic_path)
        
        vir_succ, vir_pic_coor = cv.findChessboardCorners(vir_img, (cross_corners[0], cross_corners[1]), None)
        cam_succ, cam_pic_coor = cv.findChessboardCorners(cam_img, (cross_corners[0], cross_corners[1]), None)
        
        if vir_succ == True and cam_succ == True:
            #print('vir_pic_coor.shape b4 :{}'.format(vir_pic_coor.shape));   #exit(0)   #   (54, 1, 2)   
            vir_pic_coor = vir_pic_coor.reshape(-1, 2); vir_pic_coor = np.flip(vir_pic_coor, 0)
            #print('vir_pic_coor.shape after :{}'.format(vir_pic_coor.shape));   exit(0) #   (54, 2)       
            cam_pic_coor = cam_pic_coor.reshape(-1, 2); cam_pic_coor = np.flip(cam_pic_coor, 0)
            if save_result_image:            
                cv.drawChessboardCorners(vir_img, (cross_corners[0], cross_corners[1]), vir_pic_coor, vir_succ)
                cv.drawChessboardCorners(cam_img, (cross_corners[0], cross_corners[1]), cam_pic_coor, cam_succ)
                im_concat = concatenate_images([vir_img, cam_img], False)
                fn_corner = 'output/corner/{}'.format(vir_pic_name[i]);
                cv.imwrite(fn_corner, im_concat);   #exit(0);
                print('Saved corner detection result at {}'.format(fn_corner)); #exit(0)
            vir_pic_points.append(vir_pic_coor)
            cam_pic_points.append(cam_pic_coor)
            
            #real_points_x_y.append(real_coor[:, :2])
            real_points_x_y.append(real_coor)
            #print('vir_pic_coor :\n{}'.format(vir_pic_coor));   exit(0)
            #vir_scr_H, status = cv.findHomography(real_coor[:, :2], vir_pic_coor, cv.RANSAC, 5.0)
            vir_scr_H, status = cv.findHomography(real_coor, vir_pic_coor, cv.RANSAC, 5.0)
            scr_cam_H, status = cv.findHomography(vir_pic_coor, cam_pic_coor, cv.RANSAC, 5.0)
            if save_result_image:
                check_homography(vir_scr_H, real_coor, vir_pic_coor, cross_corners, vir_img, 'output/homo/homo_vir_2_scr_{}.png'.format(i)) 
                check_homography(scr_cam_H, vir_pic_coor, cam_pic_coor, cross_corners, cam_img, 'output/homo/homo_scr_2_cam_{}.png'.format(i)) 
                print(vir_scr_H)
            #a = geek.identity(3)
            ##size = (5184, 3456)
            ##im_dst = cv.warpPerspective(vir_img, final_vir_cam_H, size)
            ##resize_im_dst = cv.resize(im_dst, (1296,864))
            ##fn_resize = 'output/{}'.format(vir_pic_name[i]);   
            ##print('\tfn_resize : {}'.format(fn_resize));  #exit(0);
            ##cv.imwrite(fn_resize, resize_im_dst)
            #cv.imshow('h', resize_im_dst)
            #cv.waitKey(0)
            
            vir_scr_homographies.append(vir_scr_H)
            scr_cam_homographies.append(scr_cam_H)
    
    return vir_scr_homographies, scr_cam_homographies
    

   
if __name__ == "__main__":
    '''
    vir_file_dir = 'D:/test/CameraControl/CameraControl/bin/Debug/blender'
    cam_file_dir = 'D:/test/CameraControl/CameraControl/bin/Debug/monitor_55'
    '''

    #import numpy as np

    a = np.arange(12).reshape(4, 3)
    '''
    a = np.array([  
                    [1, 4, 5],
                    [5, 8, 1],
                    [4, 1, 8],
                    [5, 9, 2],
                    [100, 2, 50],
                    [-300, -20, -900]
                    ])
    '''
    #x = solve_ax_0(a)

    #print('mat_B :\n{}'.format(mat_B)) 
    #K_ideal = np.array([[14568.3, 0, 2626.1], [0, 14572.2, 1800.5], [0, 0, 1]])
    #K_ideal = np.array([[14568.3, 0, 2626.1], [0, 14572.2, 1800.5], [0, 0, 1]])
    K_ideal = np.array([[14568.3, 0, 2626.1], [0, 14572.2, 1800.5], [0, 0, 1]])
    print('K_ideal :\n{}'.format(K_ideal))
    K_ideal_inv = np.linalg.inv(K_ideal);
    print('K_ideal_inv :\n{}'.format(K_ideal_inv))
    K_ideal_inv_T = K_ideal_inv.transpose()
    print('K_ideal_inv_T :\n{}'.format(K_ideal_inv_T))
    K_ideal_T = K_ideal.transpose()
    K_ideal_T_inv = np.linalg.inv(K_ideal_T)
    print('K_ideal_T_inv :\n{}'.format(K_ideal_T_inv)); #exit(0)
    B_ideal = K_ideal_inv_T @ K_ideal_inv
    #B_ideal = K_ideal_inv * K_ideal_inv_T
    print('K_ideal_inv_T @ K_ideal_inv :\n{}'.format(K_ideal_inv_T @ K_ideal_inv)); #exit(0)

    #print('K_ideal_inv @ K_ideal_inv_T :\n{}'.format(K_ideal_inv @ K_ideal_inv_T)); #exit(0)
    print('np.linalg.eigvalsh(B_ideal) : {}'.format(np.linalg.eigvalsh(B_ideal)))
    K_ideal_inv_T_recon = np.linalg.cholesky(B_ideal)
    print('K_ideal_inv_T_recon :\n{}'.format(K_ideal_inv_T_recon))
    K_ideal_inv_recon = K_ideal_inv_T_recon.transpose()
    print('K_ideal_inv_recon :\n{}'.format(K_ideal_inv_recon))
    K_ideal_recon = np.linalg.inv(K_ideal_inv_recon)
    print('K_ideal_recon :\n{}'.format(K_ideal_recon))
    #exit(0)


    vir_file_dir = 'blender_9'
    cam_file_dir = 'monitor_55_9'

    vir_scr_H, scr_cam_H = compute_homography(vir_file_dir, cam_file_dir, False)
    
    #intrinsic = get_intrinsics_param_vir(vir_scr_H, scr_cam_H)
    #intrinsic = get_intrinsics_cam(vir_scr_H, scr_cam_H, True)
    intrinsic = get_intrinsics_cam(vir_scr_H, scr_cam_H, False)
    print(intrinsic)
    
    fx_mm = intrinsic[0][0] * sensor_size[0] / image_size[0]
    fy_mm = intrinsic[1][1] * sensor_size[1] / image_size[1]
    print('focal length(mm): {}, {}'.format(fx_mm, fy_mm))
