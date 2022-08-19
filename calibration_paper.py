#!usr/bin/env/ python
# _*_ coding:utf-8 _*_

import cv2 as cv
import numpy as np
import numpy as geek
import os
from step.intrinsics import get_intrinsics_param_vir


image_size = (5184, 3456)
sensor_size = (22.3, 14.9)
image_size_vir = (3840, 2160)
sensor_size_vir = (42.67, 24)

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
    print('h : {}, w : {}'.format(h, w));    #exit(0);
    print('first : \n{}'.format(first));    #exit(0);
    ul_x = first[0, 0];             ul_y = first[0, 1]; 
    ur_x = first[w - 1, 0];         ur_y = first[w - 1, 1]; 
    ll_x = first[w * (h - 1),0];    ll_y = first[w * (h - 1), 1];
    lr_x = first[w * h - 1, 0];     lr_y = first[w * h - 1, 1];
    print('ul : ({}, {}), ur : ({}, {}), ll : ({}, {}), lr : ({}, {})'.format(ul_x, ul_y, ur_x, ur_y, ll_x, ll_y, lr_x, lr_y));  #exit(0)
    #corners1 = np.array([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2).astype(np.float32)
    corners1 = np.array([[ul_x, ul_y], [ll_x, ll_y], [lr_x, lr_y], [ur_x, ur_y]]).reshape(-1, 1, 2).astype(np.float32)
    corners2 = cv.perspectiveTransform(corners1, h_first_2_second)
    cv.polylines(img_second, [np.int32(corners2)], True, (0, 255, 0), 2, cv.LINE_AA)
    cv.imwrite(fn_img, img_second)
    print('homography check image is saved at {}'.format(fn_img));  exit(0)

def compute_homography(vir_file_dir, cam_file_dir):
    vir_pic_name = os.listdir(vir_file_dir)
    cam_pic_name = os.listdir(cam_file_dir)
    
    cross_corners = [9, 6]
    #real_coor = np.zeros((cross_corners[0] * cross_corners[1], 3), np.float32)
    real_coor = np.zeros((cross_corners[0] * cross_corners[1], 2), np.float32)
    real_coor = np.mgrid[0:9, 0:6].T.reshape(-1, 2)
   
    print('real_coor : \n{}'.format(real_coor));  #exit(0);

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
            cv.drawChessboardCorners(vir_img, (cross_corners[0], cross_corners[1]), vir_pic_coor, vir_succ)
            cv.drawChessboardCorners(cam_img, (cross_corners[0], cross_corners[1]), cam_pic_coor, cam_succ)
            #print('vir_img.shape : {}, cam_img.shape : {}'.format(vir_img.shape, cam_img.shape));   exit(0);
            #corner_images = cv.hconcat(vir_img, cam_img)
            # h, w = corner_images.shape
            # print(w, h)
            # img_w = int (w / 5)
            # img_h = int (h / 5)
            # print(img_w, img_h)
            # resize_img = cv.resize(corner_images, (img_w, img_h))
            # cv.namedWindow("corner", cv.WINDOW_AUTOSIZE)
            # #cv.resizeWindow("corner", 1280, 480)
            # cv.imshow('corner', vir_img)
            # cv.waitKey(0)
            im_concat = concatenate_images([vir_img, cam_img], False)
            fn_corner = 'output/corner/{}'.format(vir_pic_name[i]);
            cv.imwrite(fn_corner, im_concat);   #exit(0);
            print('Saved corner detection result at {}'.format(fn_corner)); #exit(0)
            #vir_pic_coor = vir_pic_coor.reshape(-1, 2)
            #vir_pic_coor = np.flip(vir_pic_coor, 0)
            vir_pic_points.append(vir_pic_coor)
            
            #cam_pic_coor = cam_pic_coor.reshape(-1, 2);            cam_pic_coor = np.flip(cam_pic_coor, 0)
            #print('cam_pic_coor.shape :{}'.format(cam_pic_coor.shape));   #exit(0)
            #print('cam_pic_coor :\n{}'.format(cam_pic_coor));   exit(0)
            
            cam_pic_points.append(cam_pic_coor)
            
            #real_points_x_y.append(real_coor[:, :2])
            real_points_x_y.append(real_coor)
            #print('vir_pic_coor :\n{}'.format(vir_pic_coor));   exit(0)
            #vir_scr_H, status = cv.findHomography(real_coor[:, :2], vir_pic_coor, cv.RANSAC, 5.0)
            vir_scr_H, status = cv.findHomography(real_coor, vir_pic_coor, cv.RANSAC, 5.0)
            scr_cam_H, status = cv.findHomography(vir_pic_coor, cam_pic_coor, cv.RANSAC, 5.0)
            check_homography(vir_scr_H, real_coor, vir_pic_coor, cross_corners, vir_img, 'output/homo_vir_2_scr_{}.png'.format(i)) 
            check_homography(scr_cam_H, vir_pic_coor, cam_pic_coor, cross_corners, cam_img, 'output/homo_scr_2_cam_{}.png'.format(i)) 
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
    print('K_ideal_inv @ K_ideal_inv_T :\n{}'.format(K_ideal_inv @ K_ideal_inv_T)); #exit(0)
    K_ideal_inv_T_recon = np.linalg.cholesky(B_ideal)
    print('K_ideal_inv_T_recon :\n{}'.format(K_ideal_inv_T_recon))
    K_ideal_inv_recon = K_ideal_inv_T_recon.transpose()
    print('K_ideal_inv_recon :\n{}'.format(K_ideal_inv_recon))
    K_ideal_recon = np.linalg.inv(K_ideal_inv_recon)
    print('K_ideal_recon :\n{}'.format(K_ideal_recon))
    #exit(0)


    vir_file_dir = 'blender_9'
    cam_file_dir = 'monitor_55_9'

    vir_scr_H, scr_cam_H = compute_homography(vir_file_dir, cam_file_dir)
    
    intrinsic = get_intrinsics_param_vir(vir_scr_H, scr_cam_H)
    print(intrinsic)
    
    fx_mm = intrinsic[0][0] * sensor_size[0] / image_size[0]
    fy_mm = intrinsic[1][1] * sensor_size[1] / image_size[1]
    print('focal length(mm): {}, {}'.format(fx_mm, fy_mm))
