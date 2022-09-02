# Written by Hayatu Abdullahi


from PIL import ImageTk, Image
import cv2
import os
import torch
import copy
from os.path import splitext
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
from pyquaternion import quaternion


def matrix(img_bin, fx, fy, cx, cy):
    
    # Initialize camera extrinsic data for all images
    images = img_bin
    #cameras = camera_bin2
    im1 = images[1]
    im2 = images[2]
    im3 = images[3]
    im4 = images[4]
    im5 = images[5]
    im6 = images[6]
    im7 = images[7]
    im8 = images[8]
    im9 = images[9]
    im10 = images[10]
    im11 = images[11]
    im12 = images[12]
    im13 = images[13]
    im14 = images[14]
    im15 = images[15]
    im16 = images[16]
    im17 = images[17]
    im18 = images[18]
    im19 = images[19]
    im20 = images[20]

    
    # Initialize translation vector
    trans = im1.tvec
    #print('translation\n', trans1)
    #trans_backup = np.array([0.025,0.0011,-0.010])
    #trans = np.array([0.02092636,0.0030388,0.002])
    #print('trans:', trans)
    #trans2 = np.expand_dims(trans, axis=1)
    #img2 is the goat
    #img5 is the goat 2

    # Convert quaternion to rotation matrix
    rotation_matrix = o3d.geometry.get_rotation_matrix_from_quaternion(im1.qvec)
    rotation_matrix3 = o3d.geometry.get_rotation_matrix_from_quaternion(im9.qvec)
    rotation_matrix_old = o3d.geometry.get_rotation_matrix_from_quaternion(im9.qvec)
    rotation_matrix2 = np.linalg.inv(rotation_matrix)

    #print('Tvec:', im3.tvec)

    #trans = np.concatenate(im1.tvec, np.ones)
    #trans2=np.reshape(3,1)

    # Obtain camera intrinsic parameters
    K = np.identity(3)
    K[0, 0] = fx
    K[1, 1] = fy
    K[0, 2] = cx
    K[1, 2] = cy

    
    #print('pp B4', points)
    #points = points @rotation_matrix 

    #points = [(rotation_matrix @ p + trans1) for p in points]

    # Transform pixel in Camera coordinate frame
   
 
    #points =  np.asarray(points)
    #print('pp AF', points)

    #Matrix2[1, 0] = -0.8
    #Matrix[1, 0] = 1
    #Matrix[2, 0] = 1
    #print('Rot matrix 2', rotation_matrix2)
    #Matrix_inv = - np.linalg.inv(rotation_matrix) * trans


    # Combine rotation matrix and translation matrix into a 4 by 4 full matrices
    Matrix = np.vstack((np.hstack((rotation_matrix, trans[:, None])), [0, 0, 0 ,1]))
    Matrix_old = np.vstack((np.hstack((rotation_matrix_old, trans[:, None])), [0, 0, 0 ,1]))
    Matrix_inv = np.vstack((np.hstack((rotation_matrix2, trans[:, None])), [0, 0, 0 ,1]))
    #Matrix_tot =  Matrix * Matrix_inv
    #Matrix_f = Matrix * np.linalg.inv(Matrix2)
    



    return rotation_matrix, trans, Matrix, Matrix_inv, K
