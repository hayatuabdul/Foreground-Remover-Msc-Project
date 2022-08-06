from PIL import ImageTk, Image
import cv2
import os
import torch
import copy
from os.path import splitext
from pyntcloud import PyntCloud
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
from pyquaternion import quaternion


def matrix(img_bin):
    
    images = img_bin
    #cameras = camera_bin2
    im1 = images[1]
    im2 = images[2]
    im3 = images[3]
    im4 = images[4]
    im5 = images[5]
    im6 = images[6]
    im7 = images[7]
    #im8 = images[8]
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

    
    trans1 = im9.tvec
    #print('translation\n', trans1)
    #trans_backup = np.array([0.025,0.0011,-0.010])
    trans = np.array([0.02092636,0.0030388,0.002])
    #print('trans:', trans)
    #trans2 = np.expand_dims(trans, axis=1)

    rotation_matrix = o3d.geometry.get_rotation_matrix_from_quaternion(im6.qvec)
    rotation_matrix3 = o3d.geometry.get_rotation_matrix_from_quaternion(im9.qvec)
    rotation_matrix_old = o3d.geometry.get_rotation_matrix_from_quaternion(im9.qvec)
    rotation_matrix2 = np.linalg.inv(rotation_matrix)

    #trans = np.concatenate(im1.tvec, np.ones)
    #trans2=np.reshape(3,1)

    w2c_poses = np.array([quaternion.Quaternion(q).transformation_matrix for q in np.array(im7.qvec).astype(np.float32)])
    translations = im7.tvec
    w2c_poses[:,:3,3] = translations
    #print('poses', w2c_poses)

    
    #print('pp B4', points)
    #points = points @rotation_matrix

    #points = [(rotation_matrix @ p + trans1) for p in points]
 
    #points =  np.asarray(points)
    #print('pp AF', points)

    #Matrix2[1, 0] = -0.8
    #Matrix[1, 0] = 1
    #Matrix[2, 0] = 1
    #print('Rot matrix 2', rotation_matrix2)


    Matrix = np.vstack((np.hstack((rotation_matrix, trans1[:, None])), [0, 0, 0 ,1]))
    Matrix2 = np.vstack((np.hstack((rotation_matrix3, trans1[:, None])), [0, 0, 0 ,1]))
    Matrix_old = np.vstack((np.hstack((rotation_matrix_old, trans[:, None])), [0, 0, 0 ,1]))
    Matrix_inv = np.vstack((np.hstack((rotation_matrix2, trans[:, None])), [0, 0, 0 ,1]))
    Matrix_tot =  Matrix * Matrix_inv
    Matrix_f = Matrix * np.linalg.inv(Matrix2)


    return rotation_matrix, trans1, Matrix
