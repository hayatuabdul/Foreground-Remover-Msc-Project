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
#from main import depth_map
#from Dependencies.read_write_model import read_model, read_next_bytes, read_cameras_text, read_cameras_binary, read_images_binary, read_array, write_array, qvec2rotmat



# Generate and visualize a 3D point cloud of the environment from the image perspective   
def point_cloudb(rotation_matrix, trans1, Matrix, depth_map2, img7, depth, pcd, fx, fy):
 

    depthz = depth_map2
    #depthz = depth_map[:, :, 0]
    #print('Updated Depth:'. depthz.shape)
    #print(depthz.shape)
    colors = []


# This function predefines the formula and computes the pixels faster due to pre processing
    centeru = depthz.shape[1] / 2
    centerv = depthz.shape[0] / 2
    height = depthz.shape[0]
    width = depthz.shape[1]
    
    row = np.arange(0, width, 1)
    u = np.array([row for i in np.arange(height)])
    col = np.arange(0, height, 1)
    v = np.array([col for i in np.arange(width)])
    v = v.transpose(1, 0)
    #color.append(rgb.getpixel((u, v)))
    
    #wid, hei = rgb.size
    colors = list(img7.getdata())
    #print('Pixel val:' ,len(pixel_values))

    x = (u - centeru) * depthz / fx
    y = (v - centerv) * depthz / fy
    z = depthz/1
    #z =  depthz / depthz.max() * x.max()
    #u = np.int
    #print(u)
    #print(v)
    
    
    #colors = colors[0:3]
    
    x = np.reshape(x, (width * height, 1)).astype(float)
    y = np.reshape(y, (width * height, 1)).astype(float)
    z = np.reshape(z, (width * height, 1)).astype(float)
    #print(z)

    points = np.concatenate((x, y, z), axis=1)
    points =  np.asarray(points)
    #print('pp B4', points)

    #points = points @rotation_matrix

    

    #points = [(rotation_matrix @ p + trans1) for p in points]
    #points =  np.asarray(points)
    #print('pp AF', points)

    colors = np.asarray(colors)
    
    pcd2 = o3d.geometry.PointCloud()

    pcd2.points = o3d.utility.Vector3dVector(points)
    pcd2.colors = o3d.utility.Vector3dVector(colors/255)
    pcd2.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

    mesh_t = copy.deepcopy(pcd2).transform(Matrix)
    #pcd.transform(Matrix)

    
    o3d.visualization.draw_geometries([pcd, mesh_t])

    return pcd2
    