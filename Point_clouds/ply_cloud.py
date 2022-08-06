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





# Generate and visualize Dense model outputted by Colmap. Also access and manipulate depth pixels         
def ply_cloud(depth):
    global ply_file
    global point_cloud_in_numpy
    global colors_cloud_in_numpy
    global pcd2
    global img_bin
    global camera_bin2
    global pointy
    global coly
    

    ply_file = ('Reconstruction/dense/0/fused.ply')
    

    pcd2 = o3d.io.read_point_cloud(ply_file)
    
    point_cloud = PyntCloud.from_file(ply_file)
    xyz_arr = point_cloud.points.loc[:, ["x", "y", "z"]].to_numpy()
    normal_arr = point_cloud.points.loc[:, ["nx", "ny", "nz"]].to_numpy()
    color_arr = point_cloud.points.loc[:, ["red", "green", "blue"]].to_numpy()
    #print(xyz_arr)
    
    pointy = xyz_arr
    coly = color_arr
    
    points = []
    colors = []
    points2 = []
    points3 = []
    colors2 = []

    # Individually access the columns from the numpy point cloud
    points7 = pointy
    print(pointy)
    coly7 = coly
    colyx = coly7[:, 0:1]
    colyy = coly7[:, 1:2]
    colyz = coly7[:, 2:3]
    
    # Remove brackets and quotations marks
    CX0 = colyx.flatten()
    CY0 = colyy.flatten()
    CZ0 = colyz.flatten()

    #print(colyx)
    #print(coly.shape)
    #print(points.shape)
    pointyx = points7[:, 0:1]
    #print(pointyx)
    pointyy = points7[:, 1:2]
    pointyz = points7[:, 2:3]
    X0 = pointyx.flatten()
    Y0 = pointyy.flatten()
    Z0 = pointyz.flatten()
    X = X0
    Y = Y0
    Z = Z0
    #X = ((str(pointyx[0]).lstrip('[').rstrip(']')))
    #print('B4',pointyx)
    #print('Aff',X)
    sizex = len(X)
    #print(sizex)
    #sizey = len(Y)
    #sizez = len(Z)
    #print('Size of X', sizex)
    #print('Size of Y', sizey)
    #print('Size of Z', sizez)
    #mask = np.abs(pointyz) > 5
    #pointyz = pointyz(mask)
    #points[:, 2:3] = np.abs(points[:, 2:3]) < 8
    # Loop through all the points in the point cloud
    for i in range(sizex):
        X2 = X[i]
        CX = CX0[i]
        #X3 = str(X2).translate(translation)
        Y2 = Y[i]
        CY = CY0[i]
        
        Z2 = Z[i]
        CZ = CZ0[i]
            
        if (Z2 > depth):
            points.append((X2,Y2,Z2))
            colors.append((CX,CY,CZ))
        else:
            pass
            
    points = np.asarray(points)
    #print('Ply Color:', colors)
    colors = np.asarray(colors)
    #points2v = points[:, 0:3]
    #points3 = np.asarray(points3)
    #print(points2)
    #print('After', points2.shape)
    #print('Point3', points.shape)
    #print(points2)
    #print(points3)
    #print(colors)
    pcd = o3d.geometry.PointCloud()

    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors/255)
    pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

    #mesh_t = copy.deepcopy(pcd).transform(Matrix)
    o3d.visualization.draw_geometries([pcd])
    #view_pcd.set_front((0.1, 0.8, 1.3))
    #vis = o3d.visualization.Visualizer()
    #view_pcd = vis.get_view_control()
    #view_pcd.set_front((-0.15, 1.3, 2.2))
    #view_pcd.set_up((0, 1, 0))
    #view_pcd.set_zoom(0.45)
    
    
    #view_pcd.change_field_of_view(step = fov_step)
    #vis.run()
    #vis.destroy_window() 
    
    #points.append((xyz_arr))
    #points = np.asarray(points)
    #pcd = o3d.geometry.PointCloud()
    #pcd.xyz_arr = o3d.utility.Vector3dVector(xyz_arr)
    
    #o3d.visualization.draw_geometries([pcd2]) 
    #point_cloud_in_numpy = np.asarray(pcd2.points) 
    #colors_cloud_in_numpy = np.asarray(pcd2.colors) 
