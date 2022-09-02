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
#from main import depth_map
#from Dependencies.read_write_model import read_model, read_next_bytes, read_cameras_text, read_cameras_binary, read_images_binary, read_array, write_array, qvec2rotmat



# Generate and visualize a 3D point cloud of the environment from the image perspective   
def point_cloud(depth_map, img4, depth, fx, fy):


    
    depthz = depth_map
    #depthz = depth_map[:, :, 0]
    #print('Updated Depth:'. depthz.shape)
    #print(depthz.shape)
    #colors = []
    poi= []
    points= []
    colu = []

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
    colors = list(img4.getdata())
    #print('Pixel val:' ,len(pixel_values))

    # Normalizes the points with the camera intrinsic parameters
    x = (u - centeru) * depthz / fx
    y = (v - centerv) * depthz / fy

    # Divide the depth to compensate for the low resolution input image and depth for monocular depth model
    z = depthz/6
    #z =  depthz / depthz.max() * x.max()
    #u = np.int
    #print(u)
    #print(v)
    
    
    #colors = colors[0:3]
    
    x = np.reshape(x, (width * height, 1)).astype(float)
    y = np.reshape(y, (width * height, 1)).astype(float)
    z = np.reshape(z, (width * height, 1)).astype(float)

    points = np.concatenate((x, y, z), axis=1)


   # This function loops through every pixel on the image to modify and manipulate depth. However, depending on the number of pixels, it can be very slow
    #for v in range(depthz.shape[0]):
        #for u in range(depthz.shape[1]):

            # Access the depth(Z) axis of the image for every pixel
            #Z = depthz[v, u]/5 

            #Only plot the depth from a specific point away from the camera
            #if (Z > depth):

                #X = (u - centeru) * Z / fx
                #Y = (v - centerv) * Z / fy
                #points.append((X, Y, Z))
                #colors.append(img4.getpixel((u, v)))

            #else:
                #pass


    # Convert list to Numpy array
    points =  np.asarray(points)
    #print('Camera Bin', camera_bin)

    #K = np.asarray(camera_bin)
    # intrinsics
    K = np.identity(3)
    K[0, 0] = fx
    K[1, 1] = fy
    #K[0, 2] = cx
    #K[1, 2] = cy
    #print('K', K)
    #print('k shape', K.shape)
    #print('k dim', K.ndim)

   
    #print('Points V1:', points.shape)
    #points2 = np.concatenate((x, y, z, np.ones_like(x)), axis=1)


    #rot = np.ones((3,3))
    #reshape(3,3)
    #print(rot)
    
    
  

    colors = np.asarray(colors)
    print('Colors Size:', len(colors))
    print('Point Size:', len(points))

    
    pcd = o3d.geometry.PointCloud()

    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors/255)

    # Transforms and flips the world space to rectify the invertion
    pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

  
    #mesh_t = copy.deepcopy(pcd).transform(Matrix)
    
    #o3d.visualization.draw_geometries([pcd, mesh_t])
    #o3d.visualization.draw_geometries([pcd])

    return pcd
    
    