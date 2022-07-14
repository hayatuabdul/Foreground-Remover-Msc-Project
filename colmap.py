import numpy as np
import matplotlib.pyplot as plt
import cv2
import pylab as plt
import argparse
import numpy as np
import os
import collections
import struct
from PIL import Image
import open3d as o3d

from read_write_model import read_model, read_next_bytes, read_cameras_text, read_cameras_binary, read_array, write_array

from GUI import run, myClick


# Read and manipulate depth data obtained from Colmap. The depth data can then be converted to a 3d point cloud visualized in open3d


def calculate(x, y):
    
    z = x+y
    #print('Camera Intrinsics:', fx,fy,cx,cy)
    return z

def cal(z):
    
    z = z+43
    #print('Camera Intrinsics:', fx,fy,cx,cy)
    return z


def camera_intrinsic(camera):
    
    fx = float(((str(camera[0]).lstrip('[').rstrip(']'))))
    fy = float(((str(camera[1]).lstrip('[').rstrip(']'))))
    cx = float(((str(camera[2]).lstrip('[').rstrip(']'))))
    cy = float(((str(camera[3]).lstrip('[').rstrip(']'))))
    #print('Camera Intrinsics:', fx,fy,cx,cy)
    return fx, fy, cx, cy


def convert_img(path, depth_map):
    
    rgb = Image.open(path)
    rgb2 = np.asarray(rgb)
    dp_size = depth_map.shape
    if rgb2.size != depth_map.size:
        img =cv2.resize(rgb2,(dp_size[1],dp_size[0]))
        img4 = Image.fromarray(img)
        
        if img4.mode != 'RGB':
            img4 = img4.convert('RGB')
        
    print('Depth Size', depth_map.shape)
    print('Image Size',img.shape)
    
    return img4, rgb

def convert_depth(depth_map):
    
    dp =cv2.resize(depth_map,(2052,1537))
    #dp =cv2.resize(depth_map,(0,0), depth_map, 1, 1)
    print('Depth Size', depth_map.shape)
    dp_img = Image.fromarray(dp)
    dp_img = dp_img.convert('RGB')

    return dp
   
def newDepth(path):

    depth = path
    #depth2 = depth

    depth2 = []
    
    for v in range(depth.shape[0]):
        for u in range(depth.shape[1]):
            Z = depth[v, u] 
            Z2 = Z - 10
            
            depth2 = depth - Z2

    return depth2
       
    
# Convert Depth to Point cloud with only RGB range
def getPointCloud(path, distance, fx, fy, cx, cy):
    thresh = 15.0

    depth = path
    #depth2 = depth
    #rgb = Image.open('3.jpeg')
    #img =cv2.resize(img_rgb,(1300,275))
    #rgb = cv2.resize(rgb, (0,0), rgb, 0.8,0.8)

    points = []
    srcPxs = []
    depth2 = []
    
    # Loop through X and Y axis. Both width and height of the image
    for v in range(depth.shape[0]):
        for u in range(depth.shape[1]):

            # Access the depth(Z) axis of the image for every pixel
            Z = depth[v, u] 
            depth2 = depth

            
            #Only plot the depth from a specific point away from the camera
            if (Z > distance):
                
                X = (u - cx) * Z / fx
                Y = (v - cx) * Z / fy
                srcPxs.append((u, v))
                points.append((X, Y, Z))
            else:
                pass
            
            

    srcPxs = np.asarray(srcPxs).T
    points = np.asarray(points)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    return pcd
   
# Convert Depth to Point cloud with actual environment color
def getPointCloud2(rgb, path, distance, fx, fy, cx, cy):
    thresh = 15.0

    depth = path
    #rgb = Image.open(rgb_file)
    rgb2 = np.asarray(rgb)
    #depth2 = np.asarray(path)
    
    #if rgb2.size != depth.size:
        #raise Exception("Color and depth image do not have the same resolution.")
    #if rgb.mode != "RGB":
        #raise Exception("Color image is not in RGB format")
    #if depth.mode != "I":
        #raise Exception("Depth image is not in intensity format")

    
    points = []
    colors = []
    srcPxs = []
    
    for v in range(depth.shape[0]):
        for u in range(depth.shape[1]):
            
            Z = depth[v, u] 
            
            if (Z > distance):
                X = (u - cx) * Z / fx
                Y = (v - cy) * Z / fy
                srcPxs.append((u, v))
                points.append((X, Y, Z))
                colors.append(rgb.getpixel((u, v)))
            else:
                pass
            

    srcPxs = np.asarray(srcPxs).T
    points = np.asarray(points)
    colors = np.asarray(colors)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors/255)
    
    return pcd
   

    
def custom_draw_geometry_with_custom_fov(pcd, fov_step):
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)
    view_pcd = vis.get_view_control()
    #view_pcd.set_front((0.1, 0.8, 1.3))
    view_pcd.set_front((-0.15, 1.3, 2.2))
    view_pcd.set_up((0, 1, 0))
    view_pcd.set_zoom(0.45)
    view_pcd.change_field_of_view(step = fov_step)
    vis.run()
    vis.destroy_window()
    
    
    
def custom_draw_geometry_with_rotation(pcd, fov_step):

    def rotate_view(vis):
        view_pcd = vis.get_view_control()
        view_pcd.set_front((0, 0, 1))
        view_pcd.set_up((0, 1, 0))
        view_pcd.set_zoom(0.8)
        view_pcd.change_field_of_view(step = fov_step)
        #ctr.rotate(10.0, 0.0)
        return False

    
        o3d.visualization.draw_geometries_with_animation_callback([pcd],
                                                              rotate_view)  
    
def main():
    
    # Read depth and normal maps corresponding to the same image.
    if not os.path.exists('maps/depth_maps/'):
        raise FileNotFoundError("File not found: {}".format('maps/depth_maps/0.jpeg.geometric'))

    if not os.path.exists('maps/normal_maps'):
        raise FileNotFoundError("File not found: {}".format('maps/depth_maps/0.jpeg.geometric'))

    depth_map = read_array('Reconstruction/dense/0/stereo/depth_maps/2.jpeg.geometric.bin', ext = ".bin")
     # Resize RGB image to match the depth size for 3D color matching
    img4, rgb = convert_img('images/2.jpeg', depth_map)
    
    #normal_map = read_array('maps/normal_maps/0.jpeg.geometric.bin', ext = ".bin")
    #min_depth, max_depth = np.percentile(depth_map, [5, 95])
    
    #camera = read_cameras_text('Reconstruction/sparse/0/cameras.txt')
    
    # Extract camera intrinsic parameters automatically from the bin file
    camera_bin = read_cameras_binary('Reconstruction/sparse/0/cameras.bin')
    fx, fy, cx, cy = camera_intrinsic(camera_bin)
    
    pic1 = Image.open('images/0.jpeg')
    pic2 = Image.open('images/1.jpeg')
    pic3 = Image.open('images/2.jpeg')
    pic4 = Image.open('images/3.jpeg')
    pic5 = Image.open('images/4.jpeg')
    pic6 = Image.open('images/5.jpeg')
    
    #myClick(fx)
    result = calculate(3,5)
    res = cal(result)
    #print(result)
    # This value decides how deep the environment gets zoomed in
    Depth = 0

    # Visualize the depth map.
    #plt.figure()
    #plt.imshow(depth_map)
    #plt.title("depth map")
    #plt.show()

    # Visualize the normal map.
    #plt.figure()
    #plt.imshow(normal_map)
    #plt.title("normal map")
    
    #pcd_plot3 = o3d.io.read_point_cloud('fused.ply')

    #rgb = normal_map

    #pcd_plot = getPointCloud(depth_map, 0, fx, fy, cx, cy)
    pcd_plot2 = getPointCloud2(img4, depth_map, Depth, fx, fy, cx, cy)
    

    # Display using Open3D visualization tools
    pcd_plot2.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    #o3d.visualization.draw_geometries([pcd_plot2])
    
    
    #custom_draw_geometry_with_custom_fov(pcd_plot2, 0)
    #custom_draw_geometry_with_rotation(pcd_plot2, -90)
    
    # Execute the Graphical User Interface
    run(fx, fy, cx, cy, pcd_plot2, pic1, pic2, pic3, pic4, pic5, pic6)
    #write_array(depth_map, 'maps/depth2.pgm')


if __name__ == "__main__":
    main()
