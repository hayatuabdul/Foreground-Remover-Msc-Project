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


# Read and manipulate depth data obtained from Colmap. The depth data can then be converted to a 3d point cloud visualized in open3d



def camera_intrinsic(camera):
    
    fx = float(((str(camera[0]).lstrip('[').rstrip(']'))))
    fy = float(((str(camera[1]).lstrip('[').rstrip(']'))))
    cx = float(((str(camera[2]).lstrip('[').rstrip(']'))))
    cy = float(((str(camera[3]).lstrip('[').rstrip(']'))))
    #print(fx,fy,cx,cy)
    return fx, fy, cx, cy


def convert_img(path, depth_map):
    
    rgb = Image.open(path)
    rgb2 = np.asarray(rgb)
    dp_size = depth_map.shape
    img =cv2.resize(rgb2,(dp_size[1],dp_size[0]))
    print('Image Size',img.shape)
    
    img4 = Image.fromarray(img)
    
 
    if img4.mode != 'RGB':
        img4 = img4.convert('RGB')
        
    return img4

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
       

def loadDepthMap2(path):
        """
        Read a depth-map
        :param filename: file name to load
        :return: image data of depth image
        """

        depth = open(path, "rb")
        if len(depth.shape) == 3:
            depth = depth[:, :, 0] * 1000.
        elif len(depth.shape) == 2:
            depth *= 1000.
        else:
            raise IOError("Invalid file: {}".format(filename))
        depth[depth > 1000. - 1e-4] = 32001
        
        return depth 
    
def loadDepthMap(path):
        """
        Read a depth-map
        :param filename: file name to load
        :return: image data of depth image
        """

        img = Image.open(path)
        # top 8 bits of depth are packed into green channel and lower 8 bits into blue
        assert len(img.getbands()) == 3
        r, g, b = img.split()
        r = np.asarray(r, np.int32)
        g = np.asarray(g, np.int32)
        b = np.asarray(b, np.int32)
        dpt = np.bitwise_or(np.left_shift(g, 8), b)
        imgdata = np.asarray(dpt, np.float32)

        return imgdata 
    
    
focalX = 1.52973780e+03
focalY = 1.00800000e+03
centerX = 7.56000000e+02
centerY = 1.06462694e-02
scalingFactor = 7
L = 87
H = 20
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
    view_pcd.set_front((-0.15, 1.3, 1.3))
    view_pcd.set_up((0, 1, 0))
    view_pcd.set_zoom(0.5)
    view_pcd.change_field_of_view(step = fov_step)
    vis.run()
    vis.destroy_window()
    
    
    
def custom_draw_geometry_with_rotation(pcd, fov_step):

    def rotate_view(vis):
        view_ctl = vis.get_view_control()
        view_ctl.set_front((0, 0, 1))
        view_ctl.set_up((0, 1, 0))
        view_ctl.set_zoom(0.8)
        view_ctl.change_field_of_view(step = fov_step)
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

    depth_map = read_array('Reconstruction/dense/0/stereo/depth_maps/11.jpeg.geometric.bin', ext = ".bin")
    normal_map = read_array('maps/normal_maps/0.jpeg.geometric.bin', ext = ".bin")
    #min_depth, max_depth = np.percentile(depth_map, [5, 95])
    
    camera = read_cameras_text('Reconstruction/sparse/0/cameras.txt')
    
    # Extract camera intrinsic parameters automatically from the bin file
    camera_bin = read_cameras_binary('Reconstruction/sparse/0/cameras.bin')
    fx, fy, cx, cy = camera_intrinsic(camera_bin)

    # Resize RGB image to match the depth size for 3D color matching
    img4 = convert_img('images/11.jpeg', depth_map)
    
    
    # This value decieds how deep the environment gets zoomed in
    Depth = 7.2

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
    #pcd_plot2.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    pcd_plot2.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    #o3d.visualization.draw_geometries([pcd_plot2])
    
    
    custom_draw_geometry_with_custom_fov(pcd_plot2, 0)
    #custom_draw_geometry_with_rotation(pcd_plot2, -90)
    
   
    
    #write_array(depth_map, 'maps/depth2.pgm')


if __name__ == "__main__":
    main()