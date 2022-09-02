# Written by Hayatu Abdullahi


from PIL import ImageTk, Image
import cv2
from os.path import splitext
import numpy as np
import matplotlib.pyplot as plt




# Obtain camera intrinsics for the images    
def camera_intrinsic(camera):
    global fx, fy, cx, cy, depth
    depth = 0
    fx = float(((str(camera[0]).lstrip('[').rstrip(']'))))
    fy = float(((str(camera[1]).lstrip('[').rstrip(']'))))
    cx = float(((str(camera[2]).lstrip('[').rstrip(']'))))
    cy = float(((str(camera[3]).lstrip('[').rstrip(']'))))
    print('Camera Intrinsics:', fx,fy,cx,cy)
    
    return fx, fy, cx, cy

# Convert RGB images to match depth size if necessary
def convert(imgselect, depth_map, depth_map2):
    global img4
    global img7
    global rgb
    global rgb2
    
    rgb = Image.open(imgselect)
    rgb2 = np.asarray(rgb)
    dp_size = depth_map.shape
    if rgb2.size != depth_map.size:
        img =cv2.resize(rgb2,(dp_size[1],dp_size[0]))
        img4 = Image.fromarray(img)
        
        if img4.mode != 'RGB':
            img4 = img4.convert('RGB')
        
    pic2 = Image.open('images/8.jpeg')
    #pic2 = Image.open('images\8.jpeg')
    rgb3 = np.asarray(pic2)
    dp_size2 = depth_map2.shape
    if rgb3.size != depth_map2.size:
        img6 =cv2.resize(rgb3,(dp_size[1],dp_size[0]))
        img7 = Image.fromarray(img6)
        
        if img7.mode != 'RGB':
            img7 = img7.convert('RGB')
    
    print('Depth Size', depth_map.shape)
    print('Image Size',img.shape)
    #depth_map2 = depth_map.reshape(2016, 1512, 3)
    #print (depth_map2.shape)
    
    return img4, rgb, img7
    

