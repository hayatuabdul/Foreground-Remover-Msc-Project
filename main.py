# Written by Hayatu Abdullahi

from tkinter import *
from tkinter import filedialog
from PIL import ImageTk, Image
import cv2
import os
from os.path import splitext
from pyntcloud import PyntCloud
import ntpath
import collections
import struct
import numpy as np
import threading
import matplotlib.pyplot as plt
import open3d as o3d
from plyfile import PlyData, PlyElement

from read_write_model import read_model, read_next_bytes, read_cameras_text, read_cameras_binary, read_images_binary, read_array, write_array, qvec2rotmat




# Initialize TKinter
tk = Tk()
tk.title("Foreground Remover")

# Set dimensions for the window
windowWidth = tk.winfo_reqwidth()
windowHeight = tk.winfo_reqheight()
positionRight = int(tk.winfo_screenwidth()/5 - windowWidth/5)
positionDown = int(tk.winfo_screenheight()/5 - windowHeight/2)
tk.geometry(f"1500x650+{positionRight}+{positionDown}")

# Initialize frame
my_frame = Frame(tk)
my_frame.grid(row=0, column=0,pady=25, padx=25)
l1 = Label(my_frame, text="Color Image", font="italic")
l1.grid(row=0, column=0)
L1 = Label(my_frame, text="RGB Image Required",height="28",width="80",bd=0.5, relief="sunken")
L1.grid(row=1, column=0, pady=10, padx=15)
l2 = Label(my_frame, text="Depth Image", font="italic")
l2.grid(row=0, column=1)
L2 = Label(my_frame, text="Depth Image Required",height="28",width="80",bd=0.5, relief="sunken")
L2.grid(row=1, column=1)
sl = Label(my_frame)
sl.grid(row=1,column=2, sticky=W, padx="200")
sl2 = Label(my_frame, text="Manipulate Depth", font="bold")
#sl2.grid(row=1,column=2, sticky=W, padx="200")
sl2.place(x=1248,y=210)

# Value for slider button
value = IntVar() 

# Manually select RGB image
def rgb_img():
    global tkimage2
    global img_rgb
    global imgselect
    
    imgselect = filedialog.askopenfilename(initialdir = "Desktop", filetypes = [('Image files', '*.jpeg'),('Image files', '*.jpg')])
    if not imgselect:
        return 0
    
    img2 = Image.open(imgselect)
    img2.thumbnail((560, 490))
    tkimage2 = ImageTk.PhotoImage(img2)
    
    L1 = Label(my_frame)
    L1.config(image = tkimage2)
    L1.grid(row=1, column=0)
    

# Automatically load depth map
def depth_map():
    global tkimage
    global depth_map
    global dp_img
    global img
    global img_rgb
    global depthselect
    global imgselect
    
    # Find depth map corresponding to RGB image selected
    direc = ('Reconstruction/dense/0/stereo/depth_maps')
    direc2 = ('midas/output')
    base = os.path.basename(imgselect)
    
    #dep_mono = Image.open('midas/output/10.png')
    #dep_mono2 = np.asarray(dep_mono)
    #print("Dep Mono:", dep_mono2)
    
    dp_name = base + '.geometric.bin'
    
    #print(dp_name)
    base2 = (os.path.splitext(imgselect)[0])
    base3 = os.path.basename(base2)
    dp_name2 = base3 + '.png'
    depth_res = Image.open('HighResDepth/Aldep/12_r.png')
    #print(dp_name2)
    
    # Merge string with directory and depth name
    dps = str(direc)
    fullpath = dps + "/" + dp_name
    #print(fullpath)
    dps2 = str(direc2)
    fullpath2 = dps2 + "/" + dp_name2
    #print(fullpath2)
    
    #fullpath3 = Image.open(fullpath2)
    #dep_mono = np.asarray(fullpath3)
    dep_high_res = np.asarray(depth_res)
    
    
    dep_mong = np.load("depthmon.npy")
    

    depth_map = read_array(fullpath)
    #depth_map = dep_high_res
    dp_img = Image.fromarray(depth_map)
    dp_img = dp_img.convert('RGB')
    dp_img.save("depth_img.png")
    img = Image.open("depth_img.png")
    img = img.convert('RGB')
    img.thumbnail((560, 490))
    tkimage = ImageTk.PhotoImage(img)
    
    L2 = Label(my_frame) 
    L2.config(image = tkimage)
    L2.grid(row=1, column=1)
    
# Update depth value when slider gets released
# Function got automated removing multiple if statements in the process
def updateValue(event):
    global depth
    
    sd = sl.get()
    
    if sd >= 1:
        depth = sd/50
        #depth = sd/0.01
        print("Maipulated Depth:", depth)
        return depth
    else:
        depth = 0

        return depth


        
# Create slider for depth manipulation    
def slider():
    global sl

    sl = Scale(my_frame, length=230,variable = value, from_ = 0, to = 1000, troughcolor="purple", orient = HORIZONTAL)
    #sl.grid(row=1,column=2)
    sl.place(x=1200,y=235)
    sl.bind("<ButtonRelease-1>", updateValue)

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
def convert():
    global imgselect
    global depth_map
    global img4
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
        
    print('Depth Size', depth_map.shape)
    print('Image Size',img.shape)
    
    return img4, rgb
    
# Generate and visualize a 3D point cloud of the environment from the image perspective   
def point_cloud():
    global depth_map
    global depth
    global img4
    global rgb
    global fx, fy, cx, cy
    global pcd
    global colors
    

    depthz = depth_map
    rgb3 = np.asarray(rgb)
    #rgb3 = np.asarray(rgb)
    #points = []
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
    
    for i in np.arange(height):
        for j in np.arange(width):
            colors.append(rgb.getpixel((j, i)))
            

    #colors = np.hstack((rgb3))
    

    #cox = colors[1:8 ,0:1]
    #coy = colors[: ,1:2]
    #coz = colors[: ,2:3]
    
    #print('RGB color:', cox)
    print(u)
    x = (u - centeru) * depthz / fx
    y = (v - centerv) * depthz / fy
    z = depthz
    #u = np.int
    print(u)
    print(v)
    
    
    #colors = colors[0:3]
    
    x = np.reshape(x, (width * height, 1)).astype(float)
    y = np.reshape(y, (width * height, 1)).astype(float)
    z = np.reshape(z, (width * height, 1)).astype(float)
    #colors = np.reshape(colors, (width * height, 1)).astype(np.float)
    points = np.concatenate((x, y, z), axis=1)
    points =  np.asarray(points)

    #points = np.asarray(points)
    colors = np.asarray(colors)
    print('Colors Size:', len(colors))
    print('Point Size:', len(points))

    
    pcd = o3d.geometry.PointCloud()

    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors/255)
    pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    o3d.visualization.draw_geometries([pcd])
    
    
# Generate and visualize Dense model outputted by Colmap. Also access and manipulate depth pixels         
def ply_cloud():
    global ply_file
    global point_cloud_in_numpy
    global colors_cloud_in_numpy
    global pcd2
    global img_bin
    global camera_bin2
    global pointy
    global coly

    ply_file = ('Reconstruction/dense/0/fused.ply')
    

    images = img_bin
    #cameras = camera_bin2
    im1 = images[13]
    print(im1.qvec)
    #print(im1)
    #cam1 = cameras[1]
    #print(cam1)
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
    coly7 = coly
    colyx = coly7[:, 0:1]
    colyy = coly7[:, 1:2]
    colyz = coly7[:, 2:3]
    
    # Remove brackets and quotations marks
    CX0 = colyx.flatten()
    CY0 = colyy.flatten()
    CZ0 = colyz.flatten()

    print(colyx)
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
    print('B4',pointyx)
    print('Aff',X)
    sizex = len(X)
    print(sizex)
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
    print(colors)
    pcd = o3d.geometry.PointCloud()

    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors/255)
    pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
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

def point_cloud2():
    global depth_map
    global depth
    global img4
    global rgb
    global fx, fy, cx, cy
    global color
    

    depthz = depth_map
    points = []
    colors = []
    
    rgb3 = np.asarray(rgb)
    points = []
    
    colors = np.hstack((rgb3))
    colors = colors[:, 0:3]
    print('RGB color:', colors)
    
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors/255)
    pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)
    view_pcd = vis.get_view_control()
    #view_pcd.set_front((0.1, 0.8, 1.3))
    view_pcd.set_front((-0.15, 1.3, 2.2))
    view_pcd.set_up((0, 1, 0))
    view_pcd.set_zoom(0.45)
    
    #view_pcd.change_field_of_view(step = fov_step)
    vis.run()
    vis.destroy_window() 
        
    
    
def save_point_cloud():
    global rgb
    global pcd
    global depth_map
    #global colors
    

    

    
    
def point_cloud3():
    global depth_map
    global fx, fy, cx, cy
    """Transform a depth image into a point cloud with one point for each
    pixel in the image, using the camera transform for a camera
    centred at cx, cy with field of view fx, fy.

    depth is a 2-D ndarray with shape (rows, cols) containing
    depths from 1 to 254 inclusive. The result is a 3-D array with
    shape (rows, cols, 3). Pixels with invalid depth in the input have
    NaN for the z-coordinate in the result.

    """
    points = []
    depth = depth_map
    rows, cols = depth.shape
    c, r = np.meshgrid(np.arange(cols), np.arange(rows), sparse=True)
    valid = (depth > 0) & (depth < 255)
    z = np.where(valid, depth / 256.0, np.nan)
    x = np.where(valid, z * (c - cx) / fx, 0)
    y = np.where(valid, z * (r - cy) / fy, 0)
    #points.append((x, y, z))
    pcd3 = np.dstack((x, y, z))
    #points = np.asarray(points)
    #plt.figure()
    #plt.imshow(pcd3)
    #plt.show()
    plt.rcParams["figure.figsize"] = [7.00, 3.50]
    plt.rcParams["figure.autolayout"] = True
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    #data = np.random.random(size=(3, 3, 3))
    z, x, y = pcd3.nonzero()
    ax.scatter(x, y, z, c=z, alpha=1)
    plt.show()
    
    #pcd3 = o3d.geometry.PointCloud()
    #pcd3.points = o3d.utility.Vector3dVector(points)
    #pcd3.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

    #3d.visualization.draw_geometries([pcd3])


# Use threads to respond to button clicks    
def target1():
    global img_bin
    global camera_bin2
    threading.Thread(target=rgb_img).start()
    img_bin = read_images_binary('Reconstruction/sparse/0/images.bin')
    camera_bin, camera_bin2 = read_cameras_binary('Reconstruction/sparse/0/cameras.bin')
    #qvec2rotmat()
def target2():
    threading.Thread(target=slider).start()
def target3():
    global camera_bin
    global camera_bin2
    threading.Thread(target=convert).start()
    camera_bin, camera_bin2 = read_cameras_binary('Reconstruction/sparse/0/cameras.bin')
    threading.Thread(target=lambda:camera_intrinsic(camera_bin)).start()
def target4():
    threading.Thread(target=point_cloud).start()
    #threading.Thread(target=point_cloud2).start()
    #threading.Thread(target=convert_depth_pixel_to_point).start()
def target5():
    
    threading.Thread(target=ply_cloud).start()
    #threading.Thread(target=save_point_cloud).start()
    

B1 = Button(tk, text = "Choose Color Image", padx=20, pady=15, command=target1, relief="solid")
B1.config(cursor="hand2")
B1.place(x=256,y=540)

# Fixed depth map selection by removing the thread
B2 = Button(tk, text = "Load Depth Image", padx=20, pady=15, command=depth_map, relief="solid")
B2.config(cursor="hand2")
B2.place(x=840,y=540)

B3 = Button(tk, text = "Calibrate", padx=57, pady=45, command=target3, relief="solid")
B3.config(cursor="hand2")
B3.place(x=1255,y=70)

B4 = Button(tk, text = "Visualize", padx=60, pady=50, command=target4, relief="solid")
B4.config(cursor="hand2")
B4.place(x=1255,y=380)

B5 = Button(tk, text = "Activate Slider", padx=50, pady=30, command=target2, relief="solid")
B5.config(cursor="hand2")
B5.place(x=1251,y=530)

B6 = Button(tk, text = "View Dense", padx=50, pady=30, command=target5, relief="solid")
B6.config(cursor="hand2")
B6.place(x=1051,y=530)



tk.mainloop()
