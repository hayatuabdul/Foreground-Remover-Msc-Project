# Written by Hayatu Abdullahi

from tkinter import *
from tkinter import filedialog
from PIL import ImageTk, Image
import cv2
import os
from os.path import splitext
import ntpath
import collections
import struct
import numpy as np
import threading
import matplotlib.pyplot as plt
from PIL import Image
import open3d as o3d

from read_write_model import read_model, read_next_bytes, read_cameras_text, read_cameras_binary, read_array, write_array, read_cameras_binary, read_images_binary

# Initialize TKinter
tk = Tk()
tk.title("Foreground Remover")

# Set dimensions for the window
windowWidth = tk.winfo_reqwidth()
windowHeight = tk.winfo_reqheight()
positionRight = int(tk.winfo_screenwidth()/5 - windowWidth/5)
positionDown = int(tk.winfo_screenheight()/5 - windowHeight/2)
tk.geometry(f"1550x650+{positionRight}+{positionDown}")

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
sl2.place(x=1254,y=210)

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
    global img_rgb
    global depthselect
    global imgselect
    
    # Find depth map corresponding to RGB image selected
    direc = ('Reconstruction/dense/0/stereo/depth_maps')
    base = os.path.basename(imgselect)
    
    dp_name = base + '.geometric.bin'
    #print(dp_name)
    #print(os.path.splitext(imgselect)[0])
    
    # Merge string with directory and depth name
    dps = str(direc)
    fullpath = dps + "/" + dp_name
    #print(fullpath)

    depth_map = read_array(fullpath)
    dp_img = Image.fromarray(depth_map)
    dp_img = dp_img.convert('RGB')
    dp_img.save("depth_img.png")
    img = Image.open("depth_img.png")
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
    

    depthz = depth_map
    points = []
    colors = []
    srcPxs = []
    
    for v in range(depthz.shape[0]):
        for u in range(depthz.shape[1]):
            
            Z = depthz[v, u] 
            
            if (Z > depth):
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


tk.mainloop()
