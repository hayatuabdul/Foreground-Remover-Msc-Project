# Written by Hayatu Abdullahi

from tkinter import *
from tkinter import filedialog
from PIL import ImageTk, Image
import cv2
import collections
import struct
import numpy as np
import threading
import matplotlib.pyplot as plt
from PIL import Image
import open3d as o3d

from read_write_model import read_model, read_next_bytes, read_cameras_text, read_cameras_binary, read_array, write_array

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


# Manually select depth map
def depth_map():
    global tkimage
    global depth_map
    global dp_img
    global img_rgb
    global depthselect
    
    # Open directory and find depth files
    depthselect = filedialog.askopenfilename(initialdir = "Desktop", filetypes = [('Depth Files', '*.bin')])
    depth_map = read_array(depthselect)
    dp_img = Image.fromarray(depth_map)
    dp_img = dp_img.convert('RGB')
    dp_img.save("depth_img.png")
    img = Image.open("depth_img.png")
    img.thumbnail((560, 490))
    tkimage = ImageTk.PhotoImage(img)
    
    L2 = Label(my_frame) 
    L2.config(image = tkimage)
    L2.grid(row=1, column=1)
    
    


# Manually select RGB image
def rgb_img():
    global tkimage2
    global img_rgb
    global imgselect
    
    imgselect = filedialog.askopenfilename(initialdir = "Desktop", filetypes = [('Image files', '*.jpeg'),('Image files', '*.jpg')])
    img2 = Image.open(imgselect)
    img2.thumbnail((560, 490))
    tkimage2 = ImageTk.PhotoImage(img2)
    
    L1 = Label(my_frame)
    L1.config(image = tkimage2)
    L1.grid(row=1, column=0)

    
# Obtain camera intrinsics for the images
def camera_intrinsic(camera):
    global fx, fy, cx, cy
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
    global img4
    global rgb
    global fx, fy, cx, cy
    

    depth = depth_map
    points = []
    colors = []
    srcPxs = []
    
    for v in range(depth.shape[0]):
        for u in range(depth.shape[1]):
            
            Z = depth[v, u] 
            
            if (Z > 0):
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
    threading.Thread(target=rgb_img).start()
def target2():
    threading.Thread(target=depth_map).start()
def target3():
    threading.Thread(target=convert).start()
    camera_bin = read_cameras_binary('Reconstruction/sparse/0/cameras.bin')
    threading.Thread(target=lambda:camera_intrinsic(camera_bin)).start()
def target4():
    threading.Thread(target=point_cloud).start()



B1 = Button(tk, text = "Choose RGB Image", padx=20, pady=15, command=target1, relief="solid")
B1.config(cursor="hand2")
B1.place(x=256,y=540)

# Fixed depth map selection by removing the thread
B2 = Button(tk, text = "Choose Depth", padx=20, pady=15, command=depth_map, relief="solid")
B2.config(cursor="hand2")
B2.place(x=850,y=540)

B3 = Button(tk, text = "Calibrate", padx=57, pady=45, command=target3, relief="solid")
B3.config(cursor="hand2")
B3.place(x=1255,y=70)

B4 = Button(tk, text = "Visualize", padx=60, pady=50, command=target4, relief="solid")
B4.config(cursor="hand2")
B4.place(x=1255,y=380)


tk.mainloop()
