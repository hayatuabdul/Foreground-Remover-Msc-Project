# Written by Hayatu Abdullahi

from tkinter import *
from tkinter import filedialog
from PIL import ImageTk, Image
import cv2
import os
from os.path import splitext
import numpy as np
import threading
import matplotlib.pyplot as plt

from Dependencies.read_write_model import read_model, read_next_bytes, read_cameras_text, read_cameras_binary, read_images_binary, read_array, write_array, qvec2rotmat
from Visualization.point_cloud1 import point_cloud
from Visualization.point_cloud2 import point_cloudb
from Visualization.ply_cloud import ply_cloud
from Transformations.matrix import matrix
from Transformations.icp import icp
from Calibration.calibrate import camera_intrinsic, convert


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
sl2 = Label(my_frame, text="Change Depth", font="italic")
#sl2.grid(row=1,column=2, sticky=W, padx="200")
sl2.place(x=1247,y=240)

# Value for slider button
value = IntVar() 

# Manually select RGB image
def rgb_img():
    global tkimage2
    global img_rgb
    global imgselect
    global img2
    global ply_file
    
    imgselect = filedialog.askopenfilename(initialdir = "Desktop", filetypes = [('Image files', '*.jpeg'),('Image files', '*.jpg'), ('Image files', '*.png')])
    if not imgselect:
        return 0
    
    img2 = Image.open(imgselect)
    print(img2)
    img2.thumbnail((560, 490))
    tkimage2 = ImageTk.PhotoImage(img2)
    
    L1 = Label(my_frame)
    L1.config(image = tkimage2)
    L1.grid(row=1, column=0)

    img3 = np.asarray(img2)
    img3 =cv2.resize(img3,(640,480))
    img4 = Image.fromarray(img3)
    img4.save("DenseDepth/pic_shoe.png")


    return imgselect, ply_file
  

# Automatically load depth map
def depthmap():
    global tkimage
    global depth_map
    global depth_map2
    global dp_img
    global dp_img2
    global img
    global img2
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
    depth_res = Image.open('HighResDepth/11.png')
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
    imgpred = np.asarray(img2)

    #depthpred = (1000 / model.predict( np.expand_dims(imgpred, axis=0)  )) / 1000
    
    
    dep = np.load("DenseDepth/depthmon_shoe.npy")
    dep = dep[0, :, :, 0]
    dep2 = np.load("DenseDepth/depthmon8.npy")
    dep2 = dep2[0, :, :, 0]
    dep3 = ('Reconstruction/dense/0/stereo/depth_maps/8.jpeg.geometric.bin')
    dep_c = np.load("DenseDepth/depthmonc.npy")
    dep_c = dep_c[:, :, 0]
    print('Depth Shape:', dep.shape)
    print('Depth Type:', type(dep))

    
    # Acquires depth map from COLMAP
    #depth_map = read_array(fullpath)

    # Acquires depth map from Deep learning model
    depth_map = dep
    depth_map2 = dep2
    #depth_map2 = read_array(dep3)
    #print('Depth Shape:', depth_map.shape)
    #print('Depth Type:', type(depth_map))
    #depth_map = dep_high_res
    dp_img = Image.fromarray(depth_map)
    dp_img = dp_img.convert('RGB')
    dp_img.save("depth_img.png")
    dp_img2 = np.asarray(dp_img)
    img = Image.open("depth_img.png")
    img = img.convert('RGB')
    img.thumbnail((560, 490))
    tkimage = ImageTk.PhotoImage(img)
    
    L2 = Label(my_frame) 
    L2.config(image = tkimage)
    L2.grid(row=1, column=1)

    return depth_map, depth_map2

    
# Update depth value when slider gets released
# Function got automated removing multiple if statements in the process
def updateValue(event):
    global depth
    
    sd = sl.get()
    
    if sd >= 1:
        # This value represents the depth for most reconstructions obtained from COLMAP
        depth = sd/50
        # This value represents the depth for most reconstructions obtained from the deep learning depth prediction model
        #depth = sd*0.0001
        print("Manipulated Depth:", depth)
        return depth
    else:
        depth = 0

        return depth


        
# Create slider for depth manipulation    
sl = Scale(my_frame, length=230,variable = value, from_ = 0, to = 1000, troughcolor="purple", orient = HORIZONTAL)
#sl.grid(row=1,column=2)
sl.place(x=1200,y=290)
sl.bind("<ButtonRelease-1>", updateValue)




#global img4, rgb, img7
#global fx, fy, cx, cy
global img_bin
global ply_file
ply_file = ('Reconstruction/dense/0/fused.ply')
global camera_bin
global camera_bin2
img_bin = read_images_binary('Reconstruction/sparse/0/images.bin')
camera_bin, camera_bin2 = read_cameras_binary('Reconstruction/sparse/0/cameras.bin')
global depth
depth = 0
    

# Use threads to respond to button clicks    
def target1():
    global imgselect
    global depth
    depth = 0
    global depth_map, depth_map2
    imgselect, ply_file = rgb_img()
    depth_map, depth_map2 = depthmap()
    
#def target2():
    #threading.Thread(target=slider).start()
def target3():
    global camera_bin
    global camera_bin2
    threading.Thread(target=convert).start()
    #camera_bin, camera_bin2 = read_cameras_binary('Reconstruction/sparse/0/cameras.bin')
    threading.Thread(target=lambda:camera_intrinsic(camera_bin)).start()
def target4():
    img4, rgb, img7 = convert(imgselect, depth_map, depth_map2)
    fx, fy, cx, cy = camera_intrinsic(camera_bin)
    pcd = point_cloud(depth_map, img4, depth, fx, fy)
    rotation_matrix, trans, Matrix, Matrix_inv, K = matrix(img_bin, fx, fy, cx, cy)
    pcd2 = point_cloudb(rotation_matrix, trans, Matrix, depth_map2, img7, depth, pcd, fx, fy, K)
    #threading.Thread(target=lambda:icp(pcd, pcd2, Matrix)).start()
    #threading.Thread(target=lambda:matrix(img_bin)).start()
def target5():
   
    fx, fy, cx, cy = camera_intrinsic(camera_bin)
    rotation_matrix, trans, Matrix, Matrix_inv, K = matrix(img_bin, fx, fy, cx, cy)
    threading.Thread(target=lambda:ply_cloud(depth, rotation_matrix, trans, Matrix, Matrix_inv, ply_file)).start()
    
   

B1 = Button(tk, text = "Choose Color Image", padx=20, pady=15, command=target1, relief="solid")
B1.config(cursor="hand2")
B1.place(x=256,y=540)

# Fixed depth map selection by removing the thread
#B2 = Button(tk, text = "Load Depth Image", padx=20, pady=15, command=depthmap, relief="solid")
#B2.config(cursor="hand2")
#B2.place(x=840,y=540)

#B3 = Button(tk, text = "Calibrate", padx=57, pady=45, command=target3, relief="solid")
#B3.config(cursor="hand2")
#B3.place(x=1255,y=70)

B4 = Button(tk, text = "Visualize", padx=60, pady=50, command=target4, relief="solid")
B4.config(cursor="hand2")
B4.place(x=1255,y=80)

#B5 = Button(tk, text = "Activate Slider", padx=50, pady=30, command=target2, relief="solid")
#B5.config(cursor="hand2")
#B5.place(x=1251,y=530)

B6 = Button(tk, text = "View Dense", padx=50, pady=30, command=target5, relief="solid")
B6.config(cursor="hand2")
B6.place(x=830,y=530)



tk.mainloop()