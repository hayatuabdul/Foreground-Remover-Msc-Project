from tkinter import*
import open3d as o3d
#from colmap import*
import matplotlib.pyplot as plt


root = Tk()

root.geometry("1000x900")

e = Entry(root, width=35, borderwidth = 50)

def myClick(fx):
    myLabel = Label(root, text = fx)
    myLabel.grid() 

def myClick2(fy):
    myLabel = Label(root, text = fy)
    myLabel.grid() 

def myClick3(cx):
    myLabel = Label(root, text = cx)
    myLabel.grid() 

def myClick4(cy):
    myLabel = Label(root, text = cy)
    myLabel.grid() 

def myClick5(pcd):
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

def myClick6(res):
    plt.figure()
    plt.imshow(res)
    plt.show()


#slide = Scale(root, from_=0, to=200, orient=HORIZONTAL)
#slide.unpack
 
def run(fx, fy, cx, cy, pcd, pic1, pic2, pic3, pic4, pic5, pic6):
    myButton = Button(root, text = "View FX", padx=50, pady=50, command =lambda: myClick(fx))
    myButton2 = Button(root, text = "View FY", padx=50, pady=50, command =lambda: myClick2(fy))
    myButton3 = Button(root, text = "View CX", padx=50, pady=50, command =lambda: myClick3(cx))
    myButton4 = Button(root, text = "View CY", padx=50, pady=50, command =lambda: myClick4(cy))
    myButton5 = Button(root, text = "Visualize", padx=80, pady=80, command =lambda: myClick5(pcd))
    myButton6 = Button(root, text = "Image 1", padx=60, pady=60, command =lambda: myClick6(pic1))
    myButton7 = Button(root, text = "Image 2", padx=60, pady=60, command =lambda: myClick6(pic2))
    myButton8 = Button(root, text = "Image 3", padx=60, pady=60, command =lambda: myClick6(pic3))
    myButton9 = Button(root, text = "Image 4", padx=60, pady=60, command =lambda: myClick6(pic4))
    myButton10 = Button(root, text = "Image 5", padx=60, pady=60, command =lambda: myClick6(pic5))
    myButton11 = Button(root, text = "Image 6", padx=60, pady=60, command =lambda: myClick6(pic6))
    myButton12 = Button(root, text = "Exit", padx=110, pady=40, command =root.quit)
    myButton.grid(row=0, column=0)
    myButton2.grid(row=1, column=0)
    myButton3.grid(row=2, column=0)
    myButton4.grid(row=3, column=0)
    myButton5.grid(row=3, column=3)
    myButton6.grid(row=0, column=2)
    myButton7.grid(row=0, column=3)
    myButton8.grid(row=0, column=4)
    myButton9.grid(row=1, column=2)
    myButton10.grid(row=1, column=3)
    myButton11.grid(row=1, column=4)
    myButton12.grid(row=6, column=3)


    root.mainloop()

 

