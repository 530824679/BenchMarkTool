# -*- coding:utf-8 -*-

import os
import sys
import cv2

from cv_bridge import CvBridge, CvBridgeError

import time
from sensor_msgs import point_cloud2
from sensor_msgs.msg import CompressedImage as CompressedImage, PointCloud2
import sensor_msgs.point_cloud2 as pc2

from Tkinter import *
import tkFileDialog as dlg

from PIL import Image, ImageTk
import message_filters
import random
import rospy
import numpy as np

#-----------------------------
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
#-----------------------------



class Application(Frame):
    def __init__(self, master=None):
        Frame.__init__(self, master, bg='black', width=1280, height=1080)

        self.object_list = []

        self.width = 1080
        self.height = 960

        self.R_cam_to_rect = np.array([[9.999239e-01, 9.837760e-03, -7.445048e-03, 0],
                                  [-9.869795e-03, 9.999421e-01, -4.278459e-03, 0],
                                  [7.402527e-03, 4.351614e-03, 9.999631e-01, 0],
                                  [0, 0, 0, 1]])

        self.Tr_velo_to_cam = np.array([[7.533745e-03, -9.999714e-01, -6.166020e-04, -4.069766e-03],
                                   [1.480249e-02, 7.280733e-04, -9.998902e-01, -7.631618e-02],
                                   [9.998621e-01, 7.523790e-03, 1.480755e-02, -2.717806e-01],
                                   [0, 0, 0, 1]])

        #self.extrinsic_matrix = np.dot(self.R_cam_to_rect, self.Tr_velo_to_cam)
        self.extrinsic_matrix = np.array([[0.910985, 0.412217, 0.013533, 0.047819],
                                     [-0.019837, 0.076565, -0.996867, 0.174158],
                                     [-0.411962, 0.907863, 0.077927, -0.539126],
                                     [0.0, 0.0, 0.0, 1.0]])

        self.bridge = CvBridge()
        self.image_topic = ''
        self.lidar_topic = ''

        self.timestamp = 0
        self.timeelapsed = 0

        self.object_index = 0
        self.attribute_index = 0

        self.save_dir = None

        self.pack(expand=YES, fill=BOTH)
        self.createMainFrame()
        self.createWidgets()

    def callback(self, image, points):
        print("image timestamp: %f, points timestamp: %f", image.header.stamp, points.header.stamp)
        self.timestamp = image.header.stamp
        self.timeelapsed += 0.01

        self.updateImage(image)
        self.updateBEV(points)

        self.refreshFigureCache()
        self.updateFigureAssess()
        self.updateFigureAttribute()

        self.updateTime()

    def updateTime(self):
        self.timestamp_entry.delete(0, 'end')
        self.elapsed_entry.delete(0, 'end')
        self.timestamp_entry.insert(0, str(self.timestamp))
        self.elapsed_entry.insert(0, str(self.timeelapsed))

    def updateImage(self, image):
        try:
            cv_image = self.bridge.compressed_imgmsg_to_cv2(image, "bgr8")
        except CvBridgeError as e:
            print(e)

        image_arr = Image.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
        image_front = image_arr.resize((460, 350), Image.ANTIALIAS)
        front = ImageTk.PhotoImage(image_front)
        self.panel_image.configure(image=front)
        self.panel_image.image = front

    def updateBEV(self, points):
        point_cloud = pc2.read_points(points)
        points_array = np.array(list(point_cloud))[:, 0:4]

        # transform = np.dot(self.extrinsic_matrix, points_array.T)
        # cloud = transform[0:3].T
        #
        # # Only keep points in front of camera (positive z)
        # point_filter = (cloud[:, 2] > 0)
        # cloud = cloud[point_filter]

        image_bev = self.pointCloud2birdeyes(points_array)
        bev_color = cv2.applyColorMap(cv2.convertScaleAbs(image_bev, alpha=15), cv2.COLORMAP_JET)
        image_bev = Image.fromarray(bev_color)
        image_bev = image_bev.resize((460, 350), Image.ANTIALIAS)
        bev = ImageTk.PhotoImage(image_bev)
        self.panel_lidar.configure(image=bev)
        self.panel_lidar.image = bev

    def refreshFigureCache(self):
        plt.clf()
        self.fig.cla()
        self.fig_1.cla()
        self.fig_2.cla()
        self.canvas_assess.flush_events()
        self.canvas_attribute.flush_events()

    def updateFigureAssess(self):
        self.create_assess_matplotlib()

        x = np.array([1, 2, 3, 4])
        y = []
        for i in range(4):
            y.append(random.randint(70, 100))
        y = np.array(y)

        self.fig.bar(x, y, width=0.35, align='center', color=['g', 'g', 'g', 'y'], edgecolor='red', alpha=0.8)
        for a, b, label in zip(x, y, y):
            self.fig.text(a, b, label, ha='center', va='bottom', fontsize=8)

        self.canvas_assess.draw()

    def updateFigureAttribute(self):
        self.create_attribute_matplotlib()

        x = np.array([1, 2, 3, 4, 5])
        y_real = []
        y_predict = []
        for i in range(5):
            y_real.append(random.randint(0, 30))
            y_predict.append(random.randint(0, 30))

        y_real = np.array(y_real)
        y_predict = np.array(y_predict)
        self.fig_1.bar(x, y_real, width=0.3, align='center', color='green', edgecolor='red', alpha=0.8)
        self.fig_1.bar(x + 0.3, y_predict, width=0.3, align='center', color='yellow', edgecolor='red', alpha=0.8)

        for a, b, label in zip(x, y_real, y_real):
            self.fig_1.text(a, b, label, ha='center', va='bottom', fontsize=8)
        for a, b, label in zip(x, y_predict, y_predict):
            self.fig_1.text(a + 0.3, b, label, ha='center', va='bottom', fontsize=8)

        self.fig_2.bar(x, y_real, width=0.3, align='center', color='green', edgecolor='red', alpha=0.8)
        self.fig_2.bar(x + 0.3, y_predict, width=0.3, align='center', color='yellow', edgecolor='red', alpha=0.8)

        for a, b, label in zip(x, y_real, y_real):
            self.fig_2.text(a, b, label, ha='center', va='bottom', fontsize=8)
        for a, b, label in zip(x, y_predict, y_predict):
            self.fig_2.text(a + 0.3, b, label, ha='center', va='bottom', fontsize=8)

        self.canvas_attribute.draw()

    def createMainFrame(self):
        self.master.title('Benchmark_Tool')
        self.master.bg = 'black'
        self.master.resizable(False, False)

        # pack center in the windows
        winddows_width, windows_height=self.master.maxsize()
        self.x = int((winddows_width / 2) - (self.width / 2))
        self.y = int((windows_height / 2) - (self.height / 2))
        self.master.geometry('{}x{}+{}+{}'.format(self.width, self.height, self.x, self.y))

    def createWidgets(self):
        # create tool_title label frame
        self.frame_title = Frame(self, bg='black')
        self.title_label = Label(self.frame_title, text="Benchmark system", font=('微软雅黑', 20), fg="white", bg='black')
        self.title_label.pack()
        self.frame_title.pack(side=TOP, fill='x', pady=2)

        # create directory label frame
        self.frame_control = Frame(self, bg='black')

        self.lf_dir = LabelFrame(self.frame_control, font=('微软雅黑', 14), bg='black')
        self.lf_dir.pack(fill=BOTH, anchor=W)

        # create diectory entry bar
        self.lidar_label_topic = Label(self.lf_dir, text=" 雷达主题：", bg='black', fg='white', font=('微软雅黑', 10))
        self.lidar_label_topic.grid(row=0, padx=10, pady=10)

        self.lidar_entry_topic = Entry(self.lf_dir, font=('微软雅黑', 14), width='33', fg='#FF4081')
        self.lidar_entry_topic.insert(0, "/velodyne_points")
        self.lidar_entry_topic.grid(row=0, column=1, rowspan=1, columnspan=4)

        self.camera_label_topic = Label(self.lf_dir, text="    相机主题：", bg='black', fg='white', font=('微软雅黑', 10))
        self.camera_label_topic.grid(row=0, column=5, rowspan=1, columnspan=1)

        self.camera_entry_topic = Entry(self.lf_dir, font=('微软雅黑', 14), width='34', fg='#FF4081')
        self.camera_entry_topic.insert(0, "/gmsl_camera/port_0/cam_0/image_raw/compressed")
        self.camera_entry_topic.grid(row=0, column=6, rowspan=1, columnspan=4)

        # create open button
        self.subscribe_btn_topic = Button(self.lf_dir, text='Subscribe', bg='#22C9C9', fg='white', font=('微软雅黑', 10), command=self.subscribe_topic, width=10)
        self.subscribe_btn_topic.grid(row=0, column=12, columnspan=2, padx=15, pady=10)

        self.dst_label_result = Label(self.lf_dir, text=" 结果目录：", bg='black', fg='white', font=('微软雅黑', 10))
        self.dst_label_result.grid(row=1)

        self.save_entry_result = Entry(self.lf_dir, font=('微软雅黑', 14), width='75', fg='#FF4081')
        self.save_entry_result.grid(row=1, column=1, rowspan=1, columnspan=10)

        # create save button
        self.save_btn_dir = Button(self.lf_dir, text='Directory', bg='#22C9C9', fg='white', font=('微软雅黑', 10), command=self.saveDirectory, width=10)
        self.save_btn_dir.grid(row=1, column=12, columnspan=2, padx=15, pady=10)

        self.frame_control.pack(side=TOP, padx=10, fill="x")

        # visualization
        self.frame_show = Frame(self, bg='black')
        image = Image.open('./test')
        image = image.resize((460, 360), Image.ANTIALIAS)
        initIamge = ImageTk.PhotoImage(image)
        self.panel_image = Label(self.frame_show, image=initIamge)
        self.panel_image.image = initIamge
        self.panel_image.grid(row=0, column=0, padx=10, pady=10)

        image = image.resize((460, 360), Image.ANTIALIAS)
        initLidar = ImageTk.PhotoImage(image)
        self.panel_lidar = Label(self.frame_show, image=initLidar)
        self.panel_lidar.image = initLidar
        self.panel_lidar.grid(row=1, column=0, padx=10, sticky=S)

        self.canvas_attribute = Canvas()
        self.figure_attribute = plt.figure(figsize=(4, 3.6), dpi=100, facecolor="white", edgecolor='green')
        self.create_attribute_matplotlib()
        self.canvas_attribute = FigureCanvasTkAgg(self.figure_attribute, self.frame_show)
        self.canvas_attribute.show()
        self.canvas_attribute.get_tk_widget().grid(row=0, column=1, pady=10)

        self.canvas_assess = Canvas()
        self.figure_assess = plt.figure(figsize=(4, 3.6), dpi=100, facecolor="white", edgecolor='green')
        self.create_assess_matplotlib()
        self.canvas_assess = FigureCanvasTkAgg(self.figure_assess, self.frame_show)
        self.canvas_assess.show()
        self.canvas_assess.get_tk_widget().grid(row=1, column=1, sticky=S)

        # create object list box
        self.lf_objectlist = LabelFrame(self.frame_show, text='object list', font=('微软雅黑', 14) , bg='black', fg='#EF5367')
        self.lf_objectlist.grid(row=0, column=2, padx=20, sticky=E+W+N)

        self.sc_object = Scrollbar(self.lf_objectlist)
        self.sc_object.pack(side=RIGHT, fill=Y)

        objectlistvar = StringVar(value="")
        self.objectlistbox = Listbox(self.lf_objectlist, listvariable=objectlistvar, height=17, yscrollcommand=self.sc_object.set)
        self.objectlistbox.configure(selectmode="single")
        self.objectlistbox.pack(side=TOP, fill=X)
        self.objectlistbox.bind('<<ListboxSelect>>', self.objectlistbox_selected)
        self.sc_object.config(command=self.objectlistbox.yview)

        # create attribute object
        self.lf_attributelistbox = LabelFrame(self.frame_show, text='attribute list', font=('微软雅黑', 14) , bg='black', fg='#EF5367')
        self.lf_attributelistbox.grid(row=1, column=2, padx=20, sticky=E+W+N)

        self.sc_attribute = Scrollbar(self.lf_attributelistbox)
        self.sc_attribute.pack(side=RIGHT, fill=Y)

        attributelistvar = StringVar("")
        self.attributelistbox = Listbox(self.lf_attributelistbox, listvariable=attributelistvar, height=15, yscrollcommand=self.sc_attribute.set)
        self.attributelistbox.configure(selectmode="single")
        self.attributelistbox.pack(side=TOP, fill=X)
        self.attributelistbox.bind('<<ListboxSelect>>', self.attributelistbox_selected)
        self.sc_attribute.config(command=self.attributelistbox.yview)

        # create save button
        self.save_btn_dir = Button(self.frame_show, bg='#22C9C9', fg='white', font=('微软雅黑', 10), text='Analysis', command=self.saveResult, width=14, height=1)
        self.save_btn_dir.grid(row=1, column=2, ipadx=10, sticky=S)

        self.frame_show.pack(side=TOP, pady=1, fill="x")

        # Timestamp
        self.frame_time = Frame(self, bg='black')
        self.lf_time = LabelFrame(self.frame_time,font=('微软雅黑', 14), bg='black', fg='white', borderwidth=0)

        self.timestamp_label = Label(self.lf_time, text="ROS Time：", bg='black', fg='white', font=('微软雅黑', 10))
        self.timestamp_label.grid(row=0, column=0)

        self.timestamp_entry = Entry(self.lf_time, font=('微软雅黑', 12), highlightbackground='black', bg='black', width='40', fg='white')
        self.timestamp_entry.insert(0, "")
        self.timestamp_entry.grid(row=0, column=1, rowspan=1, columnspan=4)

        self.elapsed_label = Label(self.lf_time, text="      ROS Elapsed：", bg='black', fg='white', font=('微软雅黑', 10))
        self.elapsed_label.grid(row=0, column=6)

        self.elapsed_entry = Entry(self.lf_time, font=('微软雅黑', 12), highlightthickness=0, bg='black', width='40', fg='white')
        self.elapsed_entry.insert(0, "")
        self.elapsed_entry.grid(row=0, column=7, rowspan=1, columnspan=4)

        self.fps_label = Label(self.lf_time, text="                  FPS：", bg='black', fg='white', font=('微软雅黑', 10))
        self.fps_label.grid(row=0, column=15)

        self.fps_entry = Entry(self.lf_time, font=('微软雅黑', 12), highlightbackground='black', width= 3, bg='black', fg='white')
        self.fps_entry.insert(0, "10")
        self.fps_entry.grid(row=0, column=16, rowspan=1, columnspan=1)

        self.lf_time.pack(fill=BOTH, anchor=W, padx=10)
        self.frame_time.pack(side=TOP, expand=YES, fill=X)

    def create_attribute_matplotlib(self):
        xticks = [1, 2, 3, 4, 5]
        transverse_ticks = [5, 10, 15, 20, 25, 30]
        longitudinal_ticks = [10, 20, 30, 40, 50, 60]

        green_patch = mpatches.Patch(color="green")
        yellow_patch = mpatches.Patch(color="yellow")
        self.figure_attribute.legend(handles=[green_patch, yellow_patch], labels=["Real value", "Predict value"], fontsize=6, loc='upper right')

        self.fig_1 = self.figure_attribute.add_subplot(211)
        self.fig_1.set_xlim([0, 6])
        self.fig_1.set_ylim([0, 30])

        self.fig_1.set_xticks(xticks)
        self.fig_1.set_xticklabels(["ID:1", "ID:2", "ID:3", "ID:4", "ID:5"], rotation=0, fontsize=8)

        self.fig_1.set_yticks(transverse_ticks)
        self.fig_1.set_yticklabels(transverse_ticks, rotation=0, fontsize=8)

        self.fig_1.set_ylabel("Lateral(m)", fontsize=10)
        self.fig_1.set_title("Distance Comparison", loc='center', color='red')

        self.fig_2 = self.figure_attribute.add_subplot(212)
        self.fig_2.set_xlim([0, 6])
        self.fig_2.set_ylim([0, 60])

        self.fig_2.set_xticks(xticks)
        self.fig_2.set_xticklabels(["ID:1", "ID:2", "ID:3", "ID:4", "ID:5"], rotation=0, fontsize=8)

        self.fig_2.set_yticks(longitudinal_ticks)
        self.fig_2.set_yticklabels(longitudinal_ticks, rotation=0, fontsize=8)

        self.fig_2.set_xlabel("Target ID", fontsize=10)
        self.fig_2.set_ylabel("Longitudinal(m)", fontsize=10)

    def create_assess_matplotlib(self):
        self.fig = self.figure_assess.add_subplot(1, 1, 1)
        self.fig.set_xlim([0, 5])
        self.fig.set_ylim([0, 100])

        xticks = [1, 2, 3, 4]
        yticks = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

        self.fig.grid(linestyle='--', alpha=0.8)

        self.fig.set_xticks(xticks)
        self.fig.set_xticklabels(["0~20m", "20~40m", "40~60m", "Longitudinal"], rotation=0, fontsize = 8)

        self.fig.set_yticks(yticks)
        self.fig.set_yticklabels(yticks, rotation=0, fontsize = 8)

        green_patch = mpatches.Patch(color="green")
        yellow_patch = mpatches.Patch(color="yellow")
        self.figure_assess.legend(handles=[green_patch, yellow_patch], labels=["Lateral", "Longitudinal"], fontsize=6, loc='upper right')

        self.fig.set_xlabel("Distance(m)", fontsize = 10)
        self.fig.set_ylabel("Precise(%)", fontsize = 10)
        self.fig.set_title("Distance Assess", loc='center', color='red')

    def subscribe_topic(self):
        self.lidar_topic = self.lidar_entry_topic.get()
        self.image_topic = self.camera_entry_topic.get()

        if (self.lidar_topic == '') or (self.image_topic == ''):
            print("Please input the topic of lidar and camera.")
            return

        self.image_sub = message_filters.Subscriber(self.image_topic, CompressedImage)
        self.points_sub = message_filters.Subscriber(self.lidar_topic, PointCloud2)
        ts = message_filters.ApproximateTimeSynchronizer([self.image_sub, self.points_sub], 100, 100)
        ts.registerCallback(self.callback)

    def saveResult(self):
        pass

    def saveDirectory(self):
        self.save_dir = dlg.askdirectory()

        if self.save_dir:
            if not os.path.exists(self.save_dir):
                try:
                    os.mkdir(self.save_dir)
                    Tk.messagebox.showinfo('create annotation directory: \n' + self.save_dir)
                except:
                    pass

            self.save_entry_result.delete(0, 'end')
            self.save_entry_result.insert(0, self.save_dir)

    def objectlistbox_selected(self):
        for i in self.objectlistbox.curselection():
            self.object_index = self.objectlistbox.get(i)

    def attributelistbox_selected(self):
        for i in self.attributelistbox.curselection():
            self.attribute_index = self.attributelistbox.get(i)

    def pointCloud2birdeyes(self, points, res=0.1, side_range=(-20, 20), forward_range=(0, 60), height_range=(-2, 2)):
        """ Creates an 2D birds eye view representation of the point cloud data.

        Args:
            points:     (numpy array)
                        N rows of points data
                        Each point should be specified by at least 3 elements x,y,z
            res:        (float)
                        Desired resolution in metres to use. Each output pixel will
                        represent an square region res x res in size.
            side_range: (tuple of two floats)
                        (-left, right) in metres
                        left and right limits of rectangle to look at.
            forward_range:  (tuple of two floats)
                        (-behind, front) in metres
                        back and front limits of rectangle to look at.
            height_range: (tuple of two floats)
                        (min, max) heights (in metres) relative to the origin.
                        All height values will be clipped to this min and max value,
                        such that anything below min will be truncated to min, and
                        the same for values above max.
        Returns:
            2D numpy array representing an image of the birds eye view.
        """
        # Extract the points for each axis
        x_points = points[:, 0]
        y_points = points[:, 1]
        z_points = points[:, 2]

        # Filter - To return only indices of points within desired cube
        # Three filters for: Front-to-back, side-to-side, and height ranges
        # Note left side is positive y axis in LIDAR coordinates
        f_filter = np.logical_and((x_points > forward_range[0]), (x_points < forward_range[1]))
        s_filter = np.logical_and((y_points > -side_range[1]), (y_points < -side_range[0]))
        filter = np.logical_and(f_filter, s_filter)
        indices = np.argwhere(filter).flatten()

        # Keeperrs
        x_points = x_points[indices]
        y_points = y_points[indices]
        z_points = z_points[indices]

        # Convert to pixel position values - Based on resolution
        x_img = (-y_points / res).astype(np.int32)  # x axis is -y in Lidar
        y_img = (-x_points / res).astype(np.int32)  # y axis is -x in Lidar

        # Shift pixels to have minimun be(0, 0)
        x_img -= int(np.floor(side_range[0] / res))
        y_img += int(np.ceil(forward_range[1] / res))

        # Clip height values - to between min and max heights
        pixel_values = np.clip(z_points, height_range[0], height_range[1])

        # Rescale the height values - to be between the range 0~255
        pixel_values = self.scale_to_255(pixel_values, min=height_range[0], max=height_range[1])

        # Initialize empty array - of the dimensions we want
        x_max = 1 + int((side_range[1] - side_range[0]) / res)
        y_max = 1 + int((forward_range[1] - forward_range[0]) / res)
        image_arr = np.zeros([y_max, x_max], dtype=np.uint8)

        # Fill pixel values in image array
        image_arr[y_img, x_img] = pixel_values

        return image_arr    

    def scale_to_255(self, value, min, max, dtype=np.uint8):
        """ Scales an array of values from specified min, max range to 0-255
                Optionally specify the data type of the output (default is uint8)
        """
        return (((value - min) / float(max - min)) * 255).astype(dtype)

if __name__ == '__main__':
    rospy.init_node('pt_to_birdeye_node')

    app = Application()

    app.mainloop()