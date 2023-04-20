import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as maticker        # for histogram plot
from mpl_toolkits.mplot3d import Axes3D     # for 3D scatter plot
from matplotlib.figure import Figure 
import numpy as np
from tkinter import *
from tkinter.ttk import *


# # ex1
# # X axis parameter:
# xaxis = np.array([2, 8])

# # Y axis parameter:
# yaxis = np.array([4, 9])

# plt.plot(xaxis, yaxis)
# plt.show()


# # ex2
# xaxis = np.array([2, 12, 3, 9])

# # Mark each data value and customize the linestyle:
# plt.plot(xaxis, marker = "o", linestyle = "--")
# plt.show()

# # A partial list of string characters that are acceptable options for marker and linestyle:

# # “-” solid line style
# # “--” dashed line style
# # “ “ no line
# # “o” letter marker


# # ex3
# # create scatter plot graph
# # X axis values:
# x = [2,3,7,29,8,5,13,11,22,33]
# # Y axis values:
# y = [4,7,55,43,2,4,11,22,33,44]

# # Create scatter plot:
# plt.scatter(x, y)

# plt.show()


# # ex4
# # accommodate multiple datasets in a single plot
# # Create random seed:
# np.random.seed(54841)

# # Create random data:
# xdata = np.random.random([2, 8])  

# # Create two datasets from the random floats: 
# xdata1 = xdata[0, :]  
# xdata2 = xdata[1, :]  

# # Sort the data in both datasets:
# xdata1.sort()  
# xdata2.sort()

# # Create y data points:  
# ydata1 = xdata1 ** 2
# ydata2 = 1 - xdata2 ** 4

# # Plot the data:  
# plt.plot(xdata1, ydata1)  
# plt.plot(xdata2, ydata2)  

# # Set x,y lower, upper limits:  
# plt.xlim([0, 1])  
# plt.ylim([0, 1])  

# # title the graph
# plt.title("Multiple Datasets in One Plot")
# plt.show()


# # ex5
# # create complex figures that contain more 
# # than one plot. In this example, multiple
# # axes are enclosed in one figure and 
# # displayed in subplots

# # Create a Figure with 2 rows and 2 columns of subplots:
# fig, ax = plt.subplots(2, 2)

# x = np.linspace(0, 5, 100)

# # Index 4 Axes arrays in 4 subplots within 1 Figure: 
# ax[0, 0].plot(x, np.sin(x), 'g') #row=0, column=0
# ax[1, 0].plot(range(100), 'b') #row=1, column=0
# ax[0, 1].plot(x, np.cos(x), 'r') #row=0, column=1
# ax[1, 1].plot(x, np.tan(x), 'k') #row=1, column=1

# plt.show()


# # ex6
# # combine matplotlib’s histogram and subplot capabilities by 
# # creating a plot containing five bar graphs. The areas in the 
# # bar graph will be proportional to the frequency of a random 
# # variable, and the widths of each bar graph will be equal to 
# # the class interval
# # Create random variable:
# data = np.random.normal(0, 3, 800)

# # Create a Figure and multiple subplots containing Axes:
# fig, ax = plt.subplots()
# weights = np.ones_like(data) / len(data)

# # Create Histogram Axe:
# ax.hist(data, bins=5, weights=weights)
# ax.yaxis.set_major_formatter(maticker.PercentFormatter(xmax=1.0, decimals=1))

# plt.title("Histogram Plot")
# plt.show()


# # ex7
# # A phase spectrum plot lets us visualize the frequency 
# # characteristics of a signal.
# # In this advanced example, we’ll plot a phase spectrum of 
# # two signals (represented as functions) that each have different
# # frequencies
# # Generate pseudo-random numbers:
# np.random.seed(0) 

# # Sampling interval:    
# dt = 0.01 

# # Sampling Frequency:
# Fs = 1 / dt  # ex[;aom Fs] 

# # Generate noise:
# t = np.arange(0, 10, dt) 
# res = np.random.randn(len(t)) 
# r = np.exp(-t / 0.05) 

# # Convolve 2 signals (functions):
# conv_res = np.convolve(res, r)*dt
# conv_res = conv_res[:len(t)] 
# s = 0.5 * np.sin(1.5 * np.pi * t) + conv_res

# # Create the plot: 
# fig, (ax) = plt.subplots() 
# ax.plot(t, s) 
# # Function plots phase spectrum:
# ax.phase_spectrum(s, Fs = Fs)

# plt.title("Phase Spectrum Plot")
# plt.show()


# # ex8
# # 3D scatter plot
# fig = plt.figure()

# # Create 1 3D subplot:
# ax = fig.add_subplot(111, projection='3d')

# # ‘111’ is a MATlab convention used in Matplotlib
# # to create a grid with 1 row and 1 column. 
# # The first cell in the grid is the new Axes location.
# # Create x,y,z coordinates:
# x =[1,2,3,4,5,6,7,8,9,10]
# y =[11,4,2,5,13,4,14,2,4,8]
# z =[2,3,4,5,5,7,9,11,19,9]

# # Create a 3D scatter plot with x,y,z orthogonal axis, and red "o" markers:
# ax.scatter(x, y, z, c='red', marker="o")

# # Create x,y,z axis labels:
# ax.set_xlabel('X Axis')
# ax.set_ylabel('Y Axis')
# ax.set_zlabel('Z Axis')

# plt.show()


# # ex9
# # Use a Matplotlib Backend

# matplotlib.use("TkAgg")

# # OO backend (Tkinter) tkagg() function:
# from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# root = Tk()

# figure = Figure(figsize=(5, 4), dpi=100)
# plot = figure.add_subplot(1, 1, 1)

# x = [ 0.1, 0.2, 0.3, 0.4 ]
# y = [ -0.1, -0.2, -0.3, -0.4 ]
# plot.plot(x, y, color="red", marker="o",  linestyle="--")

# canvas = FigureCanvasTkAgg(figure, root)
# canvas.get_tk_widget().grid(row=0, column=0)
# root.mainloop()


# Final Tip:  matplotlib script execution creates a text output 
# in the Python console (not part of the UI plot display) that may
# include warning messages or be otherwise visually unappealing. 
# To fix this, you can add a semicolon (;) at the end of the last 
# line of code before displaying the plot. For example:

# pyplot scatter() function:
plt.scatter(x, y);

plt.show()





























































