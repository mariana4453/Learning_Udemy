import matplotlib.pyplot as plt
import numpy as np
import sys

# BASICS
# plt.plot(x, y) - visualizing relationships and data

# x = np.arange(0, 10)
# y = 2 * x
#
# plt.plot(x, y)      # basic
# plt.title('Title')

# labels
# plt.xlabel('X Axis')
# plt.ylabel('Y Label')

# setting limit for axes
# plt.xlim(0, 10)     # lower, higher
# plt.ylim(0, 15)

# needed to provide the path we image will be saved
# print(help(plt.savefig))      - save as png file
# plt.savefig('myplot.png')

# plt.show()

##################################################
# FIGURES

a = np.linspace(0, 10, 11)
b = a ** 4

x = np.arange(0, 10)
y = 2 * x

# create figure object
# fig = plt.figure()

# adding set of axes to figure
# axes = fig.add_axes([0, 0, 1, 1])    # left, bottom, width, height
# axes.plot(a, b)


# creating more than 1 figure
# fig = plt.figure()

# large figure
# axes1 = fig.add_axes([0, 0, 1, 1])    # left, bottom, width, height
# axes1.plot(a, b)

# small figure
# axes2 = fig.add_axes([0.25, 0.25, 0.25, 0.25])
# axes2.plot(a, b)

# labels
# axes1.set_xlim(0, 10)
# axes1.set_ylim(0, 8000)
# axes1.set_xlabel('A')
# axes1.set_ylabel('B')
# axes1.set_title('Power of 4')

# axes2.set_xlim(1, 2)
# axes2.set_ylim(0, 50)
# axes2.set_xlabel('A')
# axes2.set_ylabel('B')
# axes2.set_title('Zoomed in')


# FIGURE PARAMETERS
# figsize - in inches

# fig = plt.figure(dpi=100, figsize=(5, 5))
#
# axes1 = fig.add_axes([0, 0, 1, 1])
# axes1.plot(a, b)

# includes x, y axes
# fig.savefig('new_figure.png', bbox_inches='tight')
# plt.show()


# SUBPLOTS
# axes type is np.array
# fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(5, 5))
#
# axes[0][0].plot(x, y)
# axes[0][0].set_ylabel('Y label')
# axes[0][0].set_title('Title')
#
# axes[0][1].plot(x, y)
# axes[0][1].set_xlim(2, 6)
#
# axes[1][1].plot(a, b)
# axes[1][0].plot(a, b)

# fig.suptitle('Figure level', fontsize=10)

# formatting to avoid overlapping
# plt.tight_layout()

# SPACING
# https://matplotlib.org/3.2.2/api/_as_gen/matplotlib.pyplot.subplots_adjust.html
# second option to format figures
# fig.subplots_adjust(wspace=0.3, hspace=0.3)

# save
# fig.savefig('new_subplot.png', bbox_inches='tight')
# plt.show()

# PLOTTING EXAMPLES
# names = ['group_a', 'group_b', 'group_c']
# values = [1, 10, 100]
#
# plt.figure(figsize=(9, 3))
#
# plt.subplot(131)
# plt.bar(names, values)
# plt.subplot(132)
# plt.scatter(names, values)
# plt.subplot(133)
# plt.plot(names, values)
# plt.suptitle('Categorical Plotting')


# STYLING MATPLOTLIB
# https://matplotlib.org/3.2.2/api/_as_gen/matplotlib.lines.Line2D.html
x = np.linspace(0, 11, 10)


# fig = plt.figure()
# ax = fig.add_axes([0, 0, 1, 1])
# ax.plot(x, x, label='X vs X')
#
# ax.plot(x, x**2, label='X vs X^2')
# ax.legend(loc=1)

# ax.legend(loc=1) # upper right corner
# ax.legend(loc=2) # upper left corner
# ax.legend(loc=3) # lower left corner
# ax.legend(loc=4) # lower right corner
# loc=0 - sets an optimal location
# ax.legend(loc=(1.1, 0.5)) - manually set location


# COLORING AND LINES
# fig = plt.figure()
# ax = fig.add_axes([0, 0, 1, 1])

# lw/linewidth - line width
# ls/linestyle '-', '--', '-.', ':'

# b - blue, line with dots etc...
# ax.plot(x, x+4, 'b.-')
# ax.plot(x, x**2, color='#8B008B', alpha=0.5)   # hex code RGB, half-transperent
# ax.plot(x, x+7, color='red', lw=5.25)
# ax.plot(x, x*10, color='green', ls='-')

# different styles
# https://matplotlib.org/stable/gallery/index.html
# https://matplotlib.org/cheatsheets/


# CUSTOM LINE STYLE
# fig, ax = plt.subplots(figsize=(12, 6))
# lines = ax.plot(x, x**3, color='#650dbd', lw=5)
#
# lines[0].set_dashes([1, 2, 5, 2, 10, 2])
# 5 -solid points, 2 blank points, 10 solid points, 2 solid points

# all markers - https://matplotlib.org/3.2.2/api/markers_api.html
# ms/markersize
# ax.plot(x, x**2, lw=0, color='g', marker='4', ms=20)
#
# plt.show()


