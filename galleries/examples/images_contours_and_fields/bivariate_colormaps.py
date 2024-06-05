"""
===================
Bivariate colormaps
===================
This demo shows how to use bivariate colormaps using
`matplotlib.axes.Axes.imshow`,
`matplotlib.axes.Axes.pcolormesh`,
and `matplotlib.collections.PatchCollection`.


Combining two images using a bivariate colormap
-----------------------------------------------

Using `matplotlib.axes.Axes.imshow`
"""
import matplotlib.pyplot as plt
import numpy as np

import matplotlib

im_A = np.arange(500)[np.newaxis, :]*np.ones((100, 500))
im_B = np.arange(100)[:, np.newaxis]*np.ones((100, 500))

im_A[:, :] = np.sin(im_A**0.5)**2
im_B[:, :] = np.sin(im_B**0.5)**2

fig, axes = plt.subplots(3, 1, figsize=(8, 6))

cm = axes[0].imshow(im_A, cmap='grey', vmin=0, vmax=1)
cbar = fig.colorbar(cm)
axes[0].set_title('A')

cm = axes[1].imshow(im_B, cmap='grey', vmin=0, vmax=1)
cbar = fig.colorbar(cm)
axes[1].set_title('B')

cm = axes[2].imshow((im_A, im_B),
                    cmap='BiOrangeBlue',
                    vmin=(0, 0), vmax=(1, 1))
cbar = fig.colorbar_2D(cm)
cbar.set_ylabel('A')
cbar.set_xlabel('B')
for ax in axes:
    ax.set_xticks([])
    ax.set_yticks([])

# %%
#
# Visualizing complex numbers
# ---------------------------
#
# Using `matplotlib.axes.Axes.pcolormesh`.
#
# Example using bivariate colormaps with `matplotlib.axes.Axes.pcolormesh` to visualize
# the mandelbrot set in the complex number plane. `matplotlib.axes.Axes.pcolor`
# can be used in the same way.

x = np.linspace(-1.5, 0.5, 200)
y = np.linspace(-1, 1, 200)
xx, yy = np.meshgrid(x, y)
c = xx+1j*yy
z = c
for i in range(7):
    z = z**2+c

fig, ax = plt.subplots(1, 1, figsize=(10, 6))

cm = ax.pcolormesh(xx, yy, (z.imag, z.real), cmap='BiCone', vmin=-1, vmax=1)
ax.set_xlabel('Re{$c$}')
ax.set_ylabel('Im{$c$}')
ax.set_title('Mandelbrot $z_{7}$\n $z_i = z_{i-1}^2+c$')

cmap = fig.colorbar_2D(cm)
cmap.set_ylabel('Im{$z$}')
cmap.set_xlabel('Re{$z$}')

fig.show()

# %%
#
# Making a bivariate cloropleth map
# ---------------------------------
#
# Uing `matplotlib.collections.PatchCollection`.
# This example shows a map of the united states.
#
# Data from https://www.bea.gov/data/gdp/gdp-state

data = '''SD 79 5.3 [32,38,33,42,33,44,44,44,45,38,45,36,41,37]
ND 95 2.0 [33,44,33,51,43,50,44,44]
AL 59 6.6 [62,10,61,11,66,12,64,20,59,20,59,13,60,10]
AZ 68 6.9 [12,23,14,27,23,25,21,13,17,13,11,18]
AR 57 6.2 [56,22,55,22,56,23,48,23,48,20,48,16,49,15,54,15,55,20]
CA 100 6.1 [0,42,7,40,5,33,12,23,11,18,7,18,6,18,1,24,0,33]
CO 89 5.9 [35,24,33,24,23,25,24,34,32,33,35,33,35,31]
CT 94 6.5 [84,40,82,39,82,37,85,38]
DE 91 3.8 [79,33,80,31,81,31,81,32,80,33,79,33,80,34]
FL 70 9.8 [75,0,77,1,77,4,73,12,66,11,66,12,61,11,62,10,66,9,69,10,71,5]
ID 60 7.1 [22,42,21,37,16,38,11,39,13,47,15,54,16,53,16,51,17,45]
IA 78 4.4 [52,32,46,32,45,36,45,38,53,38,54,36,53,31]
KS 78 8.0 [35,31,35,24,48,24,48,29,46,30]
LA 68 6.0 [58,9,57,11,53,11,54,15,49,15,49,8,55,7]
ME 65 6.2 [85,43,90,48,87,52,85,53,84,47]
MD 83 6.7 [76,32,73,31,73,32,79,33,80,31,81,31,81,30,80,29,78,33,79,30]
MI 66 5.9 [66,36,67,39,65,44,62,44,64,45,57,47,54,45,58,42,62,44,60,41,60,35]
MN 82 5.3 [51,45,55,48,47,51,47,50,43,50,44,44,45,38,53,38]
MS 49 4.6 [59,13,60,10,58,9,57,11,53,11,54,15,55,20,59,20]
MO 68 6.4 [48,29,48,24,48,23,56,23,55,22,56,22,57,23,57,24,53,31,52,32,46,32]
MT 62 5.2 [16,53,16,51,17,45,22,42,22,43,33,42,33,44,33,51]
NE 91 8.2 [41,37,45,36,46,32,46,30,35,31,35,33,32,33,32,38]
NV 75 7.4 [5,33,7,40,11,39,16,38,14,27,12,23]
NH 79 5.8 [83,47,83,46,83,41,85,42,85,43,84,47]
NJ 86 5.9 [82,33,82,36,81,36,81,37,80,38,80,36,81,35,80,34,80,33,81,32]
NM 62 3.7 [25,14,26,13,23,14,22,13,21,13,23,25,33,24,33,23,32,13]
NY 110 5.1 [82,36,84,38,85,38,82,36,81,37,79,38,72,37,71,38,72,40,78,45,80,46,82,41]
NC 71 7.1 [74,21,77,20,80,22,81,25,80,26,70,25,66,21,68,21]
OH 74 5.7 [70,37,71,34,71,32,68,28,64,30,64,35,66,36]
OK 63 4.7 [38,19,38,23,33,23,33,24,35,24,48,24,48,23,48,20,48,16]
OR 75 6.4 [7,40,11,39,13,47,3,51,0,42]
PA 75 5.8 [81,35,80,36,80,38,79,38,72,37,72,38,70,37,71,32,73,32,79,33]
RI 71 6.3 [85,38,86,39,85,40,84,40]
MA 105 6.1 [86,39,85,40,84,40,82,39,82,41,83,41,85,42,86,40,87,40]
TN 74 7.7 [57,23,56,22,55,20,59,20,64,20,66,21,70,25,67,24]
TX 85 6.7 [25,14,32,13,33,23,38,23,38,19,48,16,49,15,49,8,43,4,43,-1,39,1,34,8,32,7]
UT 80 6.3 [21,35,21,37,16,38,14,27,23,25,24,34]
VT 67 5.6 [83,41,82,41,80,46,83,46]
VA 81 6.6 [73,30,71,26,69,27,67,24,70,25,80,26,80,27,81,30,80,30,80,27,79,29,76,32]
WA 103 8.6 [15,54,13,47,3,51,3,55,6,54,7,56]
WV 56 2.1 [76,32,73,31,73,32,71,32,71,34,71,32,68,28,69,27,71,26,73,30]
WI 70 4.5 [51,45,54,45,58,42,59,36,54,36,53,38]
WY 86 2.2 [33,42,22,43,22,42,21,37,21,35,24,34,32,33,32,38]
DC 259 5.9 [78,31,78,30,77,30,77,31,78,31]
IL 87 5.6 [59,35,59,26,57,24,53,31,54,36,59,36]
KY 61 7.2 [57,23,57,24,59,26,62,27,64,30,68,28,69,27,67,24]
IN 73 5.7 [59,35,59,26,62,27,64,30,64,35,60,35]
GA 73 5.0 [73,12,66,11,66,12,64,20,66,21,68,21,73,15]
SC 60 8.3 [68,21,73,15,77,20,74,21]'''

gdp_per_capita = []
growth = []
patches = []
for line in data.split('\n'):
    l = line.split(' ')
    gdp_per_capita.append(float(l[1]))
    growth.append(float(l[2]))
    nodes = np.array([float(k) for k in l[3][1:-1].split(',')]).reshape((-1, 2))
    patches.append(matplotlib.patches.Polygon(nodes))

fig, ax = plt.subplots(1, 1, figsize=(8, 4))
cmaps = matplotlib.bivar_colormaps['BiOrangeBlue']
norm_growth = matplotlib.colors.Normalize(vmin=2, vmax=10)
norm_capita = matplotlib.colors.LogNorm(vmin=50, vmax=110)

p = matplotlib.collections.PatchCollection(patches,
                                           cmap=cmaps,
                                           norm=(norm_growth, norm_capita))
p.set_array((growth, gdp_per_capita))
ax.add_collection(p)
cb = fig.colorbar_2D(p)  # , shape = (1,2))
cb.set_ylabel('Growth [%]')
cb.set_xlabel('GDP per capita [1000$]\n(log scale)')

ax.set_aspect('equal')
ax.autoscale_view()
ax.set_xticks([])
ax.set_yticks([])

fig.suptitle('Economic statistics of select US states (2023)')
fig.show()

# %%
#
# See also:
# :doc:`/gallery/color/bivariate_colormap_reference`
#
# .. admonition:: References
#
#
#    - `matplotlib.colors.BivarColormap`
#    - `matplotlib.axes.Axes.imshow`
#    - `matplotlib.axes.Axes.pcolor`
#    - `matplotlib.axes.Axes.pcolormesh`
#    - `matplotlib.collections.PatchCollection`
