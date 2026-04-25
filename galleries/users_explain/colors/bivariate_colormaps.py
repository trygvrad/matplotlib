"""
.. redirect-from:: /tutorials/colors/bivariatecolormaps

.. _colormaps:

*******************
Bivariate Colormaps
*******************

Matplotlib has a number of built-in bivariate colormaps accessible via
`.matplotlib.bivar_colormaps`. This page shows the included colormaps
and how they can be discretized and rotated.

To get a list of all registered colormaps, you can do::

    from matplotlib import bivar_colormaps
    list(bivar_colormaps)


Overview
========

The colormaps presented here are grouped according to their function:
sequential×sequential, radial, sequential×diverging and sequential × cyclic.

The colormaps are designed in a perceptually uniform *Lab* colorspace (CAM02-LCD).
The colormaps are named by the shape they form in the *Lab* colorspace.
Here *L* is lightness, *a* is the red-green axis and *b* is the blue-yellow axis.

Classes of colormaps
====================
"""

# sphinx_gallery_thumbnail_number = 2

import matplotlib.pyplot as plt
import numpy as np

import matplotlib
from matplotlib.colors import BivarColormapFromImage


def plot_bivariate_cmaps(cmap_category, cmap_list):
    # Create figure and adjust figure height to number of colormaps
    nrows = len(cmap_list)//3 + (len(cmap_list) % 3 > 0)
    figh = 0.7 + 0.15 + (nrows)*2
    fig, axs = plt.subplots(ncols=3, nrows=nrows, figsize=(6.4, figh))
    axs = axs.ravel()
    fig.subplots_adjust(top=1-0.7 / figh, bottom=.15/figh, left=0.01, right=0.99)

    fig.suptitle(f"{cmap_category} colormaps", fontsize=14, ha='left', x=0)

    for ax, cmap_name in zip(axs, cmap_list):
        cmap = matplotlib.bivar_colormaps[cmap_name]
        ax.imshow(cmap.lut, origin='lower')
        ax.text(0.5, 1.03, cmap_name, va='bottom', ha='center', fontsize=12,
                transform=ax.transAxes)
    for ax in axs:
        if len(ax.images) == 0:
            ax.remove()

    # Turn off ticks
    for ax in axs:
        ax.set_yticks([])
        ax.set_xticks([])


# %%
# Sequential × sequential
# -----------------------

plot_bivariate_cmaps('Sequential × sequential',
                     ['BiOrangeBlue', 'BiGreenPurple'],
                     )

# %%
# Radial
# ------
#
# **Note regarding colorblindess**:
# For radial colormaps, both the *a* (red-green) and *b* (blue-yellow) axis,
# must be used, and as a degeneracy occurs for those that are red-green colorblind
# (estimated 5-10% of the male population).
# For the radial colormaps shown below, this degeneracy makes the top half of
# the colormap appear as a mirror image of the bottom half.

plot_bivariate_cmaps('Radial',
                     ['BiPeak', 'BiAbyss', 'BiFlat',
                      'BiCone', 'BiFunnel', 'BiDisk']
                     )

# %%
# Sequential × diverging
# ----------------------

plot_bivariate_cmaps('Sequential × diverging',
                     ['BiCut'],
                     )

# %%
# Sequential × cyclic
# -------------------

plot_bivariate_cmaps('Sequential × cyclic',
                     ['BiBarrel'],
                     )

# %%
# Misc
# ----


plot_bivariate_cmaps('Misc',
                     ['BiHsv'],
                     )

# %%
# Extracting scalar colormaps from bivariate colormaps
# ----------------------------------------------------
#
# A bivariate colormap can be interpreted as two scalar
# colormaps. The scalar colormaps can be accessed by indexing.
# This allows for composite figures that show both the
# combined data using a bivariate colormap as well as the
# individual channels using the constituent scalar colormaps.

# image data
im_A = np.arange(100)[np.newaxis, :]*np.ones((100, 100))
im_B = np.arange(100)[:, np.newaxis]*np.ones((100, 100))
im_A[:, :] = np.sin(im_A**0.5)**4
im_B[:, :] = np.sin(im_B**0.5)**4

fig, axs = plt.subplot_mosaic("AA\nBC", figsize=(6, 6))

cim_a = axs["A"].imshow((im_A, im_B), cmap='BiOrangeBlue')
fig.colorbar_bivar(cim_a, fraction=0.45)

cim_b = axs["B"].imshow(im_A, cmap=cim_a.cmap[0], norm=cim_a.norm.norms[0])
fig.colorbar(cim_b)
cim_c = axs["C"].imshow(im_B, cmap=cim_a.cmap[1], norm=cim_a.norm.norms[1])
fig.colorbar(cim_c)

for ax in 'ABC':
    axs[ax].set_xticks([])
    axs[ax].set_yticks([])

# %%
# Orienting bivariate colormaps
# -----------------------------
# Additional orientations of the built-in colormaps can be obtained by resampling,
# for a total of eight orientational variants.


fig, axs = plt.subplots(4, 2, figsize=(6, 6.6))
fig.subplots_adjust(top=0.95, bottom=0.01, left=0.01, right=0.99)

axs = axs.ravel()

cmap = matplotlib.bivar_colormaps['BiOrangeBlue']

cmaps = []
# default and axis swap
resampling = (np.linspace(0, 1, 256)[np.newaxis, :]*np.ones((256, 256)),
              np.linspace(0, 1, 256)[:, np.newaxis]*np.ones((256, 256)))
cmaps.append(BivarColormapFromImage(cmap(resampling), name='Default'))
resampling = (np.linspace(0, 1, 256)[:, np.newaxis]*np.ones((256, 256)),
              np.linspace(0, 1, 256)[np.newaxis, :]*np.ones((256, 256)))
cmaps.append(BivarColormapFromImage(cmap(resampling), name='Axis swap'))

# rotation clockwise
resampling = (np.linspace(1, 0, 256)[np.newaxis, :]*np.ones((256, 256)),
              np.linspace(1, 0, 256)[:, np.newaxis]*np.ones((256, 256)))
cmaps.append(BivarColormapFromImage(cmap(resampling), name='Rotate 180°'))
resampling = (np.linspace(1, 0, 256)[:, np.newaxis]*np.ones((256, 256)),
              np.linspace(1, 0, 256)[np.newaxis, :]*np.ones((256, 256)))
cmaps.append(BivarColormapFromImage(cmap(resampling), name='Rotate 180°'))

# rotation clockwise
resampling = (np.linspace(1, 0, 256)[:, np.newaxis]*np.ones((256, 256)),
              np.linspace(0, 1, 256)[np.newaxis, :]*np.ones((256, 256)))
cmaps.append(BivarColormapFromImage(cmap(resampling), name='Rotate 90° clockwise'))
resampling = (np.linspace(0, 1, 256)[np.newaxis, :]*np.ones((256, 256)),
              np.linspace(1, 0, 256)[:, np.newaxis]*np.ones((256, 256)))
cmaps.append(BivarColormapFromImage(cmap(resampling), name='Rotate 90° clockwise'))
# rotation clockwise
resampling = (np.linspace(0, 1, 256)[:, np.newaxis]*np.ones((256, 256)),
              np.linspace(1, 0, 256)[np.newaxis, :]*np.ones((256, 256)))
cmaps.append(BivarColormapFromImage(cmap(resampling),
             name='Rotate 90° counterclockwise'))
resampling = (np.linspace(1, 0, 256)[np.newaxis, :]*np.ones((256, 256)),
              np.linspace(0, 1, 256)[:, np.newaxis]*np.ones((256, 256)))
cmaps.append(BivarColormapFromImage(cmap(resampling),
             name='Rotate 90° counterclockwise'))


for ax, cmap in zip(axs, cmaps):
    cim = ax.imshow((im_A, im_B), cmap=cmap, origin='lower')
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_title(cmap.name, x=1.1, ha='center')

    cax = fig.colorbar_bivar(cim, fraction=0.45)
    cax.ax.set_yticks([])
    cax.ax.set_xticks([])

# %%
# Discretized bivariate colormaps
# -----------------------------
#
# Discretized colormaps can also be made by resampling

fig, axs = plt.subplots(2, 2, figsize=(6, 3.3))
fig.subplots_adjust(top=0.95, bottom=0.01, left=0.01, right=0.99)
axs = axs.ravel()

# image data
im_A = np.arange(100)[np.newaxis, :]*np.ones((100, 100))
im_B = np.arange(100)[:, np.newaxis]*np.ones((100, 100))
im_A[:, :] = np.sin(im_A**0.5)**4
im_B[:, :] = np.sin(im_B**0.5)**4


cmap = matplotlib.bivar_colormaps['BiOrangeBlue']
cmaps = []

resampling = (np.linspace(0, 1, 256)[np.newaxis, :]*np.ones((256, 256)),
              np.linspace(0, 1, 256)[:, np.newaxis]*np.ones((256, 256)))
cmaps.append(BivarColormapFromImage(cmap(resampling), name='Default'))


# discretization
resampling = (np.linspace(0, 1, 5)[:, np.newaxis]*np.ones((5, 5)),
              np.linspace(0, 1, 5)[np.newaxis, :]*np.ones((5, 5)))
cmaps.append(BivarColormapFromImage(cmap(resampling), name='Discrete y and x'))
resampling = (np.linspace(0, 1, 5)[:, np.newaxis]*np.ones((5, 256)),
              np.linspace(0, 1, 256)[np.newaxis, :]*np.ones((5, 256)))
cmaps.append(BivarColormapFromImage(cmap(resampling), name='Discrete y'))
resampling = (np.linspace(0, 1, 256)[:, np.newaxis]*np.ones((256, 5)),
              np.linspace(0, 1, 5)[np.newaxis, :]*np.ones((256, 5)))
cmaps.append(BivarColormapFromImage(cmap(resampling), name='Discrete x'))

for ax, cmap in zip(axs, cmaps):
    cim = ax.imshow((im_A, im_B), cmap=cmap, origin='lower')
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_title(cmap.name, x=1.1, ha='center')

    cax = fig.colorbar_bivar(cim, fraction=0.45)
    cax.ax.set_yticks([])
    cax.ax.set_xticks([])
