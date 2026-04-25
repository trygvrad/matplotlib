"""
.. redirect-from:: /tutorials/colors/multivariatecolormaps

.. _colormaps:

**********************
Multivariate Colormaps
**********************

Matplotlib has built-in multivariate colormaps accessible via
`.matplotlib.multivar_colormaps`. This page shows the included colormaps.

To get a list of all registered colormaps, you can do::

    from matplotlib import multivar_colormaps
    list(multivar_colormaps)


Overview
========

Multivariate colormaps allows users to visualize 2 or more scalar datasets
together. These colormaps are desgned to be used with sparse data, and it is
advised to visualize each channel separately in addition to the combined plot.
The component colormaps are tuned so that when combined in equal amounts a grayscale
is formed. Multivariate colormaps are categorized as either subtractive or additive,
depending on if the sRGB values from each colormap are added or subtracted when
combined.

When working with 4 or more colors, combinations of high values will exceed the
available colorspace and be clipped.

"""

# sphinx_gallery_thumbnail_number = 2

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors


def plot_multivariate_cmaps(key):
    cmap_names = [name for name in matplotlib.multivar_colormaps.keys()
                  if key in name]

    fig = plt.figure(figsize=(6.4, (len(cmap_names)+0.6)*0.32))
    vstep = 1/(len(cmap_names)+0.5)
    fig.text((1+0.15)/2, 1-0.012*vstep, key+' colormaps',
             va='top', ha='center', fontsize=12)

    for i, cmap_name in enumerate(cmap_names):
        multivar_cmap = matplotlib.multivar_colormaps[cmap_name]
        hstep = (0.8 + 0.012) / len(multivar_cmap)
        vpos = 1.01 - (i + 1.5) * vstep
        for j, component in enumerate(multivar_cmap):
            ax = fig.add_axes((0.19 + j * hstep, vpos,
                               hstep - 0.012, vstep - 0.02))
            ax.imshow(component(np.linspace(0, 1, 256)) * np.ones((20, 256, 4)),
                      aspect='auto')
        fig.text(0.01, vpos, cmap_name, va='bottom', ha='left', fontsize=12)

    for ax in fig.axes:
        ax.set_yticks([])
        ax.set_xticks([])
    fig.show()


# %%
# Subtractive colormaps
# ----------------------

plot_multivariate_cmaps('Chroma')

# %%
# Additive colormaps
# ---------------------

plot_multivariate_cmaps('Spectra')


# %%
# Accessing component colormaps
# -----------------------------
#
# Multivar colormaps can be indexed to accessed the component scalar colormaps, for
# example::
#
#    multivar_colormaps["2Chroma"][0]  # <- a white to blue colormap
#    multivar_colormaps["2Chroma"][1]  # <- a white to orange colormap
#
# This is illustrated in the following example, where the component colormaps
# and normalizations are reused for subplots showing the separate channels

im_0 = np.arange(100)[np.newaxis, :]*np.ones((100, 100))
im_1 = np.arange(100)[:, np.newaxis]*np.ones((100, 100))

im_A = np.sin(im_0**0.5)**4
im_B = np.sin(im_1**0.5)**4
im_C = np.sin((im_0**2 + im_1**2)**0.3)**4

fig, axs = plt.subplot_mosaic("AAA\nBCD", figsize=(9, 6))

cim_a = axs["A"].imshow((im_A, im_B, im_C), cmap='3Chroma')
fig.colorbar_multivar(cim_a, fraction=0.45)

cim_b = axs["B"].imshow(im_A, cmap=cim_a.cmap[0], norm=cim_a.norm.norms[0])
fig.colorbar(cim_b)
cim_c = axs["C"].imshow(im_B, cmap=cim_a.cmap[1], norm=cim_a.norm.norms[1])
fig.colorbar(cim_c)
cim_d = axs["D"].imshow(im_C, cmap=cim_a.cmap[2], norm=cim_a.norm.norms[2])
fig.colorbar(cim_d)

for ax in 'ABC':
    axs[ax].set_xticks([])
    axs[ax].set_yticks([])
