"""
The Colorizer class which handles the data to color pipeline via a
normalization and a colormap.

.. seealso::

  :doc:`/gallery/color/colormap_reference` for a list of builtin colormaps.

  :ref:`colormap-manipulation` for examples of how to make
  colormaps.

  :ref:`colormaps` an in-depth discussion of choosing
  colormaps.

  :ref:`colormapnorms` for more details about data normalization.
"""

import numpy as np
from numpy import ma
from matplotlib import _api, colors, cbook, scale, cm
import matplotlib as mpl


class Colorizer():
    """
    Class that holds the data to color pipeline
    accessible via `.to_rgba(A)` and executed via
    the `.norm` and `.cmap` attributes.
    """
    def __init__(self, cmap=None, norm=None):

        self._cmap = None
        self._set_cmap(cmap)

        self._id_norm = None
        self._norm = None
        self.norm = norm

        self.callbacks = cbook.CallbackRegistry(signals=["changed"])
        self.colorbar = None

    def _scale_norm(self, norm, vmin, vmax, A):
        """
        Helper for initial scaling.

        Used by public functions that create a ScalarMappable and support
        parameters *vmin*, *vmax* and *norm*. This makes sure that a *norm*
        will take precedence over *vmin*, *vmax*.

        Note that this method does not set the norm.
        """
        if vmin is not None or vmax is not None:
            self.set_clim(vmin, vmax)
            if isinstance(norm, colors.Normalize):
                raise ValueError(
                    "Passing a Normalize instance simultaneously with "
                    "vmin/vmax is not supported.  Please pass vmin/vmax "
                    "directly to the norm when creating it.")

        # always resolve the autoscaling so we have concrete limits
        # rather than deferring to draw time.
        self.autoscale_None(A)

    @property
    def norm(self):
        return self._norm

    @norm.setter
    def norm(self, norm):
        norm = _ensure_norm(norm, n_variates=self.cmap.n_variates)
        if norm is self.norm:
            # We aren't updating anything
            return

        in_init = self.norm is None
        # Remove the current callback and connect to the new one
        if not in_init:
            self.norm.callbacks.disconnect(self._id_norm)
        self._norm = norm
        self._id_norm = self.norm.callbacks.connect('changed',
                                                    self.changed)
        if not in_init:
            self.changed()

    def to_rgba(self, x, alpha=None, bytes=False, norm=True):
        """
        Return a normalized RGBA array corresponding to *x*.

        In the normal case, *x* is a 1D or 2D sequence of scalars, and
        the corresponding `~numpy.ndarray` of RGBA values will be returned,
        based on the norm and colormap set for this Colorizer.

        There is one special case, for handling images that are already
        RGB or RGBA, such as might have been read from an image file.
        If *x* is an `~numpy.ndarray` with 3 dimensions,
        and the last dimension is either 3 or 4, then it will be
        treated as an RGB or RGBA array, and no mapping will be done.
        The array can be `~numpy.uint8`, or it can be floats with
        values in the 0-1 range; otherwise a ValueError will be raised.
        Any NaNs or masked elements will be set to 0 alpha.
        If the last dimension is 3, the *alpha* kwarg (defaulting to 1)
        will be used to fill in the transparency.  If the last dimension
        is 4, the *alpha* kwarg is ignored; it does not
        replace the preexisting alpha.  A ValueError will be raised
        if the third dimension is other than 3 or 4.

        In either case, if *bytes* is *False* (default), the RGBA
        array will be floats in the 0-1 range; if it is *True*,
        the returned RGBA array will be `~numpy.uint8` in the 0 to 255 range.

        If norm is False, no normalization of the input data is
        performed, and it is assumed to be in the range (0-1).

        """
        # First check for special case, image input:
        # First check for special case, image input:
        try:
            if x.ndim == 3:
                if x.shape[2] == 3:
                    if alpha is None:
                        alpha = 1
                    if x.dtype == np.uint8:
                        alpha = np.uint8(alpha * 255)
                    m, n = x.shape[:2]
                    xx = np.empty(shape=(m, n, 4), dtype=x.dtype)
                    xx[:, :, :3] = x
                    xx[:, :, 3] = alpha
                elif x.shape[2] == 4:
                    xx = x
                else:
                    raise ValueError("Third dimension must be 3 or 4")
                if xx.dtype.kind == 'f':
                    # If any of R, G, B, or A is nan, set to 0
                    if np.any(nans := np.isnan(x)):
                        if x.shape[2] == 4:
                            xx = xx.copy()
                        xx[np.any(nans, axis=2), :] = 0

                    if norm and (xx.max() > 1 or xx.min() < 0):
                        raise ValueError("Floating point image RGB values "
                                         "must be in the 0..1 range.")
                    if bytes:
                        xx = (xx * 255).astype(np.uint8)
                elif xx.dtype == np.uint8:
                    if not bytes:
                        xx = xx.astype(np.float32) / 255
                else:
                    raise ValueError("Image RGB array must be uint8 or "
                                     "floating point; found %s" % xx.dtype)
                # Account for any masked entries in the original array
                # If any of R, G, B, or A are masked for an entry, we set alpha to 0
                if np.ma.is_masked(x):
                    xx[np.any(np.ma.getmaskarray(x), axis=2), 3] = 0
                return xx
        except AttributeError:
            # e.g., x is not an ndarray; so try mapping it
            pass

        # This is the normal case, mapping a scalar array:
        x = ma.asarray(x)
        if norm:
            x = self.norm(x)
        rgba = self.cmap(x, alpha=alpha, bytes=bytes)
        return rgba

    def normalize(self, x):
        """
        Normalize the data in x.

        Parameters
        ----------
        x : np.array

        Returns
        -------
        np.array

        """
        return self.norm(x)

    def autoscale(self, A):
        """
        Autoscale the scalar limits on the norm instance using the
        current array
        """
        if A is None:
            raise TypeError('You must first set_array for mappable')
        # If the norm's limits are updated self.changed() will be called
        # through the callbacks attached to the norm
        self.norm.autoscale(A)

    def autoscale_None(self, A):
        """
        Autoscale the scalar limits on the norm instance using the
        current array, changing only limits that are None
        """
        if A is None:
            raise TypeError('You must first set_array for mappable')
        # If the norm's limits are updated self.changed() will be called
        # through the callbacks attached to the norm
        self.norm.autoscale_None(A)

    def _set_cmap(self, cmap):
        """
        Set the colormap for luminance data.

        Parameters
        ----------
        cmap : `.Colormap` or str or None
        """
        in_init = self._cmap is None
        cmap_obj = _ensure_cmap(cmap, accept_multivariate=True)
        if not in_init:
            if self.norm.n_output != cmap_obj.n_variates:
                raise ValueError(f"The colormap {cmap} does not support "
                                 f"{self.norm.n_output} variates as required by "
                                 "the norm on this Colorizer.")
        self._cmap = cmap_obj
        if not in_init:
            self.changed()  # Things are not set up properly yet.

    @property
    def cmap(self):
        return self._cmap

    @cmap.setter
    def cmap(self, cmap):
        self._set_cmap(cmap)

    def set_clim(self, vmin=None, vmax=None):
        """
        Set the norm limits for image scaling.

        Parameters
        ----------
        vmin, vmax : float
             The limits.

             For scalar data, the limits may also be passed as a
             tuple (*vmin*, *vmax*) as a single positional argument.

             .. ACCEPTS: (vmin: float, vmax: float)
        """
        # If the norm's limits are updated self.changed() will be called
        # through the callbacks attached to the norm
        if self.norm.n_input == 1:
            if vmax is None:
                try:
                    vmin, vmax = vmin
                except (TypeError, ValueError):
                    pass

        if vmin is not None:
            self.norm.vmin = vmin
        if vmax is not None:
            self.norm.vmax = vmax

    def get_clim(self):
        """
        Return the values (min, max) that are mapped to the colormap limits.
        """
        return self.norm.vmin, self.norm.vmax

    def changed(self):
        """
        Call this whenever the mappable is changed to notify all the
        callbackSM listeners to the 'changed' signal.
        """
        self.callbacks.process('changed')
        self.stale = True

    @property
    def vmin(self):
        return self.get_clim[0]

    @vmin.setter
    def vmin(self, vmin):
        self.set_clim(vmin=vmin)

    @property
    def vmax(self):
        return self.get_clim[1]

    @vmax.setter
    def vmax(self, vmax):
        self.set_clim(vmax=vmax)

    @property
    def clip(self):
        return self.norm.clip

    @clip.setter
    def clip(self, clip):
        self.norm.clip = clip

    def __getitem__(self, index):
        """
        Returns a Colorizer object containing the norm and colormap for one axis
        """
        if self.cmap.n_variates > 1 and self.norm:
            if index >= 0 and index < self.cmap.n_variates:
                part = Colorizer(cmap=self._cmap[index], norm=self._norm.norms[index])
                part._super_colorizer = self
                part._super_colorizer_index = index
                part._id_parent_cmap = id(self.cmap)
                part._id_parent_norm = id(self._norm[index])
                self.callbacks.connect('changed', part._check_update_super_colorizer)
                return part
        elif self.cmap.n_variates == 1 and index == 0:
            return self
        raise ValueError(f'Only 0..{self.cmap.n_variates-1} are valid indexes'
                         ' for this Colorizer object.')

    def _check_update_super_colorizer(self):
        """
        If this `Colorizer` object was created by __getitem__ it is a
        one-dimensional component of another `Colorizer`.
        In this case, `self._super_colorizer` is the Colorizer this was generated from.

        This function propagetes changes from the `self._super_colorizer` to `self`.
        """
        if hasattr(self, '_super_colorizer'):
            # _super_colorizer, the colorizer this is a component of
            if id(self._super_colorizer.cmap) != self._id_parent_cmap:
                self.cmap = self._super_colorizer.cmap[self._super_colorizer_index]
            super_colorizer_norm =\
                    self._super_colorizer._norm[self._super_colorizer_index]
            if id(super_colorizer_norm) != self._id_parent_norm:
                self.norm = [super_colorizer_norm]


def _get_colorizer(cmap, norm):
    """
    Passes or creates a Colorizer object.

    Allows users to pass a Colorizer as the norm keyword
    where a artist.ColorizingArtist is used as the artist.
    If a Colorizer object is not passed, a Colorizer is created.
    """
    if isinstance(norm, Colorizer):
        if cmap:
            raise ValueError("Providing a `cm.Colorizer` as the norm while "
                             "at the same time providing a `cmap` is not supported.")
        return norm
    return Colorizer(cmap, norm)


class ColorizerShim:

    def _scale_norm(self, norm, vmin, vmax):
        self.colorizer._scale_norm(norm, vmin, vmax, self._A)

    def to_rgba(self, x, alpha=None, bytes=False, norm=True):
        """
        Return a normalized RGBA array corresponding to *x*.

        In the normal case, *x* is a 1D or 2D sequence of scalars, and
        the corresponding `~numpy.ndarray` of RGBA values will be returned,
        based on the norm and colormap set for this Colorizer.

        There is one special case, for handling images that are already
        RGB or RGBA, such as might have been read from an image file.
        If *x* is an `~numpy.ndarray` with 3 dimensions,
        and the last dimension is either 3 or 4, then it will be
        treated as an RGB or RGBA array, and no mapping will be done.
        The array can be `~numpy.uint8`, or it can be floats with
        values in the 0-1 range; otherwise a ValueError will be raised.
        Any NaNs or masked elements will be set to 0 alpha.
        If the last dimension is 3, the *alpha* kwarg (defaulting to 1)
        will be used to fill in the transparency.  If the last dimension
        is 4, the *alpha* kwarg is ignored; it does not
        replace the preexisting alpha.  A ValueError will be raised
        if the third dimension is other than 3 or 4.

        In either case, if *bytes* is *False* (default), the RGBA
        array will be floats in the 0-1 range; if it is *True*,
        the returned RGBA array will be `~numpy.uint8` in the 0 to 255 range.

        If norm is False, no normalization of the input data is
        performed, and it is assumed to be in the range (0-1).

        """
        return self.colorizer.to_rgba(x, alpha=alpha, bytes=bytes, norm=norm)

    def get_clim(self):
        """
        Return the values (min, max) that are mapped to the colormap limits.
        """
        return self.colorizer.get_clim()

    def set_clim(self, vmin=None, vmax=None):
        """
        Set the norm limits for image scaling.

        Parameters
        ----------
        vmin, vmax : float
             The limits.

             For scalar data, the limits may also be passed as a
             tuple (*vmin*, *vmax*) as a single positional argument.

             .. ACCEPTS: (vmin: float, vmax: float)
        """
        # If the norm's limits are updated self.changed() will be called
        # through the callbacks attached to the norm
        self.colorizer.set_clim(vmin, vmax)

    def get_alpha(self):
        """
        Returns
        -------
        float
            Always returns 1.
        """
        # This method is intended to be overridden by Artist sub-classes
        return 1.

    @property
    def cmap(self):
        return self.colorizer.cmap

    @cmap.setter
    def cmap(self, cmap):
        self.colorizer.cmap = cmap

    def get_cmap(self):
        """Return the `.Colormap` instance."""
        return self.colorizer.cmap

    def set_cmap(self, cmap):
        """
        Set the colormap for luminance data.

        Parameters
        ----------
        cmap : `.Colormap` or str or None
        """
        self.cmap = cmap

    @property
    def norm(self):
        return self.colorizer.norm

    @norm.setter
    def norm(self, norm):
        self.colorizer.norm = norm

    def set_norm(self, norm):
        """
        Set the normalization instance.

        Parameters
        ----------
        norm : `.Normalize` or str or None

        Notes
        -----
        If there are any colorbars using the mappable for this norm, setting
        the norm of the mappable will reset the norm, locator, and formatters
        on the colorbar to default.
        """
        self.norm = norm

    def autoscale(self):
        """
        Autoscale the scalar limits on the norm instance using the
        current array
        """
        self.colorizer.autoscale(self._A)

    def autoscale_None(self):
        """
        Autoscale the scalar limits on the norm instance using the
        current array, changing only limits that are None
        """
        self.colorizer.autoscale_None(self._A)

    @property
    def colorbar(self):
        return self.colorizer.colorbar

    @colorbar.setter
    def colorbar(self, colorbar):
        self.colorizer.colorbar = colorbar


def _ensure_norm(norm, n_variates=1):
    if n_variates == 1:
        _api.check_isinstance((colors.Normalize, str, None), norm=norm)
        if norm is None:
            norm = colors.Normalize()
        elif isinstance(norm, str):
            try:
                scale_cls = scale._scale_mapping[norm]
            except KeyError:
                raise ValueError(
                    "Invalid norm str name; the following values are "
                    f"supported: {', '.join(scale._scale_mapping)}"
                ) from None
            norm = colors._auto_norm_from_scale(scale_cls)()
        return norm
    else:  # n_variates > 1
        if not np.iterable(norm):
            # include tuple in the list to improve error message
            _api.check_isinstance((colors.Normalize, str, None, tuple), norm=norm)

        if norm is None:
            norm = colors.MultiNorm([None]*n_variates)
        elif isinstance(norm, str):  # single string
            norm = colors.MultiNorm([norm]*n_variates)
        elif np.iterable(norm):  # multiple string or objects
            norm = colors.MultiNorm(norm)
        if isinstance(norm, colors.Normalize) and norm.n_output == n_variates:
            return norm
        raise ValueError(
                "Invalid norm for multivariate colormap with "
                f"{n_variates} inputs."
            )


def _ensure_cmap(cmap, accept_multivariate=False):
    """
    Ensure that we have a `.Colormap` object.

    For internal use to preserve type stability of errors.

    Parameters
    ----------
    cmap : None, str, Colormap

        - if a `~matplotlib.colors.Colormap`,
          `~matplotlib.colors.MultivarColormap` or
          `~matplotlib.colors.BivarColormap`,
          return it
        - if a string, look it up in three corresponding databases
          when not found: raise an error based on the expected shape
        - if None, look up the default color map in mpl.colormaps
    accept_multivariate : bool, default True
        - if False, accept only Colormap, string in mpl.colormaps or None

    Returns
    -------
    Colormap

    """
    if not accept_multivariate:
        if isinstance(cmap, colors.Colormap):
            return cmap
        cmap_name = cmap if cmap is not None else mpl.rcParams["image.cmap"]
        # use check_in_list to ensure type stability of the exception raised by
        # the internal usage of this (ValueError vs KeyError)
        if cmap_name not in mpl.colormaps:
            _api.check_in_list(sorted(mpl.colormaps), cmap=cmap_name)

    if isinstance(cmap, (colors.Colormap,
                         colors.BivarColormap,
                         colors.MultivarColormap)):
        return cmap
    cmap_name = cmap if cmap is not None else mpl.rcParams["image.cmap"]
    if cmap_name in mpl.colormaps:
        return mpl.colormaps[cmap_name]
    if cmap_name in mpl.multivar_colormaps:
        return mpl.multivar_colormaps[cmap_name]
    if cmap_name in mpl.bivar_colormaps:
        return mpl.bivar_colormaps[cmap_name]

    # this error message is a variant of _api.check_in_list but gives
    # additional hints as to how to access multivariate colormaps

    msg = f"{cmap!r} is not a valid value for cmap"
    msg += "; supported values for scalar colormaps are "
    msg += f"{', '.join(map(repr, sorted(mpl.colormaps)))}\n"
    msg += "See matplotlib.bivar_colormaps() and"
    msg += " matplotlib.multivar_colormaps() for"
    msg += " bivariate and multivariate colormaps."

    raise ValueError(msg)

    if isinstance(cmap, colors.Colormap):
        return cmap
    cmap_name = cmap if cmap is not None else mpl.rcParams["image.cmap"]
    # use check_in_list to ensure type stability of the exception raised by
    # the internal usage of this (ValueError vs KeyError)
    if cmap_name not in cm.colormaps:
        _api.check_in_list(sorted(cm.colormaps), cmap=cmap_name)
    return cm.colormaps[cmap_name]


def _ensure_color_pipeline_compatibility(cmap, norm, data):
    """
    Ensures that the norm, colormap, and data forms a coherent pipeline.

    Checks that the dimensionality of the input to the cmap matches the
    output from the norm. Then checks that the input to the norm matches
    the data.

    Parameters
    ----------
    cmap : None, str, colors.Colormap, colors.BivarColormap, colors.multivarColormap

    norm : None, str, colors.Normalize, tuple

        If tuple, the each element must be a compatible type, and the
        length must be equal to the number of variats in the colormap.

    data : array-like
        Supported array shapes are:

        - (n, ...): where n is the number of input varites of the norm
        - (...) with a dtype with n fields.

    Returns
    -------
        Colormap : colors.Colormap, colors.BivarColormap, colors.multivarColormap
        Norm : colors.Normalize
        Data : np.ndarray, PIL.Image or None

    """
    if isinstance(norm, Colorizer):
        cmap = norm.cmap
        norm = norm.norm

    cmap = _ensure_cmap(cmap, accept_multivariate=True)
    norm = _ensure_norm(norm, cmap.n_variates)
    if not norm.n_output == cmap.n_variates:
        raise ValueError(f"The chosen colormap requires {cmap.n_variates} inputs "
                         f"but the chosen norm only provides {norm.n_output} outputs.")
    data = _ensure_multivariate_data(norm.n_input, data)

    return cmap, norm, data


def _ensure_multivariate_data(n_input, data):
    """
    Ensure that the data has dtype with n_input.
    Input data of shape (n_input, n, m) is converted to an array of shape
    (n, m) with data type np.dtype(f'{data.dtype}, ' * n_input)
    Complex data is returned as a view with dtype np.dtype('float64, float64')
    or np.dtype('float32, float32')
    If n_input is 1 and data is not of type np.ndarray (i.e. PIL.Image),
    the data is returned unchanged.
    If data is None, the function returns None
    Parameters
    ----------
    n_input : int
        -  number of variates in the data
    data : np.ndarray, PIL.Image or None
    Returns
    -------
        np.ndarray, PIL.Image or None
    """

    if isinstance(data, np.ndarray):
        if len(data.dtype.descr) == n_input:
            # pass scalar data
            # and already formatted data
            return data
        elif data.dtype in [np.complex64, np.complex128]:
            # pass complex data
            if data.dtype == np.complex128:
                dt = np.dtype('float64, float64')
            else:
                dt = np.dtype('float32, float32')
            reconstructed = np.ma.frombuffer(data.data, dtype=dt).reshape(data.shape)
            if np.ma.is_masked(data):
                for descriptor in dt.descr:
                    reconstructed[descriptor[0]][data.mask] = np.ma.masked
            return reconstructed

    if n_input > 1 and len(data) == n_input:
        # convert data from shape (n_input, n, m)
        # to (n,m) with a new dtype
        data = [np.ma.array(part, copy=False) for part in data]
        dt = np.dtype(', '.join([f'{part.dtype}' for part in data]))
        fields = [descriptor[0] for descriptor in dt.descr]
        reconstructed = np.ma.empty(data[0].shape, dtype=dt)
        for i, f in enumerate(fields):
            if data[i].shape != reconstructed.shape:
                raise ValueError("For multivariate data all variates must have same "
                                 f"shape, not {data[0].shape} and {data[i].shape}")
            reconstructed[f] = data[i]
            if np.ma.is_masked(data[i]):
                reconstructed[f][data[i].mask] = np.ma.masked
        return reconstructed

    if data is None:
        return data

    if n_input == 1:
        # PIL.Image also gets passed here
        return data

    elif n_input == 2:
        raise ValueError("Invalid data entry for mutlivariate data. The data"
                         " must contain complex numbers, or have a first dimension 2,"
                         " or be of a dtype with 2 fields")
    else:
        raise ValueError("Invalid data entry for mutlivariate data. The shape"
                         f" of the data must have a first dimension {n_input}"
                         f" or be of a dtype with {n_input} fields")
