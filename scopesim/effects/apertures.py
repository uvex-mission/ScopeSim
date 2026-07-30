# -*- coding: utf-8 -*-
"""Effects related to field masks, including spectroscopic slits."""

import warnings
from typing import ClassVar

import yaml
import numpy as np
from matplotlib.path import Path as MPLPath  # rename to avoid conflict with pathlib
from astropy.io import fits
from astropy import units as u
from astropy.table import Table

from .effects import Effect
from ..optics import image_plane_utils as imp_utils
from ..optics.fov_volume_list import FovVolumeList
from ..optics.fov import FieldOfView
from astropy.wcs import WCS

from ..utils import (quantify, quantity_from_table, from_currsys, check_keys,
                     figure_factory, get_logger)


logger = get_logger(__name__)

PLOT = True

class ApertureMask(Effect):
    """
    Only provides the on-sky window coords of the Aperture.

    - Case: Imaging
        - Covers the whole FOV of the detector
        - Round (with mask), square (without mask)
    - Case : LS Spec
        - Covers the slit FOV
        - Polygonal (with mask), square (without mask)
    - Case : IFU Spec
        - Covers the on-sky FOV of one slice of the IFU
        - Square (without mask)
    - Case : MOS Spec
        - Covers a single MOS fibre FOV
        - Round, Polygonal (with mask), square (without mask)

    The geometry of an ``ApertureMask`` can be initialised with the standard
    DataContainer methods (see Parameters below). Regardless of which method
    is used, the following columns must be present::

        x       y
        arcsec  arcsec
        float   float

    Certain keywords need to also be included in the ascii header::

        # id: <int>
        # conserve_image: <bool>
        # x_unit: <str>
        # y_unit: <str>

    If ``conserve_image`` is ``False``, the flux from all sources in the
    aperture is summed and distributed uniformly over the aperture area.


    Parameters
    ----------
    filename : str
        Path to ASCII file containing the columns listed above

    table : astropy.Table
        An astropy Table containing the columns listed above

    array_dict : dict
        A dictionary containing the columns listed above:
        ``{x: [...], y: [...], id: <int>, conserve_image: <bool>}``

    Other Parameters
    ----------------
    pixel_scale : float
        [arcsec] Defaults to ``"!INST.pixel_scale"`` from the config

    id : int
        An integer to identify the ``ApertureMask`` in a list of apertures

    """

    required_keys = {"filename", "table", "array_dict"}
    z_order: ClassVar[tuple[int, ...]] = (80, 280, 380)
    report_plot_include: ClassVar[bool] = False
    report_table_include: ClassVar[bool] = True
    report_table_rounding: ClassVar[int] = 4

    def __init__(self, **kwargs):
        if not np.any([key in kwargs for key in ["filename", "table",
                                                 "array_dict"]]):
            if "width" in kwargs and "height" in kwargs and \
                    "filename_format" in kwargs:
                kwargs = from_currsys(kwargs, self.cmds)
                w, h = kwargs["width"], kwargs["height"]
                kwargs["filename"] = kwargs["filename_format"].format(w, h)

        super().__init__(**kwargs)
        params = {
            "pixel_scale": "!INST.pixel_scale",
            "no_mask": True,
            "angle": 0,
            "shape": "rect",
            "conserve_image": True,
            "id": 0,
        }

        self.meta.update(params)
        self.meta.update(kwargs)

        self._header = None
        self._mask = None
        self.mask_sum = None

        check_keys(kwargs, self.required_keys, "warning", all_any="any")

    def apply_to(self, obj, **kwargs):
        """See parent docstring."""
        if isinstance(obj, FovVolumeList):
            logger.debug("Executing %s, FoV setup", self.meta['name'])
            x = quantity_from_table("x", self.table,
                                    u.arcsec).to_value(u.arcsec)
            y = quantity_from_table("y", self.table,
                                    u.arcsec).to_value(u.arcsec)
            obj.shrink(["x", "y"], ([min(x), max(x)], [min(y), max(y)]))

            # ..todo: HUGE HACK - Get rid of this!
            for vol in obj.volumes:
                vol["meta"]["xi_min"] = min(x) * u.arcsec
                vol["meta"]["xi_max"] = max(x) * u.arcsec

        return obj

    # Outdated. Remove when removing all old FOVManager code from effects
    def fov_grid(self, which="edges", **kwargs):
        """Return a header with the sky coordinates."""
        warnings.warn("The fov_grid method is deprecated and will be removed "
                      "in a future release.", DeprecationWarning, stacklevel=2)
        if which == "edges":
            self.meta.update(kwargs)
            return self.header
        elif which == "masks":
            self.meta.update(kwargs)
            return self.mask

    @property
    def hdu(self):
        return fits.ImageHDU(data=self.mask, header=self.header)

    @property
    def header(self):
        if not isinstance(self._header, fits.Header) \
                and "x" in self.table.colnames and "y" in self.table.colnames:
            self._header = self.get_header()
        return self._header

    def get_header(self):
        self.meta = from_currsys(self.meta, self.cmds)
        x = quantity_from_table("x", self.table, u.arcsec).to_value(u.deg)
        y = quantity_from_table("y", self.table, u.arcsec).to_value(u.deg)
        pix_scale_deg = self.meta["pixel_scale"] / 3600.
        header = imp_utils.header_from_list_of_xy(x, y, pix_scale_deg)
        header["APERTURE"] = self.meta["id"]
        header["ROT"] = self.meta["angle"]
        header["IMG_CONS"] = self.meta["conserve_image"]

        return header

    @property
    def mask(self):
        if not isinstance(self._header, fits.Header) \
                and "x" in self.table.colnames and "y" in self.table.colnames:
            self._mask = self.get_mask()
        return self._mask

    def get_mask(self):
        """
        For placing over FOVs if the Aperture is rotated w.r.t. the field.
        """
        self.meta = from_currsys(self.meta, self.cmds)

        if self.meta["no_mask"] is False:
            x = quantity_from_table("x", self.table, u.arcsec).to_value(u.deg)
            y = quantity_from_table("y", self.table, u.arcsec).to_value(u.deg)
            pixel_scale_deg = self.meta["pixel_scale"] / 3600.
            mask = mask_from_coords(x, y, pixel_scale_deg)
        else:
            mask = None

        return mask

    def plot(self, axes=None):
        if axes is None:
            fig, ax = figure_factory()
        else:
            fig = axes.figure

        x = list(self.table["x"].data)
        y = list(self.table["y"].data)
        ax.plot(x + [x[0]], y + [y[0]])
        ax.set_aspect("equal")

        return fig
    
class UVEXSlitMask(ApertureMask):
    # For the UVEX LSS, the slit mask needs to be applied *after* convolving with the PSFs at the slit
    z_order: ClassVar[tuple[int, ...]] = (27, 227, 627)
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.meta.update(kwargs)
        params = {
            "flux_accuracy": 1e-4,
            "slit_tol": 1e-4,
            "crop_x": "!SIM.computing.crop_x",
            "crop_y": "!SIM.computing.crop_y",
            "crop_unit": "!SIM.computing.crop_unit",
            "oversampling_x": "!SIM.computing.oversampling_x",
            "oversampling_y": "!SIM.computing.oversampling_y",
            "fov_x0": "!INST.fov_x0",
            "fov_y0": "!INST.fov_y0",
            "fov_unit": "!INST.fov_unit",
        }
        self.meta.update(params)
        self.meta = from_currsys(self.meta, self.cmds)

    def apply_to(self, obj, **kwargs):
        
        if isinstance(obj, FovVolumeList): # during FoV setup
            logger.debug("Executing %s, FoV setup", self.meta['name'])
            params = {}
            x = quantity_from_table("x", self.table,
                                    u.arcsec).to_value(u.arcsec)
            y = quantity_from_table("y", self.table,
                                    u.arcsec).to_value(u.arcsec)
            params['slit_x'] = x
            params['slit_y'] = y
            self.meta.update(params)

            # Automatically detect slit orientation: longer dimension is spatial
            x_extent = max(x) - min(x)
            y_extent = max(y) - min(y)
            for vol in obj.volumes:
                if x_extent > y_extent:
                    # Horizontal slit: x is spatial (xi)
                    vol["meta"]["xi_min"] = min(x) * u.arcsec
                    vol["meta"]["xi_max"] = max(x) * u.arcsec
                else:
                    # Vertical slit: y is spatial (xi)
                    vol["meta"]["xi_min"] = min(y) * u.arcsec
                    vol["meta"]["xi_max"] = max(y) * u.arcsec

            # optionally add a buffer in the spectral direction so we can convolve and then apply the slit mask
            # can also crop the region in the spatial direction for memory reasons
            crop_x = self.meta.get("crop_x", None)
            crop_y = self.meta.get("crop_y", None)
            fov_xcen = self.meta.get("fov_x0", 0.)
            fov_ycen = self.meta.get("fov_y0", 0.)
            fov_unit = u.Unit(self.meta.get("fov_unit", "arcsec"))

            fov_xcen = (fov_xcen * fov_unit).to(u.arcsec).value
            fov_ycen = (fov_ycen * fov_unit).to(u.arcsec).value

            if crop_x != [-13.2, 13.2]:
                logger.warning(f"Recommended value for crop_x is [-13.2, 13.2]. Otherwise, the slit and image may be offset in pixel space.")

            if crop_x is not None:
                xmin, xmax = fov_xcen + crop_x[0], fov_xcen + crop_x[1]
            else:
                xmin, xmax = min(x), max(x)
            
            oversampled_scale = from_currsys(self.meta["pixel_scale"], self.cmds) / self.meta["oversampling_x"] # already in arcsec
            n_pix = int(round((xmax - xmin) / oversampled_scale))
            if n_pix % 2 == 0:
                n_pix += 1  # Force the oversampled pixel grid to be odd
            half_width = 0.5 * n_pix * oversampled_scale
            xmin, xmax = fov_xcen - half_width, fov_xcen + half_width
            if crop_y is not None:
                ymin, ymax = fov_ycen + crop_y[0], fov_ycen + crop_y[1]
            else:
                ymin, ymax = min(y), max(y)

            obj.shrink(["x", "y"], ([xmin, xmax], [ymin, ymax]))

        elif isinstance(obj, FieldOfView): # During application of the effect itself
            logger.debug("Executing %s, applying slit mask effect", self.meta['name'])
            
            data = obj.hdu.data.astype(float)
                
            hdr = obj.hdu.header
            wcs = WCS(hdr)

            slit_xmin = min(self.meta["slit_x"]) * u.arcsec
            slit_xmax = max(self.meta["slit_x"]) * u.arcsec
            slit_ymin = min(self.meta["slit_y"]) * u.arcsec
            slit_ymax = max(self.meta["slit_y"]) * u.arcsec

            # Build a pixel mask and zero out what's outside the slit
            nlam, ny, nx = data.shape
            slit_xmin, slit_xmax = slit_xmin.to(u.deg).value, slit_xmax.to(u.deg).value
            slit_ymin, slit_ymax = slit_ymin.to(u.deg).value, slit_ymax.to(u.deg).value

            ys, xs = np.mgrid[0:ny, 0:nx]
            lambdas = np.zeros_like(xs, dtype=float) # just use first wavelength slice (mask is same for all wavelenght slices)
            xfld, yfld, _ = wcs.pixel_to_world(xs, ys, lambdas) # deg
            xfld = xfld.value
            yfld = yfld.value

            mask = (xfld >= slit_xmin) & (xfld <= slit_xmax) & (yfld >= slit_ymin) & (yfld <= slit_ymax)
            
            x_mask = (xfld[xfld.shape[0] // 2, :] >= slit_xmin) & (xfld[xfld.shape[0] // 2, :] <= slit_xmax)
            xfld_inmask = xfld[xfld.shape[0] // 2, x_mask]
            x_pixelscale = (np.abs(obj.hdu.header["CDELT1"]) * u.deg).value
            
            # in WCS, pixel coordinates fall at the center of pixels
            leftedge = xfld_inmask[0] - slit_xmin
            rightedge = xfld_inmask[-1] - slit_xmax
            
            if (np.abs(np.abs(leftedge) - (x_pixelscale / 2)) >= self.meta["slit_tol"]) or (np.abs(np.abs(rightedge) - (x_pixelscale / 2)) >= self.meta["slit_tol"]):
                logger.warning("The UVEX slit width is not sampled perfectly, which can give an inaccurate throughput. Consider changing oversampling_x or crop_x.")
                
            img = np.where(mask[np.newaxis,:,:], data, 0.0)

            if PLOT:
                import matplotlib.pyplot as plt
                plt.title("Image slice after slit mask")
                plt.imshow(img[img.shape[0] // 2,:,:])
                plt.show()

            logger.info(f"Calculated slit throughput: {img.sum() / data.sum()}")
            obj.hdu.data = img

        return obj

class RectangularApertureMask(ApertureMask):
    required_keys = {"x", "y", "width", "height"}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        params = {"x_unit": "arcsec",
                  "y_unit": "arcsec"}
        self.meta.update(params)
        self.meta.update(kwargs)
        check_keys(self.meta, self.required_keys)

        self.table = self.get_table(**kwargs)

    def get_table(self, **kwargs):
        self.meta.update(kwargs)
        x = from_currsys(self.meta["x"], self.cmds)
        y = from_currsys(self.meta["y"], self.cmds)
        dx = 0.5 * from_currsys(self.meta["width"], self.cmds)
        dy = 0.5 * from_currsys(self.meta["height"], self.cmds)
        xs = [x - dx, x + dx, x + dx, x - dx]
        ys = [y - dy, y - dy, y + dy, y + dy]
        tbl = Table(names=["x", "y"], data=[xs, ys], meta=self.meta)

        return tbl


class ApertureList(Effect):
    """
    A list of apertures, useful for IFU or MOS instruments.

    Parameters
    ----------

    Examples
    --------

    File format
    -----------

    Much like an ApertureMask, an ApertureList can be initialised by either
    of the three standard DataContainer methods. The easiest is however to
    make an ASCII file with the following columns::

        id   left    right   top     bottom  angle  conserve_image  shape
             arcsec  arcsec  arcsec  arcsec  deg
        int  float   float   float   float   float  bool            str/int

    Acceptable ``shape`` entries are: ``round``, ``rect``, ``hex``, ``oct``, or
    an integer describing the number of corners the polygon should have.

    A polygonal mask is generated for a given ``shape`` and will be scaled
    to fit inside the edges of each aperture list row. The corners of each
    aperture defined by shape are found by finding equidistant positions around
    an ellipse constrained by the edges (``left``, ..., etc). An additional
    optional column ``offset`` may be added. This column describes the offset
    from 0 deg to the angle where the first corner is set.

    Additionally, the filename of an ``ApertureMask`` polygon file can be
    given. The geometry of the polygon defined in the file will be scaled to
    fit inside the edges of the row.

    .. note:: ``shape`` values ``"rect"`` and ``4`` do not produce equal results

       Both use 4 equidistant points around an ellipse constrained by
       [``left``, ..., etc]. However ``"rect"`` aims to fill the contraining
       area, while ``4`` simply uses 4 points on the ellipse.
       Consequently, ``4`` results in a diamond shaped mask covering only
       half of the constraining area filled by ``"rect"``.


    """

    z_order: ClassVar[tuple[int, ...]] = (81, 281)
    report_plot_include: ClassVar[bool] = True
    report_table_include: ClassVar[bool] = True
    report_table_rounding: ClassVar[int] = 4

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        params = {
            "pixel_scale": "!INST.pixel_scale",
            "n_round_corners": 32,        # number of corners use to estimate ellipse
            "no_mask": False,             # .. todo:: is this necessary when we have conserve_image?
        }
        self.meta.update(params)
        self.meta.update(kwargs)

        if self.table is not None:
            # Why not always?
            required_keys = {"id", "left", "right", "top", "bottom", "angle",
                             "conserve_image", "shape"}
            check_keys(self.table.colnames, required_keys)

    def apply_to(self, obj, **kwargs):
        """See parent docstring."""
        if isinstance(obj, FovVolumeList):
            logger.debug("Executing %s, FoV setup", self.meta['name'])
            new_vols = []
            for row in self.table:
                vols = obj.extract(["x", "y"], ([row["left"], row["right"]],
                                                [row["bottom"], row["top"]]))
                for vol in vols:
                    vol["meta"]["aperture_id"] = row["id"]

                    # ..todo: HUGE HACK - Get rid of this!
                    vol["meta"]["xi_min"] = row["left"] * u.arcsec
                    vol["meta"]["xi_max"] = row["right"] * u.arcsec
                    vol["conserve_image"] = row["conserve_image"]
                    vol["shape"] = row["shape"]
                    vol["angle"] = row["angle"]

                new_vols += vols

            obj.volumes = new_vols

        return obj

    @property
    def apertures(self):
        return self.get_apertures(range(len(self.table)))

    def get_apertures(self, row_ids):
        if isinstance(row_ids, int):
            row_ids = [row_ids]

        apertures_list = []
        for ii in row_ids:
            row = self.table[ii]
            row_dict = {col: row[col] for col in row.colnames}
            row_dict["n_round"] = self.meta["n_round_corners"]
            array_dict = make_aperture_polygon(**row_dict)
            params = {
                "id": row["id"],
                "angle": row["angle"],
                "shape": row["shape"],
                "conserve_image": yaml.full_load(str(row["conserve_image"])),
                "no_mask": self.meta["no_mask"],
                "pixel_scale": self.meta["pixel_scale"],
                "x_unit": "arcsec",
                "y_unit": "arcsec",
                "angle_unit": "arcsec",
            }
            apertures_list.append(ApertureMask(array_dict=array_dict, **params))

        return apertures_list

    def plot(self):
        fig, ax = figure_factory()

        for ap in self.apertures:
            ap.plot(ax)

        return fig

    def plot_masks(self):
        aps = self.apertures
        n = len(aps)
        w = np.ceil(n ** 0.5).astype(int)
        assert int(n ** 0.5) == w + 1
        h = np.ceil(n / w).astype(int)
        assert int(n / w) == h + 1
        # TODO: change these?

        fig, axes = figure_factory(w, h)
        for ap, ax in zip(aps, axes):
            ax.imshow(ap.mask.T)
        fig.show()
        return fig

    def __add__(self, other):
        if isinstance(other, ApertureList):
            from astropy.table import vstack
            self.table = vstack([self.table, other.table])

            return self
        else:
            raise ValueError("Secondary argument not of type ApertureList: "
                             f"{type(other) = }")

    # def __getitem__(self, item):
    #     return self.get_apertures(item)[0]


class SlitWheel(Effect):
    """
    Selection of predefined spectroscopic slits and possibly other field masks.

    It should contain an open position.
    A user can define a non-standard slit by directly using the Aperture
    effect.

    .. todo: This is based on FilterWheel. There is a more efficient way to do this, when we have time.

    Parameters
    ----------
    slit_names : list of str

    filename_format : str
        A f-string for the path to the slit files

    current_slit : str
        Default name

    Examples
    --------
    This Effect assumes a folder full of ASCII files containing the edges of
    each slit. Each file should be names the same except for the slit's name
    or identifier.

    This example assumes a folder ``masks`` containing the slit ASCII files
    with the naming convention: ``slit_A.dat``, ``slit_B.dat``, etc.
    ::

        name: slit_wheel
        class: SlitWheel
        kwargs:
            slit_names:
                - A
                - B
                - C
            filename_format: "masks/slit_{}.dat
            current_slit: "C"

    """

    required_keys = {"slit_names", "filename_format", "current_slit"}
    z_order: ClassVar[tuple[int, ...]] = (80, 280, 580)
    report_plot_include: ClassVar[bool] = False
    report_table_include: ClassVar[bool] = True
    report_table_rounding: ClassVar[int] = 4
    _current_str = "current_slit"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        check_keys(kwargs, self.required_keys, action="error")

        params = {
            "path": "",
        }
        self.meta.update(params)
        self.meta.update(kwargs)

        path = self._get_path()
        self.slits = {}
        for name in from_currsys(self.meta["slit_names"], self.cmds):
            kwargs["name"] = name
            fname = str(path).format(name)
            self.slits[name] = ApertureMask(filename=fname, **kwargs)

        self.table = self.get_table()

    def apply_to(self, obj, **kwargs):
        """Use apply_to of current_slit."""
        return self.current_slit.apply_to(obj, **kwargs)

    def fov_grid(self, which="edges", **kwargs):
        """See parent docstring."""
        warnings.warn("The fov_grid method is deprecated and will be removed "
                      "in a future release.", DeprecationWarning, stacklevel=2)
        return self.current_slit.fov_grid(which=which, **kwargs)

    def change_slit(self, slitname=None):
        """Change the current slit."""
        if not slitname or slitname in self.slits.keys():
            self.meta["current_slit"] = slitname
            self.include = slitname
        else:
            raise ValueError("Unknown slit requested: " + slitname)

    def add_slit(self, newslit, name=None):
        """
        Add a slit to the SlitWheel.

        Parameters
        ----------
        newslit : Slit
        name : string
           Name to be used for the new slit. If ``None``, a name from
           the newslit object is used.
        """
        if name is None:
            name = newslit.display_name
        self.slits[name] = newslit

    @property
    def current_slit(self):
        """Return the currently used slit."""
        currslit = from_currsys(self.meta["current_slit"], self.cmds)
        if not currslit:
            return False
        return self.slits[currslit]

    def __getattr__(self, item):
        return getattr(self.current_slit, item)

    def get_table(self):
        """
        Create a table of slits with centre position, width and length.

        Width is defined as the extension in the y-direction, length in the
        x-direction. All values are in milliarcsec.
        """
        names = list(self.slits.keys())
        slits = self.slits.values()
        xmax = np.array([slit.data["x"].max() * u.Unit(slit.meta["x_unit"])
                         .to(u.mas) for slit in slits])
        xmin = np.array([slit.data["x"].min() * u.Unit(slit.meta["x_unit"])
                         .to(u.mas) for slit in slits])
        ymax = np.array([slit.data["y"].max() * u.Unit(slit.meta["y_unit"])
                         .to(u.mas) for slit in slits])
        ymin = np.array([slit.data["y"].min() * u.Unit(slit.meta["y_unit"])
                         .to(u.mas) for slit in slits])
        xmax = quantify(xmax, u.mas)
        xmin = quantify(xmin, u.mas)
        ymax = quantify(ymax, u.mas)
        ymin = quantify(ymin, u.mas)

        lengths = xmax - xmin
        widths = ymax - ymin
        x_centres = (xmax + xmin) / 2
        y_centres = (ymax + ymin) / 2
        tbl = Table(names=["name", "x_centre", "y_centre", "length", "width"],
                    data=[names, x_centres, y_centres, lengths, widths])
        return tbl


###############################################################################


def make_aperture_polygon(left, right, top, bottom, angle, shape, **kwargs):

    n_round = kwargs["n_round"] if "n_round" in kwargs else 32
    offset = kwargs["offset"] if "offset" in kwargs else 0.

    n_corners = {"rect": 4, "hex": 6, "oct": 8, "round": n_round}
    try:
        shape = int(float(shape))
        n_corners[shape] = shape
    except:
        pass

    x0, y0 = 0.5 * (right + left), 0.5 * (top + bottom)
    dx, dy = 0.5 * (right - left), 0.5 * (top - bottom)
    n = n_corners[shape]

    if isinstance(shape, str) and "rect" in shape:
        dx *= 1.41421356
        dy *= 1.41421356
        offset += 45.

    x, y = points_on_a_circle(n=n, x0=x0, y0=y0, dx=dx, dy=dy, offset=offset)
    if angle != 0.:
        x, y = rotate(x=x, y=y, x0=np.average(x), y0=np.average(y), angle=angle)

    return {"x": x, "y": y}


def points_on_a_circle(n, x0=0, y0=0, dx=1, dy=1, offset=0):
    deg2rad = np.pi / 180
    d_angle = np.arange(0, 360, 360 / n) + offset
    x = x0 + dx * np.cos(d_angle * deg2rad)
    y = y0 + dy * np.sin(d_angle * deg2rad)

    return x, y


def mask_from_coords(x, y, pixel_scale):
    naxis1 = int(np.ceil((np.max(x) - np.min(x)) / pixel_scale))
    naxis2 = int(np.ceil((np.max(y) - np.min(y)) / pixel_scale))
    xrange = np.linspace(np.min(x), np.max(x), naxis1)
    yrange = np.linspace(np.min(y), np.max(y), naxis2)
    coords = [(xi, yi) for xi in xrange for yi in yrange]

    corners = [(xi, yi) for xi, yi in zip(x, y)]
    path = MPLPath(corners)
    # ..todo: known issue - for super thin apertures, the first row is masked
    # rad = 0.005
    rad = 0  # increase this to include slightly more points within the polygon
    mask = path.contains_points(coords, radius=rad).reshape((naxis2, naxis1))

    return mask


def rotate(x, y, x0, y0, angle):
    """Rotate a line by `angle` [deg] around the point (`x0`, `y0`)."""
    # TODO: isn't that just a rotation matrix?
    angle_rad = angle / 57.29578
    xnew = x0 + (x - x0) * np.cos(angle_rad) - (y - y0) * np.sin(angle_rad)
    ynew = y0 + (x - x0) * np.sin(angle_rad) + (y - y0) * np.cos(angle_rad)

    return xnew, ynew
