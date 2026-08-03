# -*- coding: utf-8 -*-
"""Image-plane illumination effects."""

from typing import ClassVar
from collections.abc import Callable, Mapping

import numpy as np
from scipy.ndimage import zoom
from astropy import units as u
from astropy.io import fits
from astropy.modeling.functional_models import Gaussian2D

from . import Effect
from ..optics.image_plane import ImagePlane
from ..utils import figure_factory
from ..utils import find_file

__all__ = ["Illumination", "FitsIllumination", "gaussian2d", "quadratic_vignetting"]


def gaussian2d(
    shape: tuple[int, int],
    amp: float = 1.0,
    mu: tuple[float, float] = (0.0, 0.0),
    sigma: tuple[float, float] = (2000.0, 2000.0),
    theta: u.Quantity[u.deg] | float = 0.0 * u.deg,
) -> np.ndarray:
    """
    2D elliptical Gaussian to be used for vignetting map.

    .. versionadded:: 0.11.3

    Parameters
    ----------
    shape : tuple[int, int]
        Image shape in pixels (ny, nx).
    amp : float, optional
        Peak throughput. The default is 1.0.
    mu : tuple[float, float], optional
        Offset of the peak center in pixels (x, y) from the image center.
        The default is (0.0, 0.0), i.e. no offset.
    sigma : tuple[float, float], optional
        Gaussian widths in pixels (sx, sy). The default is (2000.0, 2000.0).
    theta : float, optional
        Rotation angle (if float, the angle is interpreted in degrees),
        counterclockwise. The default is 0°.

    Returns
    -------
    np.ndarray
        Vignetting map.

    """
    nx, ny = reversed(shape)
    y, x = np.ogrid[:ny, :nx]
    x = x - nx / 2
    y = y - ny / 2

    model = Gaussian2D(
        amplitude=amp,
        x_mean=mu[0],
        y_mean=mu[1],
        x_stddev=sigma[0],
        y_stddev=sigma[1],
        theta=theta << u.deg,
    )
    return model(x, y)


def quadratic_vignetting(
    shape: tuple[int, int],
    falloff: float = 0.01,
    r_ref: float | None = None,
    mu: tuple[float, float] = (0.0, 0.0),
    stretch: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0),
) -> np.ndarray:
    """
    Quadratic vignetting pattern with independent stretch factors.

    .. versionadded:: 0.11.3

    Parameters
    ----------
    shape : tuple[int, int]
        Image shape in pixels (ny, nx).
    falloff : float, optional
        Fractional illumination drop at `r_ref`. The default is 0.01 (= 1 %).
    r_ref : float | None, optional
        Reference radius in stretched pixels. If None (the default), use the
        corner distance.
    mu : tuple[float, float], optional
        Offset of the vignetting center in pixels (x, y) from the image center.
        The default is (0.0, 0.0), i.e. no offset.
    stretch : tuple[float, float, float, float], optional
        ``(+x, -x, +y, -y)`` independent scale factors for half-planes
        respectively. All 1.0 gives a circular pattern. A value > 1 widens the
        falloff in that direction (shallower); < 1 narrows it (steeper).
        The default is (1.0, 1.0, 1.0, 1.0).

    Returns
    -------
    np.ndarray
        Vignetting map.

    """
    nx, ny = reversed(shape)

    yy, xx = np.ogrid[:ny, :nx]
    dx = xx - (nx / 2 + mu[0])
    dy = yy - (ny / 2 + mu[1])

    sx = np.where(dx >= 0, stretch[0], stretch[1])
    sy = np.where(dy >= 0, stretch[2], stretch[3])

    r2 = (dx / sx)**2 + (dy / sy)**2

    if r_ref is None:
        r2_ref = r2.max()
    else:
        r2_ref = r_ref**2

    return np.clip(1.0 - falloff * r2 / r2_ref, 0.0, 1.0)


class FitsIllumination(Effect):
    """
    Image-plane illumination map loaded from a FITS file.

    If the FITS map dimensions do not match the ScopeSim ImagePlane,
    the map is resized using bilinear interpolation.

    The illumination map is then applied by multiplying:

        obj.hdu.data *= illumination_map
    """

    z_order: ClassVar[tuple[int, ...]] = (750,)

    def __init__(
        self,
        filename: str,
        normalize: bool = False,
        interpolate: bool = True,
        **kwargs,
    ) -> None:

        super().__init__(**kwargs)

        self.meta.setdefault("include", "!DET.include_illumination")
        self.meta["filename"] = filename
        self.meta["normalize"] = normalize
        self.meta["interpolate"] = interpolate

        # Cached full-resolution map
        self._map = None
        self._map_shape = None

    def apply_to(self, obj, **kwargs):
        if not isinstance(obj, ImagePlane):
            return obj

        # Image arrays use shape = (ny, nx)
        image_plane_shape = obj.hdu.data.shape

        # Build the map only when needed
        if self._map is None or image_plane_shape != self._map_shape:

            print(
                "FitsIllumination image-plane shape:",
                image_plane_shape,
            )

            self._map = self._make_map(image_plane_shape)
            self._map_shape = image_plane_shape

        # Apply vignetting in place
        obj.hdu.data *= self._map

        return obj

    def _make_map(self, target_shape):
        """
        Load the FITS illumination map and resize it when necessary.

        Parameters
        ----------
        target_shape : tuple[int, int]
            Required ScopeSim image-plane shape, given as (ny, nx).

        Returns
        -------
        np.ndarray
            Illumination map matching target_shape.
        """

        requested_filename = self.meta["filename"]
        filename = find_file(requested_filename)

        if filename is None:
            raise FileNotFoundError(
                "Could not locate illumination FITS file: "
                f"{requested_filename}"
            )

        # Load the coarse FITS map as float32
        illumination_map = fits.getdata(filename).astype(
            np.float32,
            copy=False,
        )

        if illumination_map.ndim != 2:
            raise ValueError(
                "Illumination FITS file must be 2D, "
                f"got shape {illumination_map.shape}"
            )

        source_shape = illumination_map.shape
        target_shape = tuple(int(value) for value in target_shape)

#        print("FitsIllumination file:", filename)
        print("FitsIllumination input-map shape:", source_shape)
#        print("FitsIllumination required shape:", target_shape)

        # ---------------------------------------------------------
        # Resize only when the FITS map and image plane differ
        # ---------------------------------------------------------
        if source_shape != target_shape:

            if not self.meta["interpolate"]:
                raise ValueError(
                    f"Illumination FITS shape {source_shape} does not "
                    f"match image-plane shape {target_shape}, and "
                    "interpolation is disabled."
                )

            # Shape ordering is (ny, nx), so the zoom factors are
            # also ordered as (y factor, x factor).
            zoom_factors = (
                target_shape[0] / source_shape[0],
                target_shape[1] / source_shape[1],
            )

            print("Interpolating illumination map:")
#            print(f"  y zoom factor = {zoom_factors[0]:.8f}")
#            print(f"  x zoom factor = {zoom_factors[1]:.8f}")

            illumination_map = zoom(
                illumination_map,
                zoom=zoom_factors,
                order=1,          # bilinear interpolation in 2D
                mode="nearest",   # avoid artificial zero-valued borders
                prefilter=False,
                grid_mode=True,
            )

            illumination_map = np.asarray(
                illumination_map,
                dtype=np.float32,
            )

            # Confirm that SciPy produced exactly the required shape
            if illumination_map.shape != target_shape:
                raise RuntimeError(
                    "Interpolation produced an unexpected shape: "
                    f"{illumination_map.shape}; expected {target_shape}"
                )

            print(
                "FitsIllumination interpolated shape:",
                illumination_map.shape,
            )

        else:
            print(
                "FitsIllumination map already matches image plane; "
                "no interpolation needed."
            )

        # ---------------------------------------------------------
        # Optional normalization
        # ---------------------------------------------------------
        if self.meta["normalize"]:
            maxval = np.nanmax(illumination_map)

            if maxval > 0:
                illumination_map = illumination_map / maxval
        """
        print("FitsIllumination final-map statistics:")
        print("  min  =", np.nanmin(illumination_map))
        print("  max  =", np.nanmax(illumination_map))
        print("  mean =", np.nanmean(illumination_map))
        print(
            "  nans =",
            np.count_nonzero(~np.isfinite(illumination_map)),
        )
        """
        return illumination_map

    def plot(self):
        if self._map is None:
            raise RuntimeError(
                "No illumination map cached — run a simulation first."
            )

        fig, ax = figure_factory()

        im = ax.imshow(
            self._map,
            origin="lower",
            cmap="gray_r",
        )

        fig.colorbar(
            im,
            ax=ax,
            label="Relative illumination",
        )

        ax.set_title("Fits Illumination")
        ax.set_xlabel("x [px]")
        ax.set_ylabel("y [px]")

        return fig
        

class Illumination(Effect):
    """Large-scale illumination variation across the image plane.

    .. versionadded:: 0.11.3

    Parameters
    ----------
    model : callable, optional
        Function ``f(shape, **kwargs) -> ndarray`` returning the
        illumination map. Defaults to :func:`gaussian2d`.
    modelargs : dict, optional
        Keyword arguments forwarded to ``model``. If omitted, the model's
        own defaults are used.

    include : str
        Turn effect on/off from the IRDB
        default.yaml.  Defaults to ``"!DET.include_illumination"``.

    Examples
    --------
    Polynomial vignetting with <1 % falloff (auto r_ref from image shape)

    >>> eff = Illumination(
    ...     model=quadratic_vignetting,
    ...     modelargs={"falloff": 0.01},
    ... )

    Custom model

    >>> def my_model(shape, slope=-0.001):
    >>>     ny, nx = shape[-2], shape[-1]
    >>>     y, x = np.ogrid[:ny, :nx]
    >>>     r = np.sqrt((x - nx / 2)**2 + (y - ny / 2)**2)
    >>>     return np.clip(1 + slope * r, 0, None)
    >>>
    >>> eff = Illumination(model=my_model, modelargs={"slope": -0.0005})

    """

    z_order: ClassVar[tuple[int, ...]] = (750,)

    def __init__(
        self,
        model: Callable = gaussian2d,
        modelargs: Mapping | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.meta.setdefault("include", "!DET.include_illumination")
        self._model = model
        self._modelargs = modelargs or {}
        self._map = None
        self._map_shape = None

    def apply_to(self, obj, **kwargs):
        if not isinstance(obj, ImagePlane):
            return obj

        shape = obj.hdu.data.shape
        print("illumination image-plane shape:", shape)

        if self._map is None or shape != self._map_shape:
            self._map = self._make_map(shape)
            self._map_shape = shape

        obj.hdu.data *= self._map
        return obj

    def _make_map(self, shape):
        illumination_map = self._model(shape, **self._modelargs)
        return illumination_map.astype(np.float32)

    def plot(self):
        """Plot effect."""
        if self._map is None:
            raise RuntimeError(
                "No illumination map cached — run a simulation first."
            )

        fig, ax = figure_factory()
        im = ax.imshow(
            self._map, origin="lower", vmin=0.98, vmax=1., cmap="gray_r",
        )
        fig.colorbar(im, ax=ax, label="Relative illumination")
        ax.set_title("Illumination")
        ax.set_xlabel("x [px]")
        ax.set_ylabel("y [px]")
        return fig
