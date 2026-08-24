# -*- coding: utf-8 -*-
from typing import ClassVar

import numpy as np
import os
from tqdm.auto import tqdm
from scipy.signal import fftconvolve

from astropy import units as u
from astropy.io import fits
from astropy.wcs import WCS

from ...optics.fov import FieldOfView
from ...optics.image_plane import ImagePlane
from ...optics.fov_volume_list import FovVolumeList
from ...utils import from_currsys, quantify
from . import logger
from ..effects import Effect
from .psf_base import get_bkg_level
from pathlib import Path

PLOT = True

# Get absolute path to irdb directory
try:
    import irdb as _irdb
    irdb_path = os.path.abspath(os.path.dirname(_irdb.__file__))
except Exception: # should be four levels up
    full_path = Path(__file__).resolve().parents[4] / "irdb"
    if full_path.exists():
        irdb_path = str(full_path)
    else:
        raise RuntimeError("Could not find irdb directory.")

class GriddedPSF(Effect):
    z_order: ClassVar[tuple[int, ...]] = (72, 672)
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        params = {
            "bkg_width": 0.0, # No background subtraction by default: see psf_base.get_bkg_level for details
            "flux_accuracy": 1e-4,
            "psf_oversampling": 10,
            "oversampling_x": "!SIM.computing.oversampling_x",
            "oversampling_y": "!SIM.computing.oversampling_y",
            "fov_x0": "!INST.fov_x0",
            "fov_y0": "!INST.fov_y0",
            "fov_unit": "!INST.fov_unit",
        }
        params.update(kwargs)
        self.meta.update(params)
        self.meta = from_currsys(self.meta, self.cmds)
        self.psf_dir = find_directory(self.meta.get("directory", None))
        self.psf_lib = self._load_psf_files()
        self.oversampling_x = self.meta.get("oversampling_x", 1)
        self.oversampling_y = self.meta.get("oversampling_y", 1)
        self._waveset = []
        self.convolution_classes = (FieldOfView, ImagePlane)
        self.psfs: list[np.ndarray] | np.ndarray = None
        self.grid_xypos: np.ndarray = None
        self.x_vals = None
        self.y_vals = None
        self.x_min, self.x_max = None, None
        self.y_min, self.y_max = None, None
        self.fov_x0 = quantify(self.meta.get("!INST.fov_x0", 0.), u.Unit(self.meta.get("!INST.fov_unit", "arcsec"))).to(u.arcsec)
        self.fov_y0 = quantify(self.meta.get("!INST.fov_y0", 0.), u.Unit(self.meta.get("!INST.fov_unit", "arcsec"))).to(u.arcsec)
        self.max_psf_size = 512

    def _load_psf_files(self):
        """Find the PSF directory and load in the PSF files."""
        if self.psf_dir is None:
            logger.error("PSF library directory not found")
            return []
        psf_files = [f for f in os.listdir(self.psf_dir) if f.endswith('.fits')]
        return sorted(psf_files)
    
    def _calc_bounding_points(self, x, y):
        """
        Obtain the indices and coordinates of the four points on the grid that bound the input coordinates(x, y).
        This is (heavily) adapted from the source code for the photutils class GriddedPSFModel (https://photutils.readthedocs.io/).
        """
        xidx = np.searchsorted(self.x_vals, x) - 1
        yidx = np.searchsorted(self.y_vals, y) - 1

        # Clip the indices to valid ranges
        xidx = np.clip(xidx, 0, len(self.x_vals) - 2)
        yidx = np.clip(yidx, 0, len(self.y_vals) - 2)

        # Find the four bounding points in the sorted grid
        # (x0, y0) is the lower-left corner of the grid
        # (x1, y1) is the upper-right corner of the grid
        x0, x1 = self.x_vals[xidx], self.x_vals[xidx + 1]
        y0, y1 = self.y_vals[yidx], self.y_vals[yidx + 1]

        # Find the indices of these points in grid_xypos
        xcoords, ycoords = self.grid_xypos.T
        lower_left = np.where((xcoords == x0) & (ycoords == y0))[0][0]
        lower_right = np.where((xcoords == x1) & (ycoords == y0))[0][0]
        upper_left = np.where((xcoords == x0) & (ycoords == y1))[0][0]
        upper_right = np.where((xcoords == x1) & (ycoords == y1))[0][0]

        grid_idx = (lower_left, lower_right, upper_left, upper_right)
        grid_xy = (x0, x1, y0, y1)
        
        return grid_idx, grid_xy
        
    def _psf_interp(self, xi, yi):
        """
        Given input coordinates (xi, yi), compute the effective PSF by interpolating between
        the four PSFs at the bounding grid points.
        """
        # given xi, yi, find the bounding points
        grid_idx, grid_xy = self._calc_bounding_points(xi, yi)
        llid, lrid, ulid, urid = grid_idx
        x0, x1, y0, y1 = grid_xy
        xi = np.clip(xi, x0, x1)
        yi = np.clip(yi, y0, y1)
        # x0 < xi < x1 (lambda)
        # y0 < yi < y1 (slit pos)
        t = (xi - x0) / (x1 - x0)
        u = (yi - y0) / (y1 - y0)
        
        psf_x0_y0 = self.psfs[llid]
        psf_x1_y0 = self.psfs[lrid]
        psf_x0_y1 = self.psfs[ulid]
        psf_x1_y1 = self.psfs[urid]
        
        # Pad to make sure all PSFs have the same size
        max_psf_size = self.max_psf_size
        psf_arr = []
        for _, psf in enumerate([psf_x0_y0, psf_x0_y1, psf_x1_y0, psf_x1_y1]):
            if psf.shape[0] < max_psf_size:
                # The PSFs are centered, so pad symmetrically (this is presumably by construction for the current libraries)
                pad_left = (max_psf_size - psf.shape[0]) // 2
                pad_right = max_psf_size - pad_left - psf.shape[0]
                psf = np.pad(psf, ((pad_left, pad_right), (pad_left, pad_right)), mode='constant', constant_values=0.)
            psf_arr.append(psf)
        
        psf_x0_y0, psf_x0_y1, psf_x1_y0, psf_x1_y1 = psf_arr
        epsf = (1-t)*(1-u) * psf_x0_y0 + t*(1-u) * psf_x1_y0 + t*u * psf_x1_y1 + (1-t)*u * psf_x0_y1
        
        epsf /= epsf.sum()

        return epsf
    
    def _sample_psf(self, epsf):
        """Bin up the ePSF by the oversampling factor to get the detector pixel scale."""
        psf_oversampling = int(self.meta["psf_oversampling"])
        if self.oversampling_x == 1 and self.oversampling_y == 1:
            if epsf.ndim != 2:
                raise ValueError(f"Expected 2D PSF array, got shape={epsf.shape!r}")
            # Pad to a multiple of oversampling so we can block-sum efficiently
            ny, nx = epsf.shape
            pad_y = (-ny) % psf_oversampling
            pad_x = (-nx) % psf_oversampling
            pad_y0, pad_y1 = pad_y // 2, pad_y - pad_y // 2
            pad_x0, pad_x1 = pad_x // 2, pad_x - pad_x // 2
            epsf_padded = np.pad(epsf, ((pad_y0, pad_y1), (pad_x0, pad_x1)), mode="constant", constant_values=0.0)
            new_ny = epsf_padded.shape[0] // psf_oversampling
            new_nx = epsf_padded.shape[1] // psf_oversampling
            # (new_ny, os, new_nx, os) -> (new_ny, new_nx)
            psf_sampled = epsf_padded.reshape(new_ny, psf_oversampling, new_nx, psf_oversampling).sum(axis=(1, 3))
            # Renormalize after downsampling
            psf_sum = psf_sampled.sum()
            if np.isfinite(psf_sum) and psf_sum > 0:
                psf_sampled /= psf_sum
            else:
                logger.warning("Downsampled PSF sum is invalid: %s", psf_sum)
        
        else:
            image_oversampling_x = int(self.oversampling_x)
            image_oversampling_y = int(self.oversampling_y)
            if image_oversampling_x == psf_oversampling and image_oversampling_y == psf_oversampling:
                return epsf
            if image_oversampling_x > psf_oversampling:
                raise ValueError(
                    f"Image oversampling_x factor {image_oversampling_x} is larger than PSF oversampling factor {psf_oversampling}; "
                    "upsampling PSFs requires interpolation and is not supported by block-sum resampling."
                )
            if image_oversampling_y > psf_oversampling:
                raise ValueError(
                    f"Image oversampling_y factor {image_oversampling_y} is larger than PSF oversampling factor {psf_oversampling}; "
                    "upsampling PSFs requires interpolation and is not supported by block-sum resampling."
                )
            # If the image oversampling is different from the PSF oversampling, we can downsample the PSF by the ratio of the oversampling factors
            # Not guaranteed to work if the oversampling factors are not integer multiples, so the program aborts
            elif image_oversampling_x < psf_oversampling and image_oversampling_y < psf_oversampling:
                if psf_oversampling % image_oversampling_x == 0 and psf_oversampling % image_oversampling_y == 0:
                    factor_x = psf_oversampling // image_oversampling_x
                    factor_y = psf_oversampling // image_oversampling_y
                    # Pad to a multiple of the downsampling factor to avoid reshape errors
                    ny, nx = epsf.shape
                    pad_y = (-ny) % factor_y
                    pad_x = (-nx) % factor_x
                    pad_y0, pad_y1 = pad_y // 2, pad_y - pad_y // 2
                    pad_x0, pad_x1 = pad_x // 2, pad_x - pad_x // 2
                    epsf_padded = np.pad(epsf, ((pad_y0, pad_y1), (pad_x0, pad_x1)), mode="constant", constant_values=0.0)
                    psf_sampled = self._downsample(epsf_padded, f=[factor_x, factor_y])
                    psf_sum = psf_sampled.sum()
                    if np.isfinite(psf_sum) and psf_sum > 0:
                        psf_sampled /= psf_sum
                    else:
                        logger.warning("Sampled PSF sum is invalid: %s", psf_sum)
                else:
                    raise ValueError(f"PSF oversampling factor {psf_oversampling} is not an integer multiple of image oversampling factors {image_oversampling_x} and {image_oversampling_y}, " \
                                     "cannot sample PSF to match image oversampling.")
        return psf_sampled
        
    def _ePSF(self, xi, yi):
        """Master function to get the effective PSF at the given input coordinates (xi, yi)."""
        epsf = self._psf_interp(xi, yi)
        epsf_sampled = self._sample_psf(epsf)
        psf_sum = epsf_sampled.sum()
        if (not np.isfinite(psf_sum)) or (psf_sum <= 0.):
            logger.warning(f"PSF at image pixel location ({xi}, {yi}) is invalid")
        return epsf_sampled
        
    def _oversample(self, img, f=None):
        """Oversample an input image by either the image oversampling factors or a custom factor f."""
        if f is None:
            oversampling_x = int(self.oversampling_x)
            oversampling_y = int(self.oversampling_y)
        else:
            oversampling_x = f[0]
            oversampling_y = f[1]
        logger.debug("Oversampling image by factor of %d in x direction and %s in y direction", oversampling_x, oversampling_y)
        if img.ndim == 3: # Flux density. TODO: add BUNIT check
            if oversampling_y == 1 and oversampling_x != 1:
                new_img = np.repeat(img, oversampling_x, axis=2) # x only
            elif oversampling_y != 1 and oversampling_x == 1:
                new_img = np.repeat(img, oversampling_y, axis=1) # y only
            elif oversampling_x != 1 and oversampling_y != 1:
                new_img = np.repeat(np.repeat(img, oversampling_y, axis=1), oversampling_x, axis=2)
            else:
                new_img = img
        elif img.ndim == 2: # Electrons. TODO: add BUNIT check
            if oversampling_y == 1 and oversampling_x != 1:
                oversampled_image = np.repeat(img, oversampling_x, axis=1) # x only
                new_img = oversampled_image / oversampling_x
            elif oversampling_y != 1 and oversampling_x == 1:
                oversampled_image = np.repeat(img, oversampling_y, axis=0) # y only
                new_img = oversampled_image / oversampling_y
            elif oversampling_x != 1 and oversampling_y != 1:
                oversampled_image = np.repeat(np.repeat(img, oversampling_y, axis=0), oversampling_x, axis=1)
                new_img = oversampled_image / (oversampling_x * oversampling_y)
            else:
                new_img = img

        # check flux conservation after oversampling + normalization
        img_sum = img.sum()
        new_sum = new_img.sum()
        if img.ndim == 3:
            new_sum /= (self.oversampling_x * self.oversampling_y)
        if np.isfinite(img_sum) and img_sum != 0:
            rel_diff = np.abs(img_sum - new_sum) / np.abs(img_sum)
            if rel_diff > self.meta["flux_accuracy"]:
                logger.warning("Flux is not conserved by oversampling: difference is %.2f%%", rel_diff * 100)
        return new_img
        
    def _downsample(self, img, f=None):
        """Downsample an input image by either the image oversampling factors or a custom factor f."""
        if f is None:
            oversampling_x = int(self.oversampling_x)
            oversampling_y = int(self.oversampling_y)
        else:
            oversampling_x = f[0]
            oversampling_y = f[1]
        if img.ndim == 3:
            n_lambda, n_y, n_x = img.shape
            if oversampling_y == 1 and oversampling_x != 1:
                new_n_x = n_x // oversampling_x
                downsampled_image = img.reshape(n_lambda, n_y, new_n_x, oversampling_x).sum(axis=3)
            elif oversampling_y != 1 and oversampling_x == 1:
                new_n_y = n_y // oversampling_y
                downsampled_image = img.reshape(n_lambda, new_n_y, oversampling_y, n_x).sum(axis=2)
            elif oversampling_y != 1 and oversampling_x != 1:
                new_n_y = n_y // oversampling_y
                new_n_x = n_x // oversampling_x
                downsampled_image = img.reshape(n_lambda, new_n_y, oversampling_y, new_n_x, oversampling_x).sum(axis=(2,4))
            else:
                downsampled_image = img
        elif img.ndim == 2:
            n_y, n_x = img.shape
            if oversampling_y == 1 and oversampling_x != 1:
                new_n_x = n_x // oversampling_x
                downsampled_image = img.reshape(n_y, new_n_x, oversampling_x).sum(axis=2)
            elif oversampling_y != 1 and oversampling_x == 1:
                new_n_y = n_y // oversampling_y
                downsampled_image = img.reshape(new_n_y, oversampling_y, n_x).sum(axis=1)
            elif oversampling_y != 1 and oversampling_x != 1:
                new_n_y = n_y // oversampling_y
                new_n_x = n_x // oversampling_x
                downsampled_image = img.reshape(new_n_y, oversampling_y, new_n_x, oversampling_x).sum(axis=(1,3))
            else:
                downsampled_image = img
        # check flux conservation after downsampling
        img_sum = img.sum()
        down_sum = downsampled_image.sum()
        if np.isfinite(img_sum) and img_sum != 0:
            rel_diff = np.abs(img_sum - down_sum) / np.abs(img_sum)
            if rel_diff > self.meta["flux_accuracy"]:
                logger.warning("Flux is not conserved by downsampling: difference is %.2f%%", rel_diff * 100)
        new_img = downsampled_image
        return new_img
    
class SlitPSF(GriddedPSF):
    z_order: ClassVar[tuple[int, ...]] = (225, 625)

    def __init__(self, **kwargs):
        """
        Initialize the SlitPSF effect (load the PSF library, set up grid for interpolation, etc.)
        Note: this currently assumes the input field coordinates are in deg.
        """
        super().__init__(**kwargs)
        arrs: list[np.ndarray] = []
        slit_positions: list[float] = []
        for psf_file in self.psf_lib:
            with fits.open(os.path.join(self.psf_dir, psf_file)) as hdul:
                arr = hdul[0].data / hdul[0].data.sum()
                arrs.append(arr)
                y_field = float(hdul[0].header['YFLD']) # deg
                x_field = float(hdul[0].header['XFLD']) # deg
                slit_positions.append(y_field)
        
        # Sort slit positions and PSF array so the PSFs lie on a regular grid
        sortidx = np.argsort(slit_positions, kind='stable')
        slit_positions = np.array(slit_positions)[sortidx]
        arrs = [arrs[i] for i in sortidx]
        
        # For use with our interpolator, we will copy the PSF arrays into a second dimension
        x_pos = np.array([-1.*u.arcsec.to(u.deg), 0., 1.*u.arcsec.to(u.deg)]) + self.fov_x0.to(u.deg).value
        grid_xypos: list[tuple[float, float]] = []
        for _, slit_pos in enumerate(slit_positions):
            for j in range(3):
                grid_xypos.append((x_pos[j], slit_pos))
        data = np.repeat(arrs, 3, axis=0)
        self.psfs = data
        self.grid_xypos = np.asarray(grid_xypos) # shape N x 2
        self.x_vals = self.grid_xypos[:,0]
        self.y_vals = self.grid_xypos[:,1]
        self.x_min, self.x_max = self.x_vals.min(), self.x_vals.max()
        self.y_min, self.y_max = self.y_vals.min(), self.y_vals.max()
        self.max_psf_size = max([psf.shape[0] for psf in self.psfs])
        
    def apply_to(self, obj, tile_size_x=32, tile_size_y=32, **kwargs):
        # 1. During setup of the FieldOfViews
        if isinstance(obj, FovVolumeList) and self._waveset is not None:
            logger.debug("Executing %s, FoV setup", self.meta['name'])
            waveset = self._waveset
            if len(waveset) != 0:
                waveset_edges = 0.5 * (waveset[:-1] + waveset[1:])
                obj.split("wave", quantify(waveset_edges, u.um).value)
           
        # 2. During observation (where the convolution happens)
        elif isinstance(obj, self.convolution_classes):
            logger.debug("UVEX LSS slit PSF convolution start")
            assert obj.hdu.data.ndim == 3, "Data dimensions should be 3D; check FOV creation and effect ordering." # not mapped to detector plane yet

            os_state = getattr(obj, "_oversampled", None)
            if self.oversampling_x != 1 or self.oversampling_y != 1:
                if os_state is None:
                    raise ValueError("Either oversampling_x or oversampling_y is greater than 1, but the Oversampling effect has not been applied to the image yet; aborting.")
            tile_size_x *= self.oversampling_x
            tile_size_y *= self.oversampling_y
            if tile_size_y > obj.hdu.data.shape[1] or tile_size_x > obj.hdu.data.shape[2]:
                logger.warning(f"Tile size {tile_size_y}*{tile_size_x} is larger than the current image dimensions ({obj.hdu.data.shape[1]}, {obj.hdu.data.shape[2]}), which may causee issues with convolution.")
            
            cube_wcs = WCS(obj.hdu.header)
            image = obj.hdu.data.astype(float)
            
            _, n_y, n_x = image.shape
            # Subtract background level before convolution and add back after
            bkg_level = get_bkg_level(image, self.meta["bkg_width"])
            if self.meta["bkg_width"] == 0:
                bkg_level = bkg_level[:, None, None]
            image -= bkg_level
            
            if n_y > n_x: # across slit (spectral) direction is n_x
                wcs_y = cube_wcs.sub([2])
                slit_y_img =  wcs_y.all_pix2world(np.arange(n_y), 0)[0] * u.Unit(wcs_y.wcs.cunit[0]).to(u.Unit(obj.hdu.header["CUNIT2"]))
                wcs_xi = cube_wcs.sub([1])
                xi_img = wcs_xi.all_pix2world(np.arange(n_x), 0)[0] * u.Unit(wcs_xi.wcs.cunit[0]).to(u.Unit(obj.hdu.header["CUNIT1"]))
                n_spec = n_x
                n_spat = n_y
                
            else: # spectral direction is n_y or second axis
                wcs_xi = cube_wcs.sub([2])
                xi_img =  wcs_xi.all_pix2world(np.arange(n_y), 0)[0] * u.Unit(wcs_xi.wcs.cunit[0]).to(u.Unit(obj.hdu.header["CUNIT2"]))
                wcs_y = cube_wcs.sub([1])
                slit_y_img = wcs_y.all_pix2world(np.arange(n_x), 0)[0] * u.Unit(wcs_y.wcs.cunit[0]).to(u.Unit(obj.hdu.header["CUNIT1"]))
                n_spec = n_y
                n_spat = n_x
            
            n_tiles_spec = n_spec // tile_size_x + (1 if n_spec % tile_size_x != 0 else 0)
            n_tiles_spat = n_spat // tile_size_y + (1 if n_spat % tile_size_y != 0 else 0)
            
            convolved_image = np.zeros_like(image)
            with tqdm(total=n_tiles_spec*n_tiles_spat, desc=" Slit PSF Convolution") as pbar:
                for x in range(n_tiles_spec):
                    for y in range(n_tiles_spat):
                        x0 = x * tile_size_x # tile start index
                        x1 = min((x+1)*tile_size_x, n_spec) # tile end in pixels (don't go outside the image)
                        y0 = y * tile_size_y
                        y1 = min((y+1)*tile_size_y, n_spat)

                        x_cen = min(x0 + (x1 - x0) // 2, n_spec - 1)
                        y_cen = min(y0 + (y1 - y0) // 2, n_spat - 1)
                            
                        # Corresponding field coordinates for the PSF center
                        x_fld0 = float(xi_img[x_cen])
                        y_fld0 = float(slit_y_img[y_cen])
                        # Clamp to PSF grid bounds if necessary, so tiles beyond the PSF grid will just get mapped to the edge PSFs
                        x_fld0 = np.clip(x_fld0, self.x_min, self.x_max)
                        y_fld0 = np.clip(y_fld0, self.y_min, self.y_max)
                        
                        # Get the effective PSF for the tile center
                        ePSF = self._ePSF(x_fld0, y_fld0)
                        
                        # Add wavelength axis to convolve all wavelength slices at once
                        kernel_3d = ePSF[None, :, :]
                        # Basic overlap add logic: zero pad the image tile by PSF size - 1 on each side to avoid edge effects in convolution
                        pad_y = ePSF.shape[0] - 1
                        pad_x = ePSF.shape[1] - 1
                        orig_tile = image[:, y*tile_size_y:(y+1)*tile_size_y, x*tile_size_x:(x+1)*tile_size_x]
                        
                        if orig_tile.shape[1] != tile_size_y or orig_tile.shape[2] != tile_size_x:
                            pad_x_orig = tile_size_x - orig_tile.shape[2]
                            pad_y_orig = tile_size_y - orig_tile.shape[1]
                            tile = np.pad(orig_tile, ((0, 0), (0, pad_y_orig), (0, pad_x_orig)), mode='constant', constant_values=0.)
                        else:
                            tile = orig_tile
                            
                        padded_image = np.pad(tile, ((0, 0), (pad_y, pad_y), (pad_x, pad_x)), mode='constant', constant_values=0.)
                        # Note that using fftconvolve here might not the most efficient route since there is some N-d convolution overhead
                        # however this handles kernel centering automatically as opposed to, e.g., doing the convolution in Fourier space with scipy.fft
                        convolved_image_ij = fftconvolve(padded_image, kernel_3d, mode='same')
                        
                        # Absolute detector image indices covered by the convolved patch
                        g_y0 = y0 - pad_y
                        g_y1 = y0 + tile_size_y + pad_y
                        g_x0 = x0 - pad_x
                        g_x1 = x0 + tile_size_x + pad_x
                        # Detector image indices trimmed to image bounds
                        cminy = max(0, g_y0)
                        cmaxy = min(n_spat, g_y1)
                        cminx = max(0, g_x0)
                        cmaxx = min(n_spec, g_x1)
                        # Convolved image tile indices
                        start_y = cminy - g_y0
                        end_y = start_y + (cmaxy - cminy)
                        start_x = cminx - g_x0
                        end_x = start_x + (cmaxx - cminx)

                        convolved_image_cen = convolved_image_ij[:, start_y:end_y, start_x:end_x]
                        convolved_image[:, cminy:cmaxy, cminx:cmaxx] += convolved_image_cen
                        
                        if y % tile_size_y == 0:
                            pbar.update(tile_size_y)
            
            img_sum = image.sum()
            conv_sum = convolved_image.sum()
            if np.isfinite(img_sum) and img_sum != 0:
                rel_diff = np.abs(img_sum - conv_sum) / np.abs(img_sum)
                if rel_diff > self.meta["flux_accuracy"]:
                    logger.warning("Flux is not conserved by slit PSF convolution: difference is %.2f%%",rel_diff * 100)        
             
            final_image = convolved_image + bkg_level

            if PLOT:
                import matplotlib.pyplot as plt
                plt.title("Image slice after SlitPSF")
                plt.imshow(final_image[final_image.shape[0] // 2, :, :])
                plt.show()
            
            obj.hdu.data = final_image
        return obj
            
class LSSDetectorPSF(GriddedPSF):
    z_order: ClassVar[tuple[int, ...]] = (273, 673)

    def __init__(self, **kwargs):
        """
        Initialize the LSSDetectorPSF effect (load the PSF library, set up grid for interpolation, etc.)
        Note: this currently assumes the input wavelengths are in nm, and field positions are in deg.
        """
        super().__init__(**kwargs)
        arrs: list[np.ndarray] = []
        positions: list[tuple[float, float]] = []
        for psf_file in self.psf_lib:
            with fits.open(os.path.join(self.psf_dir, psf_file)) as hdul:
                arr = hdul[0].data / hdul[0].data.sum() # normalize
                arrs.append(arr)
                lam = float(hdul[0].header['CEN_WAVE']) * 1e-3   # convert from nm to um
                y_field = float(hdul[0].header['YFLD']) * 3600.  # convert from deg to arcsec
                # x = wavelength, y = field position
                positions.append((lam, y_field))
        
        # Sort PSF grid by y field position, then by wavelength
        x, y = np.array(positions)[:,0], np.array(positions)[:,1]
        sortidx = np.lexsort((x, y))
        x, y = x[sortidx], y[sortidx]
        arrs = [arrs[i] for i in sortidx]
        positions = [positions[i] for i in sortidx]
        
        self.psfs = arrs
        self.grid_xypos = np.asarray(positions) # shape N x 2
        self.x_vals = self.grid_xypos[:,0]
        self.y_vals = self.grid_xypos[:,1]
        self.x_min, self.x_max = self.x_vals.min(), self.x_vals.max()
        self.y_min, self.y_max = self.y_vals.min(), self.y_vals.max()
        self.max_psf_size = max([psf.shape[0] for psf in self.psfs])
        
    def apply_to(self, obj, tile_size_x=32, tile_size_y=32, **kwargs): 
        # TODO: adaptive resolution based on position in the FOV? Only because the red end PSFs degrade rapidly
        # 1. During setup of the FieldOfViews
        if isinstance(obj, FovVolumeList) and self._waveset is not None:
            logger.debug("Executing %s, FoV setup", self.meta['name'])
            waveset = self._waveset
            if len(waveset) != 0:
                waveset_edges = 0.5 * (waveset[:-1] + waveset[1:])
                obj.split("wave", quantify(waveset_edges, u.um).value)
        
        # 2. During observe: convolution
        elif isinstance(obj, self.convolution_classes):
            logger.debug("UVEX LSS detector PSF convolution start")

            os_state = getattr(obj, "_oversampled", None)
            if self.oversampling_x != 1 or self.oversampling_y != 1:
                if os_state is None:
                    raise ValueError("Either oversampling_x or oversampling_y is greater than 1, but the Oversampling effect has not been applied to the image yet; aborting.")
            
            tile_size_x *= self.oversampling_x
            tile_size_y *= self.oversampling_y
            assert obj.hdu.data.ndim == 2, "Image should be mapped to detector plane but is not; check FOV creation." # should be mapped to the detector plane already
            if tile_size_y > obj.hdu.data.shape[0] or tile_size_x > obj.hdu.data.shape[1]:
                logger.warning(f"Tile size {tile_size_y}*{tile_size_x} is larger than the current image dimensions ({obj.hdu.data.shape[0]}, {obj.hdu.data.shape[1]}), which may cause issues with convolution.")
            
            image = obj.hdu.data.astype(float)
            xi_map = obj.hdu.xi_map
            lam_map = obj.hdu.lam_map

            ydim, xdim = image.shape
            # subtract background level before convolution and add back after
            bkg_level = get_bkg_level(image, self.meta["bkg_width"])
            image -= bkg_level

            # must be true or the logic below breaks
            assert xi_map.shape == image.shape
            assert lam_map.shape == image.shape
            
            convolved_image = np.zeros_like(image)
            
            # Add 1 if pixel extent does not perfectly divide tile size to capture partial tiles at the edges
            n_tiles_y = ydim // tile_size_y + (1 if ydim % tile_size_y != 0 else 0)
            n_tiles_x = xdim // tile_size_x + (1 if xdim % tile_size_x != 0 else 0)
            with tqdm(total=n_tiles_y*n_tiles_x, desc=" LSS Detector PSF Convolution") as pbar:
                for y in range(n_tiles_y):
                    for x in range(n_tiles_x):
                        y0 = y * tile_size_y # tile start in pixels (index into detector image)
                        y1 = min((y+1)*tile_size_y, ydim) # tile end in pixels (don't go outside the detector image)
                        x0 = x * tile_size_x
                        x1 = min((x+1)*tile_size_x, xdim)

                        y_cen = min(y0 + (y1-y0) // 2, ydim - 1)
                        x_cen = min(x0 + (x1-x0) // 2, xdim - 1)
                        
                        # Get the corresponding field/wavelength coordinates for the PSF
                        # Ensure center in wavelength and slit coords is within bounds, and clamp if not
                        # this effectively means tiles beyond the PSF grid will just get mapped to the edge PSFs
                        lam0 = float(lam_map[y_cen, x_cen])
                        xi0 = float(xi_map[y_cen, x_cen])
                        
                        lam0 = np.clip(lam0, self.x_min, self.x_max)
                        xi0 = np.clip(xi0, self.y_min, self.y_max)
                            
                        # Get the convolution kernel
                        ePSF = self._ePSF(lam0, xi0)
                        
                        # Basic overlap add logic: zero pad the image tile by PSF size - 1 on each side to avoid edge effects in convolution
                        pad_y = ePSF.shape[0] - 1
                        pad_x = ePSF.shape[1] - 1
                        orig_tile = image[y*tile_size_y:(y+1)*tile_size_y, x*tile_size_x:(x+1)*tile_size_x]
                        if orig_tile.shape[0] != tile_size_y or orig_tile.shape[1] != tile_size_x:
                            pad_x_orig = tile_size_x - orig_tile.shape[1]
                            pad_y_orig = tile_size_y - orig_tile.shape[0]
                            tile = np.pad(orig_tile, ((0, pad_y_orig), (0, pad_x_orig)), mode='constant', constant_values=0.)
                        else:
                            tile = orig_tile
                        
                        padded_image = np.pad(tile, ((pad_y, pad_y), (pad_x, pad_x)), mode='constant', constant_values=0.)
                        convolved_image_ij = fftconvolve(padded_image, ePSF, mode='same')
                        
                        # Absolute detector image indices covered by the convolved patch
                        g_y0 = y0 - pad_y
                        g_y1 = y0 + tile_size_y + pad_y
                        g_x0 = x0 - pad_x
                        g_x1 = x0 + tile_size_x + pad_x
                        # Detector image indices trimmed to image bounds
                        cminy = max(0, g_y0)
                        cmaxy = min(ydim, g_y1)
                        cminx = max(0, g_x0)
                        cmaxx = min(xdim, g_x1)
                        # Convolved image tile indices
                        start_y = cminy - g_y0
                        end_y = start_y + (cmaxy - cminy)
                        start_x = cminx - g_x0
                        end_x = start_x + (cmaxx - cminx)

                        convolved_image_cen = convolved_image_ij[start_y:end_y, start_x:end_x]
                        convolved_image[cminy:cmaxy, cminx:cmaxx] += convolved_image_cen
                        
                        if x % tile_size_x == 0:
                            pbar.update(tile_size_x)

            img_sum = image.sum()
            conv_sum = convolved_image.sum()
            if np.isfinite(img_sum) and img_sum != 0:
                rel_diff = np.abs(img_sum - conv_sum) / np.abs(img_sum)
                if rel_diff > self.meta["flux_accuracy"]:
                    logger.warning("Flux is not conserved by LSS detector PSF convolution: difference is %.2f%%",rel_diff * 100) 
            
            final_image = convolved_image + bkg_level
            obj.hdu.data = final_image

            if PLOT:
                import matplotlib.pyplot as plt
                plt.title("Image slice after DetectorPSF")
                plt.imshow(final_image[:, :])
                plt.show()
            
        return obj

class UVIMImagerPSF(GriddedPSF):
    """Spatially varying detector-plane PSF for the UVIM NUV/FUV imagers."""

    z_order: ClassVar[tuple[int, ...]] = (273, 673)

    def __init__(self, **kwargs):
        kwargs.setdefault("fov_unit", "arcsec")
        super().__init__(**kwargs)

        if self.psf_dir is None:
            raise FileNotFoundError("UVIM PSF directory could not be found.")
        if not self.psf_lib:
            raise FileNotFoundError(f"No PSF FITS files found in {self.psf_dir}")

        def _to_arcsec(value):
            if isinstance(value, u.Quantity):
                return value.to(u.arcsec)
            unit = u.Unit(self.meta.get("fov_unit", "arcsec"))
            return (float(value) * unit).to(u.arcsec)

        self.fov_x0 = _to_arcsec(self.meta.get("fov_x0", 0.0))
        self.fov_y0 = _to_arcsec(self.meta.get("fov_y0", 0.0))

        arrs, positions, xpos_det, ypos_det = [], [], [], []

        for psf_file in self.psf_lib:
            with fits.open(os.path.join(self.psf_dir, psf_file)) as hdul:
                arr = np.asarray(hdul[0].data, dtype=float)
                hdr = hdul[0].header

                if arr.ndim != 2:
                    raise ValueError(
                        f"{psf_file}: expected a 2D PSF, got shape {arr.shape}"
                    )

                arr_sum = np.nansum(arr)
                if not np.isfinite(arr_sum) or arr_sum <= 0:
                    raise ValueError(f"{psf_file}: invalid PSF sum {arr_sum}")

                arrs.append(arr / arr_sum)
                positions.append((
                    (float(hdr["XFLD"]) * u.deg).to_value(u.arcsec),
                    (float(hdr["YFLD"]) * u.deg).to_value(u.arcsec),
                ))
                xpos_det.append(float(hdr["XPOS"]))
                ypos_det.append(float(hdr["YPOS"]))

        if len(set(positions)) != len(positions):
            raise ValueError(
                "Duplicate XFLD/YFLD positions were found in the UVIM PSF library."
            )

        positions = np.asarray(positions)
        sortidx = np.lexsort((positions[:, 0], positions[:, 1]))

        self.psfs = [arrs[i] for i in sortidx]
        self.grid_xypos = positions[sortidx]
        self.xpos_det = np.asarray(xpos_det)[sortidx]
        self.ypos_det = np.asarray(ypos_det)[sortidx]

        self.x_vals = np.unique(self.grid_xypos[:, 0])
        self.y_vals = np.unique(self.grid_xypos[:, 1])
        self.x_min, self.x_max = self.x_vals.min(), self.x_vals.max()
        self.y_min, self.y_max = self.y_vals.min(), self.y_vals.max()
        self.max_psf_size = max(psf.shape[0] for psf in self.psfs)

        n_expected = len(self.x_vals) * len(self.y_vals)
        if n_expected != len(self.psfs):
            raise ValueError(
                f"Incomplete UVIM PSF grid in {self.psf_dir}: "
                f"{len(self.psfs)} PSFs found, but "
                f"{len(self.x_vals)} x-values × {len(self.y_vals)} y-values "
                f"= {n_expected}"
            )

    @staticmethod
    def _wcs_pixels_to_arcsec(sky_wcs, pixel_coordinates):
        """Convert image pixel coordinates to WCS coordinates in arcsec."""
        pixels = np.asarray(pixel_coordinates, dtype=float)
        world = sky_wcs.all_pix2world(pixels, 0)

        cunit_x = sky_wcs.wcs.cunit[0]
        cunit_y = sky_wcs.wcs.cunit[1]
        x_unit = u.arcsec if cunit_x is None or str(cunit_x) == "" else u.Unit(cunit_x)
        y_unit = u.arcsec if cunit_y is None or str(cunit_y) == "" else u.Unit(cunit_y)

        return np.column_stack((
            (world[:, 0] * x_unit).to_value(u.arcsec),
            (world[:, 1] * y_unit).to_value(u.arcsec),
        ))

    def apply_to(self, obj, tile_size_x=32, tile_size_y=32, **kwargs):
        # During FOV setup
        if isinstance(obj, FovVolumeList) and self._waveset is not None:
            logger.debug("Executing %s, FoV setup", self.meta["name"])
            if len(self._waveset) != 0:
                waveset_edges = 0.5 * (self._waveset[:-1] + self._waveset[1:])
                obj.split("wave", quantify(waveset_edges, u.um).value)

        # During observation: spatially varying PSF convolution
        elif isinstance(obj, self.convolution_classes):
            logger.debug("UVIM imager PSF convolution start")

            os_state = getattr(obj, "_oversampled", None)
            if (self.oversampling_x != 1 or self.oversampling_y != 1) and os_state is None:
                raise ValueError(
                    "oversampling_x or oversampling_y is greater than 1, but "
                    "ScopeSim's Oversampling effect has not yet been applied."
                )

            tile_size_x *= int(self.oversampling_x)
            tile_size_y *= int(self.oversampling_y)

            if obj.hdu.data.ndim != 2:
                raise ValueError(
                    f"UVIMImagerPSF expected a 2D image, "
                    f"but received shape {obj.hdu.data.shape}"
                )

            image = obj.hdu.data.astype(float)
            ydim, xdim = image.shape

            n_tiles_y = ydim // tile_size_y + (ydim % tile_size_y != 0)
            n_tiles_x = xdim // tile_size_x + (xdim % tile_size_x != 0)

            bkg_level = get_bkg_level(image, self.meta["bkg_width"])
            image -= bkg_level

            sky_wcs = WCS(obj.hdu.header)
            convolved_image = np.zeros_like(image)

            with tqdm(
                total=n_tiles_y * n_tiles_x,
                desc=" UVIM Imager PSF Convolution",
            ) as pbar:

                for y in range(n_tiles_y):
                    for x in range(n_tiles_x):
                        y0, x0 = y * tile_size_y, x * tile_size_x
                        y1 = min((y + 1) * tile_size_y, ydim)
                        x1 = min((x + 1) * tile_size_x, xdim)

                        y_cen = min(y0 + (y1 - y0) // 2, ydim - 1)
                        x_cen = min(x0 + (x1 - x0) // 2, xdim - 1)

                        x_fld0, y_fld0 = self._wcs_pixels_to_arcsec(
                            sky_wcs, [[x_cen, y_cen]]
                        )[0]

                        x_fld0 += self.fov_x0.to_value(u.arcsec)
                        y_fld0 += self.fov_y0.to_value(u.arcsec)

                        x_fld0 = np.clip(x_fld0, self.x_min, self.x_max)
                        y_fld0 = np.clip(y_fld0, self.y_min, self.y_max)

                        ePSF = self._ePSF(x_fld0, y_fld0)
                        pad_y, pad_x = ePSF.shape[0] - 1, ePSF.shape[1] - 1

                        orig_tile = image[y0:y1, x0:x1]

                        if orig_tile.shape != (tile_size_y, tile_size_x):
                            tile = np.pad(
                                orig_tile,
                                (
                                    (0, tile_size_y - orig_tile.shape[0]),
                                    (0, tile_size_x - orig_tile.shape[1]),
                                ),
                                mode="constant",
                                constant_values=0.0,
                            )
                        else:
                            tile = orig_tile

                        padded_tile = np.pad(
                            tile,
                            ((pad_y, pad_y), (pad_x, pad_x)),
                            mode="constant",
                            constant_values=0.0,
                        )
                        convolved_tile = fftconvolve(padded_tile, ePSF, mode="same")

                        g_y0, g_y1 = y0 - pad_y, y0 + tile_size_y + pad_y
                        g_x0, g_x1 = x0 - pad_x, x0 + tile_size_x + pad_x

                        cminy, cmaxy = max(0, g_y0), min(ydim, g_y1)
                        cminx, cmaxx = max(0, g_x0), min(xdim, g_x1)

                        start_y, start_x = cminy - g_y0, cminx - g_x0
                        end_y = start_y + (cmaxy - cminy)
                        end_x = start_x + (cmaxx - cminx)

                        convolved_image[cminy:cmaxy, cminx:cmaxx] += (
                            convolved_tile[start_y:end_y, start_x:end_x]
                        )

                        pbar.update(1)

            img_sum = image.sum()
            conv_sum = convolved_image.sum()

            if np.isfinite(img_sum) and img_sum != 0:
                rel_diff = np.abs(img_sum - conv_sum) / np.abs(img_sum)
                if rel_diff > self.meta["flux_accuracy"]:
                    logger.warning(
                        "Flux is not conserved by UVIM imager PSF convolution: "
                        "difference is %.2f%%",
                        rel_diff * 100,
                    )

            obj.hdu.data = convolved_image + bkg_level

        return obj
                
def find_directory(dir_name, search_root=irdb_path):
    """Find directory by name and return its absolute path."""
    if dir_name is None:
        return None # prevent search if no directory
    if os.path.isdir(dir_name):
        return os.path.abspath(dir_name) # check if input is already a valid directory
    for root, dirs, files in os.walk(search_root):
        if dir_name in dirs:
            return os.path.abspath(os.path.join(root, dir_name))
    return None
