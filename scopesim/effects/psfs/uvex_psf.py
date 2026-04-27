# -*- coding: utf-8 -*-
from typing import ClassVar

import numpy as np
import os
from tqdm import tqdm
from scipy.signal import fftconvolve

from astropy import units as u
from astropy.io import fits
from astropy.wcs import WCS

from ...optics.fov import FieldOfView
from ...optics.image_plane import ImagePlane
from ...optics.fov_volume_list import FovVolumeList
from ...utils import from_currsys, quantify
from ...rc import __search_path__
from . import logger
from ..effects import Effect
from .psf_base import get_bkg_level


class GriddedPSF(Effect):
    z_order: ClassVar[tuple[int, ...]] = (72, 672)
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.meta.update(kwargs)
        params = {
            "bkg_width": 0, # No background subtraction by default: see psf_base.get_bkg_level for details
            "flux_accuracy": 1e-4,
            "psf_oversampling": 10,
            "fov_x0": 3.5 # in deg, note that this is slightly redundant with fov_x_cen in UVIM_DET_LSS
        }
        self.meta.update(params)
        self.meta = from_currsys(self.meta, self.cmds)
        self.psf_dir = find_directory(self.meta.get("directory", None))
        self.psf_lib = self._load_psf_files()
        self.oversampling = self.meta.get("oversampling", 1)
        self.oversample_image_flag = self.meta.get("oversample_flag", False)
        self._waveset = []
        self.convolution_classes = (FieldOfView, ImagePlane)
        self.psfs: list[np.ndarray] | np.ndarray = None
        self.grid_xypos: np.ndarray = None
        self.x_vals = None
        self.y_vals = None
        self.x_min, self.x_max = None, None
        self.y_min, self.y_max = None, None

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
        llid, lrid, ulid, urid = self._calc_bounding_points(xi,yi)[0]
        x0, x1, y0, y1 = self._calc_bounding_points(xi,yi)[1]
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
        # we assume the PSF centers are all aligned at roughly the same pixel and that the PSF shapes are square
        max_psf_size = max(psf_x0_y0.shape[0], psf_x0_y1.shape[0], psf_x1_y0.shape[0], psf_x1_y1.shape[0])
        psf_arr = []
        for _, psf in enumerate([psf_x0_y0, psf_x0_y1, psf_x1_y0, psf_x1_y1]):
            if psf.shape[0] < max_psf_size:
                psf = np.pad(psf, ((0, max_psf_size - psf.shape[0]), (0, max_psf_size - psf.shape[1])), mode='constant', constant_values=(0., 0.))
            psf_arr.append(psf)
        
        psf_x0_y0, psf_x0_y1, psf_x1_y0, psf_x1_y1 = psf_arr
        epsf = (1-t)*(1-u) * psf_x0_y0 + t*(1-u) * psf_x1_y0 + t*u * psf_x1_y1 + (1-t)*u * psf_x0_y1
        return epsf
    
    def _sample_psf(self, epsf):
        """Bin up the ePSF by the oversampling factor to get the detector pixel scale."""
        psf_oversampling = int(self.meta["psf_oversampling"])
        if not self.oversample_image_flag:
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
            # Block-sum: (new_ny, os, new_nx, os) -> (new_ny, new_nx)
            psf_sampled = epsf_padded.reshape(new_ny, psf_oversampling, new_nx, psf_oversampling).sum(axis=(1, 3))
            # Renormalize after downsampling
            psf_sum = psf_sampled.sum()
            if np.isfinite(psf_sum) and psf_sum > 0:
                psf_sampled /= psf_sum
            else:
                logger.warning("Downsampled PSF sum is invalid: %s", psf_sum)
        
        else:
            image_oversampling = int(self.oversampling)
            if image_oversampling == psf_oversampling:
                return epsf
            if image_oversampling > psf_oversampling:
                raise ValueError(
                    f"Image oversampling factor {image_oversampling} is larger than PSF oversampling factor {psf_oversampling}; "
                    "upsampling PSFs requires interpolation and is not supported by block-sum resampling."
                )
            # If the image oversampling is different from the PSF oversampling, we can downsample the PSF by the ratio of the oversampling factors
            # Not guaranteed to work if the oversampling factors are not integer multiples, so the program aborts
            elif image_oversampling < psf_oversampling:
                if psf_oversampling % image_oversampling == 0:
                    factor = psf_oversampling // image_oversampling
                    # Pad to a multiple of the downsampling factor to avoid reshape errors
                    ny, nx = epsf.shape
                    pad_y = (-ny) % factor
                    pad_x = (-nx) % factor
                    pad_y0, pad_y1 = pad_y // 2, pad_y - pad_y // 2
                    pad_x0, pad_x1 = pad_x // 2, pad_x - pad_x // 2
                    epsf_padded = np.pad(epsf, ((pad_y0, pad_y1), (pad_x0, pad_x1)), mode="constant", constant_values=0.0)
                    psf_sampled = self._downsample(epsf_padded, f=factor)
                    psf_sum = psf_sampled.sum()
                    if np.isfinite(psf_sum) and psf_sum > 0:
                        psf_sampled /= psf_sum
                    else:
                        logger.warning("Sampled PSF sum is invalid: %s", psf_sum)
                else:
                    raise ValueError(f"PSF oversampling factor {psf_oversampling} is not an integer multiple of image oversampling factor {image_oversampling}, cannot sample PSF to match image oversampling.")
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
        if f is None:
            oversampling = int(self.oversampling)
        else:
            oversampling = int(f)
        logger.debug("Oversampling image by factor of %d", oversampling)
        if img.ndim == 3: # not mapped to detector plane yet
            oversampled_image = np.repeat(np.repeat(img, oversampling, axis=1), oversampling, axis=2)
        elif img.ndim == 2: # already mapped to detector plane
            oversampled_image = np.repeat(np.repeat(img, oversampling, axis=0), oversampling, axis=1)
        new_img = oversampled_image / oversampling**2 # conserve flux
        # check flux conservation after oversampling + normalization
        img_sum = img.sum()
        new_sum = new_img.sum()
        if np.isfinite(img_sum) and img_sum != 0:
            rel_diff = np.abs(img_sum - new_sum) / np.abs(img_sum)
            if rel_diff > self.meta["flux_accuracy"]:
                logger.warning("Flux is not conserved by oversampling: difference is %.2f%%", rel_diff * 100)
        return new_img
        
    def _downsample(self, img, f=None):
        if f is None:
            oversampling = int(self.oversampling)
        else:
            oversampling = int(f)
        if img.ndim == 3: # not mapped to detector plane yet
            n_lambda, n_y, n_x = img.shape
            new_n_y = n_y // oversampling
            new_n_x = n_x // oversampling
            downsampled_image = img.reshape(n_lambda, new_n_y, oversampling, new_n_x, oversampling).sum(axis=(2,4))
        elif img.ndim == 2: # already mapped to detector plane
            n_y, n_x = img.shape
            new_n_y = n_y // oversampling
            new_n_x = n_x // oversampling
            downsampled_image = img.reshape(new_n_y, oversampling, new_n_x, oversampling).sum(axis=(1,3))
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
    z_order: ClassVar[tuple[int, ...]] = (231, 631)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # For use with our interpolator, we will copy the PSF arrays into a second dimension
        arrs: list[np.ndarray] = []
        slit_positions: list[float] = []
        for psf_file in sorted(self.psf_lib):
            with fits.open(os.path.join(self.psf_dir, psf_file)) as hdul:
                arr = hdul[0].data / hdul[0].data.sum()
                arrs.append(arr)
                y_field = float(hdul[0].header['YFLD']) # deg
                x_field = float(hdul[0].header['XFLD']) # deg
                slit_positions.append(y_field)
        
        x_pos = np.array([-1.*u.arcsec.to(u.deg), 0., 1.*u.arcsec.to(u.deg)]) + self.meta["fov_x0"]
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
        
    def apply_to(self, obj, tile_size=32, **kwargs):
        # 1. During setup of the FieldOfViews
        if isinstance(obj, FovVolumeList) and self._waveset is not None:
            logger.debug("Executing %s, FoV setup", self.meta['name'])
            waveset = self._waveset
            if len(waveset) != 0:
                waveset_edges = 0.5 * (waveset[:-1] + waveset[1:])
                obj.split("wave", quantify(waveset_edges, u.um).value)
           
        if isinstance(obj, self.convolution_classes):
            logger.debug("UVEX LSS slit PSF convolution start")
            assert obj.hdu.data.ndim == 3 # not mapped to detector plane yet
            if tile_size > obj.hdu.data.shape[1] or tile_size > obj.hdu.data.shape[2]:
                logger.warning(f"Tile size {tile_size} is larger than the current image dimensions ({obj.hdu.data.shape[1]}, {obj.hdu.data.shape[2]}), which may causee issues with convolution.")
            
            cube_wcs = WCS(obj.hdu.header)
            
            if self.oversample_image_flag:
                image = self._oversample(obj.hdu.data.astype(np.float32))
                tile_size *= int(self.oversampling)
            else: 
                image = obj.hdu.data.astype(float)
            
            n_lambda, n_y, n_x = image.shape
            # subtract background level before convolution and add back after
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
            
            n_tiles_spec = n_spec // tile_size + (1 if n_spec % tile_size != 0 else 0)
            n_tiles_spat = n_spat // tile_size + (1 if n_spat % tile_size != 0 else 0)
            
            convolved_image = np.zeros_like(image)
            with tqdm(desc=" Slit PSF Convolution") as pbar:
                for l in range(n_lambda):
                    for x in range(n_tiles_spec):
                        for y in range(n_tiles_spat):
                            x0 = x * tile_size # tile start index
                            x1 = min((x+1)*tile_size, n_spec) # tile end in pixels (don't go outside the image)
                            y0 = y * tile_size
                            y1 = min((y+1)*tile_size, n_spat)

                            x_cen = min(x0 + (x1 - x0) // 2, n_spec - 1)
                            y_cen = min(y0 + (y1 - y0) // 2, n_spat - 1)
                            
                            # Corresponding field coordinates for the PSF center
                            x_fld0 = float(xi_img[x_cen])
                            y_fld0 = float(slit_y_img[y_cen])
                            # Clamp to PSF grid bounds if necessary, so tiles beyond the PSF grid will just get mapped to the edge PSFs
                            x_fld0 = np.clip(x_fld0, self.x_min, self.x_max)
                            y_fld0 = np.clip(y_fld0, self.y_min, self.y_max)
                        
                            # Get the effective PSF convolution kernel for this tile
                            ePSF = self._ePSF(x_fld0, y_fld0)
                            
                            # Basic overlap add logic: zero pad the image tile by PSF size - 1 on each side to avoid edge effects in convolution
                            pad_y = ePSF.shape[0] - 1
                            pad_x = ePSF.shape[1] - 1
                            orig_tile = image[l, y*tile_size:(y+1)*tile_size, x*tile_size:(x+1)*tile_size]
                        
                            if orig_tile.shape[0] != tile_size or orig_tile.shape[1] != tile_size:
                                pad_x_orig = tile_size - orig_tile.shape[1]
                                pad_y_orig = tile_size - orig_tile.shape[0]
                                tile = np.pad(orig_tile, ((0, pad_y_orig), (0, pad_x_orig)), mode='constant', constant_values=0.)
                            else:
                                tile = orig_tile
                            
                            padded_image = np.pad(tile, ((pad_y, pad_y), (pad_x, pad_x)), mode='constant', constant_values=((0.,0.), (0.,0.)))
                            convolved_image_ij = fftconvolve(padded_image, ePSF, mode='same')
                            
                            # Absolute detector image indices covered by the convolved patch
                            g_y0 = y0 - pad_y
                            g_y1 = y0 + tile_size + pad_y
                            g_x0 = x0 - pad_x
                            g_x1 = x0 + tile_size + pad_x
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

                            convolved_image_cen = convolved_image_ij[start_y:end_y, start_x:end_x]
                            convolved_image[l, cminy:cmaxy, cminx:cmaxx] += convolved_image_cen
                        
                    pbar.update(x*n_tiles_spat)
            if (np.abs(image.sum()-convolved_image.sum())/image.sum()) > self.meta["flux_accuracy"]:
                logger.warning(f"Flux is not conserved by LSS slit PSF convolution: difference is {np.abs(image.sum()-convolved_image.sum())/image.sum()*100:.2f}%")
            
            if self.oversample_image_flag:
                final_image = self._downsample(convolved_image + bkg_level)
            else:
                final_image = convolved_image + bkg_level
            obj.hdu.data = final_image
        return obj
            
class LSSDetectorPSF(GriddedPSF):
    z_order: ClassVar[tuple[int, ...]] = (273, 673)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        """Note: this currently assumes the input wavelengths are in nm,
        and field positions are in deg."""
        arrs: list[np.ndarray] = []
        positions: list[tuple[float, float]] = []
        for psf_file in sorted(self.psf_lib):
            with fits.open(os.path.join(self.psf_dir, psf_file)) as hdul:
                arr = hdul[0].data / hdul[0].data.sum() # normalize
                arrs.append(arr)
                lam = float(hdul[0].header['CEN_WAVE']) * 1e-3   # convert from nm to um
                y_field = float(hdul[0].header['YFLD']) * 3600.  # convert from deg to arcsec
                # x = wavelength, y = field position
                positions.append((lam, y_field))
        
        self.psfs = arrs
        self.grid_xypos = np.asarray(positions) # shape N x 2
        self.x_vals = self.grid_xypos[:,0]
        self.y_vals = self.grid_xypos[:,1]
        self.x_min, self.x_max = self.x_vals.min(), self.x_vals.max()
        self.y_min, self.y_max = self.y_vals.min(), self.y_vals.max()
        
    def apply_to(self, obj, tile_size=32, **kwargs):
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
            assert obj.hdu.data.ndim == 2 # should be mapped to the detector plane already
            if tile_size > obj.hdu.data.shape[0] or tile_size > obj.hdu.data.shape[1]:
                logger.warning(f"Tile size {tile_size} is larger than the current image dimensions ({obj.hdu.data.shape[0]}, {obj.hdu.data.shape[1]}), which may cause issues with convolution.")
            
            # If oversampling is enabled, oversample the maps and use tile size in oversampled pixels
            if self.oversample_image_flag:
                oversampling = int(self.oversampling)
                image = self._oversample(obj.hdu.data.astype(np.float32))
                ydim, xdim = image.shape
                tile_size *= oversampling
                xi_map = np.repeat(np.repeat(obj.hdu.xi_map, oversampling, axis=0), oversampling, axis=1)
                lam_map = np.repeat(np.repeat(obj.hdu.lam_map, oversampling, axis=0), oversampling, axis=1)
            else:
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
            n_tiles_y = ydim // tile_size + (1 if ydim % tile_size != 0 else 0)
            n_tiles_x = xdim // tile_size + (1 if xdim % tile_size != 0 else 0)
            with tqdm(desc=" LSS Detector PSF Convolution") as pbar:
                for y in range(n_tiles_y):
                    for x in range(n_tiles_x):
                        y0 = y * tile_size # tile start in pixels (index into detector image)
                        y1 = min((y+1)*tile_size, ydim) # tile end in pixels (don't go outside the detector image)
                        x0 = x * tile_size
                        x1 = min((x+1)*tile_size, xdim)

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
                        orig_tile = image[y*tile_size:(y+1)*tile_size, x*tile_size:(x+1)*tile_size]
                        if orig_tile.shape[0] != tile_size or orig_tile.shape[1] != tile_size:
                            pad_x_orig = tile_size - orig_tile.shape[1]
                            pad_y_orig = tile_size - orig_tile.shape[0]
                            tile = np.pad(orig_tile, ((0, pad_y_orig), (0, pad_x_orig)), mode='constant', constant_values=0.)
                        else:
                            tile = orig_tile
                        
                        padded_image = np.pad(tile, ((pad_y, pad_y), (pad_x, pad_x)), mode='constant', constant_values=((0.,0.), (0.,0.)))
                        convolved_image_ij = fftconvolve(padded_image, ePSF, mode='same')
                        
                        # Absolute detector image indices covered by the convolved patch
                        g_y0 = y0 - pad_y
                        g_y1 = y0 + tile_size + pad_y
                        g_x0 = x0 - pad_x
                        g_x1 = x0 + tile_size + pad_x
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
                        
                    pbar.update(y*n_tiles_x)

            img_sum = image.sum()
            conv_sum = convolved_image.sum()
            if np.isfinite(img_sum) and img_sum != 0:
                rel_diff = np.abs(img_sum - conv_sum) / np.abs(img_sum)
                if rel_diff > self.meta["flux_accuracy"]:
                    logger.warning("Flux is not conserved by LSS detector PSF convolution: difference is %.2f%%",rel_diff * 100) 
            
            if self.oversample_image_flag:
                final_image = self._downsample(convolved_image + bkg_level)
            else:
                final_image = convolved_image + bkg_level
            obj.hdu.data = final_image
            
        return obj
        
def find_directory(dir_name, search_root="."):
    """Find directory by name and return its absolute path."""
    if dir_name is None:
        return None
    # check if the input path is already a valid directory
    if os.path.isdir(dir_name):
        return os.path.abspath(dir_name)
    for root, dirs, files in os.walk(search_root):
        if dir_name in dirs:
            return os.path.abspath(os.path.join(root, dir_name))
    return None