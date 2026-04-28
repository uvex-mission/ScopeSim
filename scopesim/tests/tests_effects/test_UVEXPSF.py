"""Some little tests of the UVEX PSF module."""
import pytest
from pytest import approx

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import fftconvolve
import os
from pathlib import Path
from astropy import units as u
from astropy.io import fits

from scopesim.effects.psfs.uvex_psf import GriddedPSF, LSSDetectorPSF, SlitPSF

PLOTS = True

# get the full paths to the directories containing Jason Fucik's PSF files
TEST_ROOT = Path(__file__).resolve().parents[4]
LSS_DET_PSF_DIR = TEST_ROOT / "irdb" / "UVEX" / "code" / "inputs" / "LSS_DET_PSF"
LSS_SLIT_PSF_DIR = TEST_ROOT / "irdb" / "UVEX" / "code" / "inputs" / "LSS_SLIT_PSF"

FLUX_ACCURACY = 1e-4

def monochromatic_point_source(ntiles=3, npix=64):
    # delta function in wavelength, point source spatially
    n = ntiles*npix
    img = np.zeros((n, n))
    img[n // 2, n // 2] = 1.
    return img

def uniform_point_source(ntiles=3, npix=64, nlambda=2):
    # point source spatially, uniform in wavelength
    n = ntiles*npix
    img = np.zeros((nlambda, n, n))
    img[:, n // 2, n // 2] = 1.
    return img

class TestGriddedPSFInit:
    def test_basic_init(self):
        kwargs = {
            "directory" : "some_little_directory",
            "oversampling" : 2,
            "oversample_flag": True
        }
        gpsf = GriddedPSF(**kwargs)
        assert isinstance(gpsf, GriddedPSF), \
            f"Expected instance of GriddedPSF but got {type(gpsf)}"
        assert gpsf["#directory"] == "some_little_directory", \
            f"Expected directory to be 'some_little_directory' but got {gpsf['#directory']}"
        assert gpsf["#oversampling"] == 2, \
            f"Expected oversampling to be 2 but got {gpsf['#oversampling']}"
        assert gpsf["#oversample_flag"] == True, \
            f"Expected oversample_flag to be True but got {gpsf['#oversample_flag']}"
        
class TestInterpolator:
    """
    Test the PSF interpolator for both slit and detector plane PSFs.
    
    At a given location which coincides with a point on the grid of PSFs, 
    the effective PSF returned by the interpolator should match the PSF from file.
    """
    def test_ePSF_slit(self):
        xi, yi = 3.5, 0. # in deg
        kwargs = {
            "directory" : str(LSS_SLIT_PSF_DIR),
            "oversampling" : 10, # don't downsample the PSF for this
            "oversample_flag": True
        }
        spsf = SlitPSF(**kwargs)
        epsf = spsf._ePSF(xi, yi)
        
        # now load the true PSF at this location
        true_psf_file = "UVEX_SLIT_PSF_1um_F006.fits"
        # just to check...
        with fits.open(os.path.join(spsf.psf_dir, true_psf_file)) as hdul:
            yfld = hdul[0].header['YFLD']
            xfld = hdul[0].header['XFLD']
            assert xfld == xi, f"Expected XFLD to be {xi} but got {xfld}"
            assert yfld == yi, f"Expected YFLD to be {yi} but got {yfld}"
            true_psf = hdul[0].data / hdul[0].data.sum()
        
        diff = np.abs(true_psf - epsf)
        assert diff == approx(0), \
            f"Effective PSF does not match true PSF, differences range from {diff.min()} to {diff.max()}"
        
    def test_ePSF_LSS_det(self):
        lam0, xi0 = 0.164, 0. # um, arcsec
        kwargs = {
            "directory" : str(LSS_DET_PSF_DIR),
            "oversampling" : 10, # don't downsample the PSF for this
            "oversample_flag": True
        }
        dpsf = LSSDetectorPSF(**kwargs)
        epsf = dpsf._ePSF(lam0, xi0)
        
        true_psf_file = "UVEX_LSS_PSF_1um_F131.fits"
        with fits.open(os.path.join(dpsf.psf_dir, true_psf_file)) as hdul:
            yfld = hdul[0].header['YFLD'] * 3600.
            cenwave = hdul[0].header['CEN_WAVE'] * 1e-3
            assert cenwave == lam0, f"Expected CEN_WAVE to be {lam0} but got {cenwave}"
            assert yfld == xi0, f"Expected YFLD to be {xi0} but got {yfld}"
            true_psf = hdul[0].data / hdul[0].data.sum()
        
        diff = np.abs(true_psf - epsf)
        assert diff == approx(0), \
            f"Effective PSF does not match true PSF, differences range from {diff.min()} to {diff.max()}"

class TestApplyTo:
    """
    Unit test of the apply_to method for both slit and detector plane PSFs.
    Presently only tests convolution relevant to the LSS mode.
    
    For a monochromatic point source image, the output image after convolution
    with the PSF should be identical to the PSF.
    """
    def test_applyto_slit(self):
        dir = str(LSS_SLIT_PSF_DIR)
        xi, yi = 3.5, 0.
        true_psf_file = "UVEX_SLIT_PSF_1um_F006.fits"
        x_id_to_fld = np.array([-1.*u.arcsec.to(u.deg), 0., 1.*u.arcsec.to(u.deg)]) + xi
        y_id_to_fld = np.array([0.1, 0.0, -0.1]) + yi
            
        kwargs = {
            "directory" : dir,
            "oversampling" : 10, # don't downsample the PSF for this
            "oversample_flag": True
        }
        
        psf_class = SlitPSF(**kwargs)
        with fits.open(os.path.join(psf_class.psf_dir, true_psf_file)) as hdul:
            yfld = hdul[0].header['YFLD']
            xfld = hdul[0].header['XFLD']
            assert xfld == xi, f"Expected XFLD to be {xi} but got {xfld}"
            assert yfld == yi, f"Expected YFLD to be {yi} but got {yfld}"
            true_psf = hdul[0].data / hdul[0].data.sum()
        
        tile_size = 64
        img = uniform_point_source()
        _, n_spec, n_spat = img.shape
        n_tiles_spec = n_spec // tile_size + (1 if n_spec % tile_size != 0 else 0)
        n_tiles_spat = n_spat // tile_size + (1 if n_spat % tile_size != 0 else 0)
        convolved_image = np.zeros_like(img)
        for x in range(n_tiles_spec):
            for y in range(n_tiles_spat):
                x0 = x * tile_size # tile start index
                y0 = y * tile_size
                
                # Corresponding field coordinates for the PSF center
                x_fld0 = float(x_id_to_fld[x])
                y_fld0 = float(y_id_to_fld[y])
                
                # Clamp to PSF grid bounds if necessary, so tiles beyond the PSF grid will just get mapped to the edge PSFs
                x_fld0 = np.clip(x_fld0, psf_class.x_min, psf_class.x_max)
                y_fld0 = np.clip(y_fld0, psf_class.y_min, psf_class.y_max)
                        
                # Get the effective PSF convolution kernel for this tile
                ePSF = psf_class._ePSF(x_fld0, y_fld0)
                            
                # Basic overlap add logic: zero pad the image tile by PSF size - 1 on each side to avoid edge effects in convolution
                pad_y = ePSF.shape[0] - 1
                pad_x = ePSF.shape[1] - 1
                orig_tile = img[:, y*tile_size:(y+1)*tile_size, x*tile_size:(x+1)*tile_size]
                        
                if orig_tile.shape[1] != tile_size or orig_tile.shape[2] != tile_size:
                    pad_x_orig = tile_size - orig_tile.shape[2]
                    pad_y_orig = tile_size - orig_tile.shape[1]
                    tile = np.pad(orig_tile, ((0, 0), (0, pad_y_orig), (0, pad_x_orig)), mode='constant', constant_values=0.)
                else:
                    tile = orig_tile
                            
                padded_image = np.pad(tile, ((0, 0), (pad_y, pad_y), (pad_x, pad_x)), mode='constant', constant_values=0.)
                kernel_3d = ePSF[None, :, :] 
                convolved_image_ij = fftconvolve(padded_image, kernel_3d, mode='same')
                         
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

                convolved_image_cen = convolved_image_ij[:, start_y:end_y, start_x:end_x]
                convolved_image[:, cminy:cmaxy, cminx:cmaxx] += convolved_image_cen
        
        assert np.abs(img.sum()-convolved_image.sum())/img.sum() < FLUX_ACCURACY, \
            f"Flux not conserved to within {FLUX_ACCURACY*100}%"
        
        if PLOTS:
            # plot the convolved image and the x0 = 3.5 deg, y0 = 0 deg PSF
            title = f"True PSF along slit (x={xi} deg, y={yi} deg)"
            plt.figure(figsize=(8,8))
            plt.title(title)
            plt.imshow(true_psf, origin="lower")
            plt.colorbar()
            plt.show()
            
            # extract just the central tile of the convolved image for one wavelength slice
            plt.figure(figsize=(8,8))
            title = f"Image convolved with PSF along slit \n (x={xi} deg, y={yi} deg)"
            ycen, xcen = n_spat // 2, n_spec // 2
            plt.title(title)
            plt.imshow(convolved_image[0, ycen-tile_size//2:ycen+tile_size//2, xcen-tile_size//2:xcen+tile_size//2], origin="lower")
            plt.colorbar()
            plt.show()
        
    
    def test_applyto_LSS_det(self):
        dir = str(LSS_DET_PSF_DIR)
        xi, yi = 0.164, 0. # um, arcsec
        true_psf_file = "UVEX_LSS_PSF_1um_F131.fits"
        x_id_to_fld = np.array([-0.01, 0., 0.01]) + xi
        y_id_to_fld = np.array([0.1*u.deg.to(u.arcsec), 0.0*u.deg.to(u.arcsec), -0.1*u.deg.to(u.arcsec)]) + yi
        
        kwargs = {
            "directory" : dir,
            "oversampling" : 10, # don't downsample the PSF for this
            "oversample_flag": True
        }
        
        psf_class = LSSDetectorPSF(**kwargs)
        with fits.open(os.path.join(psf_class.psf_dir, true_psf_file)) as hdul:
            yfld = hdul[0].header['YFLD'] * 3600.
            cenwave = hdul[0].header['CEN_WAVE'] * 1e-3
            assert cenwave == xi, f"Expected CEN_WAVE to be {xi} but got {cenwave}"
            assert yfld == yi, f"Expected YFLD to be {yi} but got {yfld}"
            true_psf = hdul[0].data / hdul[0].data.sum()
            
        tile_size = 64
        img = monochromatic_point_source()
        n_spec, n_spat = img.shape
        n_tiles_spec = n_spec // tile_size + (1 if n_spec % tile_size != 0 else 0)
        n_tiles_spat = n_spat // tile_size + (1 if n_spat % tile_size != 0 else 0)
        convolved_image = np.zeros_like(img)
        for x in range(n_tiles_spec):
            for y in range(n_tiles_spat):
                x0 = x * tile_size # tile start index
                y0 = y * tile_size
                
                # Corresponding field coordinates for the PSF center
                x_fld0 = float(x_id_to_fld[x])
                y_fld0 = float(y_id_to_fld[y])
                
                # Clamp to PSF grid bounds if necessary, so tiles beyond the PSF grid will just get mapped to the edge PSFs
                x_fld0 = np.clip(x_fld0, psf_class.x_min, psf_class.x_max)
                y_fld0 = np.clip(y_fld0, psf_class.y_min, psf_class.y_max)
                        
                # Get the effective PSF convolution kernel for this tile
                ePSF = psf_class._ePSF(x_fld0, y_fld0)
                            
                # Basic overlap add logic: zero pad the image tile by PSF size - 1 on each side to avoid edge effects in convolution
                pad_y = ePSF.shape[0] - 1
                pad_x = ePSF.shape[1] - 1
                orig_tile = img[y*tile_size:(y+1)*tile_size, x*tile_size:(x+1)*tile_size]
                        
                if orig_tile.shape[0] != tile_size or orig_tile.shape[1] != tile_size:
                    pad_x_orig = tile_size - orig_tile.shape[1]
                    pad_y_orig = tile_size - orig_tile.shape[0]
                    tile = np.pad(orig_tile, ((0, pad_y_orig), (0, pad_x_orig)), mode='constant', constant_values=0.)
                else:
                    tile = orig_tile
                            
                padded_image = np.pad(tile, ((pad_y, pad_y), (pad_x, pad_x)), mode='constant', constant_values=0.)
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
                convolved_image[cminy:cmaxy, cminx:cmaxx] += convolved_image_cen
        
        assert np.abs(img.sum()-convolved_image.sum())/img.sum() < FLUX_ACCURACY, \
            f"Flux not conserved to within {FLUX_ACCURACY*100}%"
        
        if PLOTS:
            # plot the convolved image and the x0 = 3.5 deg, y0 = 0 deg PSF
            title = f"True PSF at LSS detector plane \n (lambda={xi} um, y={yi} arcsec)"
            plt.figure(figsize=(8,8))
            plt.title(title)
            plt.imshow(true_psf, origin="lower")
            plt.colorbar()
            plt.show()
            
            # extract just the central tile of the convolved image
            plt.figure(figsize=(8,8))
            title = f"Image convolved with PSF at LSS detector plane \n (lambda={xi} um, y={yi} arcsec)"
            ycen, xcen = n_spat // 2, n_spec // 2
            plt.title(title)
            plt.imshow(convolved_image[ycen-tile_size//2:ycen+tile_size//2, xcen-tile_size//2:xcen+tile_size//2], origin="lower")
            plt.colorbar()
            plt.show()