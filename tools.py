import numpy as np
import matplotlib.pyplot as plt
import math as m

from tqdm import tqdm
from scipy.interpolate import griddata, interp1d
from sklearn.preprocessing import scale

def band_image(frequency_band, electrodes_location, img_size=32):
    locs_2d = elec_proj( electrodes_location )
    frequency_band = frequency_band / np.min( frequency_band )
    frequency_band = frequency_band.reshape( (frequency_band.shape[0], -1) )

    images = image_generation( frequency_band, locs_2d, img_size )
    return images

def elec_proj(loc_3d):
    '''
        Take the 3D locations of all the electrodes and return their azimuthal projections.
    '''
    locs_2d = []
    for l in loc_3d:
        locs_2d.append( azim_proj( l ) )
    return np.asarray( locs_2d )

def azim_proj(pos):
    '''
        Take a point in 3D cartesian coordinates and return its 2D azimuthal projection.
    '''
    [r, elev, az] = cart2sph( pos[0], pos[1], pos[2] )

    return pol2cart( az, m.pi / 2 - elev )

def pol2cart(theta, rho):
    '''
        Take a point in 2D polar coordinate and return its 2D cartesian coordinates.
    '''
    return rho * m.cos( theta ), rho * m.sin( theta )

def cart2sph(x, y, z):
    '''
        Take a 3D point in cartesian coordinates and return its 3D spheric coordinates.
    '''
    x2_y2 = x ** 2 + y ** 2
    r = m.sqrt( x2_y2 + z ** 2 )  # r
    elev = m.atan2( z, m.sqrt( x2_y2 ) )  # Elevation
    az = m.atan2( y, x )  # Azimuth
    return r, elev, az

def image_generation(feature_matrix, electrodes_loc, n_gridpoints):
    n_electrodes = electrodes_loc.shape[0]  # number of electrodes
    n_bands = feature_matrix.shape[1] // n_electrodes  # number of frequency bands considered in the feature matrix
    n_samples = feature_matrix.shape[0]  # number of samples to consider in the feature matrix.

    # Checking the dimension of the feature matrix
    if feature_matrix.shape[1] % n_electrodes != 0:
        print( 'The combination feature matrix - electrodes locations is not working.' )
    assert feature_matrix.shape[1] % n_electrodes == 0
    new_feat = []

    # Reshape a novel feature matrix with a list of array with shape [n_samples x n_electrodes] for each frequency band
    for bands in range( n_bands ):
        new_feat.append( feature_matrix[:, bands * n_electrodes: (bands + 1) * n_electrodes] )

    # Creation of a meshgrid data interpolation
    #   Creation of an empty grid
    grid_x, grid_y = np.mgrid[
                     np.min( electrodes_loc[:, 0] ): np.max( electrodes_loc[:, 0] ): n_gridpoints * 1j,  # along x_axis
                     np.min( electrodes_loc[:, 1] ): np.max( electrodes_loc[:, 1] ): n_gridpoints * 1j  # along y_axis
                     ]

    interpolation_img = []
    #   Interpolation
    #       Creation of the empty interpolated feature matrix
    for bands in range( n_bands ):
        interpolation_img.append( np.zeros( [n_samples, n_gridpoints, n_gridpoints] ) )
    #   Interpolation between the points
    # print('Signals interpolations.')
    for sample in tqdm( range( n_samples ) ):
        for bands in range( n_bands ):
            interpolation_img[bands][sample, :, :] = griddata( electrodes_loc, new_feat[bands][sample, :],
                                                               (grid_x, grid_y), method='cubic', fill_value=np.nan )
    #   Normalization - replacing the nan values by interpolation
    for bands in range( n_bands ):
        interpolation_img[bands][~np.isnan( interpolation_img[bands] )] = scale(
            interpolation_img[bands][~np.isnan( interpolation_img[bands] )] )
        interpolation_img[bands] = np.nan_to_num( interpolation_img[bands] )
    return np.swapaxes( np.asarray( interpolation_img ), 0, 1 )  # swap axes to have [samples, colors, W, H]
