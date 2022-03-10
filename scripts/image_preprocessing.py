# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 10:17:27 2020

@author: pkiuru

Script file for preprocessing 3D microtomography images and extracting pore
network

Required input:
    Folder containing the microtomography image files (topfolder)
    File containing the image rotation parameters and image grayscale
    intensity ranges (paramfile)

Output:
    3D binary image (bin_img_nofloat) with void space = 1 & solid space = 0

"""

import os


import matplotlib.pyplot as plt

from skimage import filters
from skimage import exposure
from skimage import io
from skimage.morphology import ball

import numpy as np
import more_models as mm
from scipy import ndimage
import porespy as ps

#Folder of the image stack folders
topfolder = '../IO/uCT'

#File that contains the image rotation parameters and grayscale intensity ranges
paramfile = '../IO/Scan_parameters.csv'

#Path and filename for saving the binary image
save_switch = False
savefile = '../IO/20_25_4.npy'

# Number of the analyzed tomography image in the alphabetized folder list
scan = 1

# Switch for reading the image files and stacking a 3D image.
# The first array axis gives the axial coordinate of the sample cylinder.
# Creates also images of central cross sections in all directions (Figures 1-3)
# and horizontal cross sections approximately at the top and bottom of
# the sample (Figure 4).
imageread = True

# Switch for performing straightening and cylinder extraction of the 3D image.
# Creates also voxel grayscale intensity histogram (Figure 5) and horizontal
# cross sections approximately at the top and bottom of the original and
# straightened images (Figure 6).
cylinder_analysis = True
    
# Performs conversion from 16-bit to 8-bit grayscale image
bitscale = 1#True

# Performs noise filtering and thresholding and creates binary image (with void
# space having value 1)
# Creates also voxel grayscale intensity histograms for the original and filtered
# 8-bit images (Figure 7) and horizontal cross-section images of the original,
#  filtered, and binary images (Figure 8)
binary = 1#True

# Calculates image porosity and vertical porosity profile
# Creates also a plot of vertical porosity profile (Figure 9)
poros = 1#True

# -----------------------------------------------------------------------------

#Image voxel edge length (m)
voxel_size = 50e-6

#Fraction of the area of a sphere with diameter of 1000 voxels to the area
#of a square with edge length of 1000 woxels in a binary image
caf = 0.785321


if imageread:
    # Generate  list of image stack folders
    image_folders = []
    filenames = []
    
    for root, dirs, files in os.walk(topfolder, topdown=False):
        for name in dirs:
            image_folders.append(os.path.join(root, name))
    
    #Read the image in 16 bit grayscale
    stack_list = []
    
    print('Loading ' + os.path.basename(image_folders[scan-1]))
    for root, dirs, files in os.walk(image_folders[scan-1], topdown=False):
        for filename in files:
            filenames.append(os.path.join(root, filename))
            stack_list.append(io.imread(os.path.join(root, filename)))
    
    whole_image = np.asarray(stack_list)
    del stack_list   
    
    #Middle z
    mm.osakuva(whole_image[576,:,:], 1, cmap='gray')
    #Middle x
    mm.osakuva(whole_image[:,576,:], 2, cmap='gray') 
    #Middle y
    mm.osakuva(whole_image[:,:,576], 3, cmap='gray')
    
    fig, ax = plt.subplots(1, 2, num=4, figsize=(12, 6), sharex=True, sharey=True)
    ax[0].imshow(whole_image[150, :, :], cmap='gray', interpolation='nearest')
    ax[1].imshow(whole_image[1050, :, :], cmap='gray', interpolation='nearest')
    plt.tight_layout()
    
    
if cylinder_analysis:
    
    crop_coords, rot_params, scale_params = mm.read_params(paramfile)
    
    cropped = whole_image[crop_coords[scan-1,0]:crop_coords[scan-1,1],
                          crop_coords[scan-1,2]:crop_coords[scan-1,3], 
                          crop_coords[scan-1,4]:crop_coords[scan-1,5]]
    
    rotated = np.copy(whole_image)
    
    if rot_params[scan-1,0] != 0:
        print('Rotating around x axis')
        rotated = ndimage.rotate(rotated, rot_params[scan-1,0],
                             axes=(0, 1),
                             mode='reflect', reshape=False)
    
    if rot_params[scan-1,1] != 0:
        print('Rotating around y axis')
        rotated = ndimage.rotate(rotated, rot_params[scan-1,1],
                             axes=(0, 2),
                             mode='reflect', reshape=False)
    
    cropped_rot = rotated[crop_coords[scan-1,0]:crop_coords[scan-1,1],
                          crop_coords[scan-1,2]:crop_coords[scan-1,3], 
                          crop_coords[scan-1,4]:crop_coords[scan-1,5]]
    
    print('Extracting cylinder')
    syl_rot = ps.tools.extract_cylinder(cropped_rot, axis=0)
    
    borders = syl_rot == 0
    
    hist_s, bins_center_s = exposure.histogram(syl_rot[~borders])
    
    plt.figure(num=5, figsize=(6,6))
    plt.clf()
    plt.plot(bins_center_s, hist_s, linewidth=2,label='Original image')
    plt.legend()
    
    plt.figure(num=6, figsize=(11, 9))
    plt.clf()
    ax1 = plt.subplot(221)    
    plt.imshow(whole_image[150,:,:], cmap='gray', interpolation='nearest')#, vmin=0, vmax=25000)
    plt.title('Orig 100')
    ax2 = plt.subplot(222, sharex=ax1, sharey=ax1)    
    plt.imshow(whole_image[-250,:,:], cmap='gray', interpolation='nearest')#, vmin=0, vmax=25000)
    plt.title('Orig 1000')
    ax3 = plt.subplot(223, sharex=ax1, sharey=ax1)    
    plt.imshow(rotated[150,:,:], cmap='gray', interpolation='nearest')#, vmin=0, vmax=25000)
    plt.title('Rot 100')
    ax4 = plt.subplot(224, sharex=ax1, sharey=ax1)    
    plt.imshow(rotated[-250,:,:], cmap='gray', interpolation='nearest')#, vmin=0, vmax=25000)
    plt.title('Rot 1000') 

    
if bitscale:

    G8min = 0
    G8max = 255
    
    h = np.copy(syl_rot)
    crop_coords, rot_params, scale_params = mm.read_params(paramfile)
    G16min = scale_params[scan-1,0]
    G16max = scale_params[scan-1,1]
        
    h[h>G16max] = G16max
    h[h<G16min] = G16min           
    image_8bit = (G8min + (G8max-G8min)/(G16max-G16min) * (h-G16min)).astype(np.uint8)
    del h

    hists, bins_centers = exposure.histogram(image_8bit[~borders])

if binary:    
    
    print('Filtering...')
    
    #Median filtering with radius 2
    
    filtered = ndimage.median_filter(image_8bit, footprint=ball(2))  
    print('Filtered')
    
    histf, bins_centerf = exposure.histogram(filtered[~borders])

    plt.figure(num=7, figsize=(6,6))
    plt.clf()
    plt.plot(bins_centerf, histf, lw=2,label='Filtered')
    try:
        plt.plot(bins_centers, hists, lw=2,label='Rescaled')
    except Exception:
        pass
    plt.legend()
    
    print('Thresholding...')
    
    #Otsu thresholding
    
    val  = filters.threshold_otsu(filtered[~borders])

    print('Thresholded')
    
    #Binary image (void voxels = 1)
    
    bin_img = filtered <= val
    
    #Region outside the cylinder is set solid
    
    bin_img[borders] = False

    #Identification of floating solids and isolated void space
    
    bin_img2 = np.copy(bin_img)
    disconn = ps.filters.find_disconnected_voxels(~bin_img2)
    disconns = ps.filters.find_disconnected_voxels(bin_img2)
    
    floating_solid = np.sum(disconn)
    isolated_void = np.sum(disconns)
    
    del bin_img2
    
    #REmoval of floating solids from the final binary image
    
    bin_img_nofloat = np.copy(bin_img)
    bin_img_nofloat[disconn] = True
    
    slic = 150

    plt.figure(num=8, figsize=(11, 4))
    plt.clf()
    ax1 = plt.subplot(131)
    plt.imshow(syl_rot[slic,:,:], cmap='gray', interpolation='nearest')
    plt.title('Original 8 bit')
    ax2 = plt.subplot(132, sharex=ax1, sharey=ax1)    
    plt.imshow(filtered[slic,:,:], cmap='gray', interpolation='nearest', vmin=0, vmax=255)
    plt.title('Filtered')
    ax3 = plt.subplot(133, sharex=ax1, sharey=ax1)    
    plt.imshow(bin_img[slic,:,:], cmap='gray', interpolation='nearest')
    plt.title('Otsu thresholding')
    plt.tight_layout()
    
    
if poros:

    print('Starting porosity calculations...')
    
    #Total porosity (= void fraction) of the air-filled sample
    p = ps.metrics.porosity(bin_img_nofloat[~borders])
    
    #Vertical porosity profile of the cylindrical sample
    p_prof = ps.metrics.porosity_profile(bin_img_nofloat, 0) / (np.pi / 4)
    
    print('Image porosity ' + str(p))
    
    plt.figure(num=9)
    plt.clf()
    plt.plot(p_prof,label='Otsu w floating removal')
    plt.legend()
    plt.title('Vertical porosity profile')

if save_switch:
   
    np.save(savefile, bin_img_nofloat)
        