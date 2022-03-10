# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 12:22:14 2020

@author: pkiuru

Script file for extracting a pore network and an OpenPNM network object from
a binary image

Required input:
    3D binary image (bin_img_nofloat) with void space = 1 & solid space = 0

Output:
    OpenPNM network object for use in the following step from memory
    

"""

import openpnm as op
import porespy as ps
import numpy as np
import matplotlib.pyplot as plt
import more_models as mm


# True: Load a binary image from file; False: use the binary image in memory 
load_switch = False
# Path for the image array .npy file
binary_file = '../IO/binimgs/40_45_1.npy'


# Shape of the network domain ('cyl' or 'cube'):
# cyl: 800 x 1000 cylinder (height x diameter)
# cube: centered 600 x 600 x600 cube
domain_switch = 'cyl'

# Switch for network extraction with PoreSpy
# Generates also images of horizontal cross-sections of the binary image
#  (Figure 10) and the resulted pore segmentation (Figure 11)
extract_network = True

# Switch for OpenPNM object generation
openpnm_switch = True

# -----------------------------------------------------------------------------

if load_switch:
    bin_img_nofloat = np.load(binary_file)
    
    #Image voxel edge length (m)
    voxel_size = 50e-6

if domain_switch == 'cyl':
    #Fraction of the area of a sphere with diameter of 1000 voxels to the area
    #of a square with edge length of 1000 woxels in a binary image
    caf = 0.785321
else:
    caf = 1.0


#LISÄÄ TALLENNUSKYTKIN - KATSO MYÖS TALLENNUSKOMENTO!!!!!
    #ei kun ei tallennusta tähän, koska rev_sim_comb(2) lisää vasta sen oikean
    #throatin pituuden! rev_sim:n ja rev_sim_comb pitää siis ajaa yhdessä!

# Volume conversion factor m^3 -> mm^3
volfact = 1e9

if extract_network:
    
    # Select the applied network domain
    
    if domain_switch == 'cyl':
    
        # Sample 1: z_top= 150, z_bot = 750 (z = 600)
        # Sample 2: 200...900 (z = 700)
        # Sample 7: 100...800 (z = 700)
        # Sample 9: 150...950 (z = 800)
        # Otherwise: 100...900 (z = 800)
        
        z_top = 150
        z_bot = 750
        h_min = 0
        h_med = 500
        h_max = 1000
        
    elif domain_switch == 'cube':
        
        z_top = 200
        z_bot = 800
        h_min = 200
        h_med = 500
        h_max = 800
        
    
    np.random.seed(42)
    psnet_1 = ps.networks.snow(bin_img_nofloat[z_top:z_bot,h_min:h_med,h_min:h_med], voxel_size = voxel_size, boundary_faces=['left','right'])
    np.random.seed(42)
    psnet_2 = ps.networks.snow(bin_img_nofloat[z_top:z_bot,h_min:h_med,h_med:h_max], voxel_size = voxel_size, boundary_faces=['left','right'])
    np.random.seed(42)
    psnet_3 = ps.networks.snow(bin_img_nofloat[z_top:z_bot,h_med:h_max,h_min:h_med], voxel_size = voxel_size, boundary_faces=['left','right'])
    np.random.seed(42)
    psnet_4 = ps.networks.snow(bin_img_nofloat[z_top:z_bot,h_med:h_max,h_med:h_max], voxel_size = voxel_size, boundary_faces=['left','right'])
    
    
    if domain_switch == 'cube':
        np.random.seed(42)
        #psnet_5 = ps.networks.snow(bin_img_nofloat[z_top:z_bot,h_min:h_max,h_min:h_max], voxel_size = voxel_size, 
        #                           boundary_faces=['left','right', 'top','bottom', 'front', 'back'])
        psnet_5 = ps.networks.snow(bin_img_nofloat[z_top:z_bot,h_min:h_max,h_min:h_max], voxel_size = voxel_size, boundary_faces=['left','right'])
        
        duu = np.copy(psnet_5.regions)
        duu[duu>0] +=1
        
        mm.osakuva(bin_img_nofloat[z_top+100,h_min:h_max,h_min:h_max],10)
        
        mm.osakuva(duu[103,:,:],11,cmap=plt.cm.Reds)
        
    else:
        
        mm.osakuva(bin_img_nofloat[z_top+100,h_min:h_max,h_min:h_max],10)
        
        plt.figure(num=11,figsize=(6,6))
        plt.clf
        plt.subplot(221)
        plt.imshow(psnet_1.regions[103,:,:],cmap=plt.cm.Reds)
        plt.xticks([])
        plt.yticks([])
        plt.subplot(222)
        plt.imshow(psnet_2.regions[103,:,:],cmap=plt.cm.Reds)
        plt.xticks([])
        plt.yticks([])
        plt.subplot(223)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(psnet_3.regions[103,:,:],cmap=plt.cm.Reds)
        plt.subplot(224)
        plt.imshow(psnet_4.regions[103,:,:],cmap=plt.cm.Reds)
        plt.xticks([])
        plt.yticks([])
        plt.tight_layout()


if openpnm_switch:
    
    ws = op.Workspace()
    ws.clear()
    
    proj1 = ws.new_project(name = 'subimage_1')
    proj2 = ws.new_project(name = 'subimage_2')
    proj3 = ws.new_project(name = 'subimage_3')
    proj4 = ws.new_project(name = 'subimage_4')
    if domain_switch == 'cube':
        proj5 = ws.new_project(name = 'image')
    
    pn_1 = op.network.GenericNetwork(project=proj1)
    pn_1.update(psnet_1)
    geom_1 = op.geometry.GenericGeometry(network=pn_1, pores=pn_1.Ps, throats=pn_1.Ts)
        
    pn_2 = op.network.GenericNetwork(project=proj2)
    pn_2.update(psnet_2)
    geom_2 = op.geometry.GenericGeometry(network=pn_2, pores=pn_2.Ps, throats=pn_2.Ts)
        
    pn_3 = op.network.GenericNetwork(project=proj3)
    pn_3.update(psnet_3)
    geom_3 = op.geometry.GenericGeometry(network=pn_3, pores=pn_3.Ps, throats=pn_3.Ts)
    
    pn_4 = op.network.GenericNetwork(project=proj4)
    pn_4.update(psnet_4)
    geom_4 = op.geometry.GenericGeometry(network=pn_4, pores=pn_4.Ps, throats=pn_4.Ts)
    
            
    subvol = (z_bot - z_top) * (h_med - h_min) * (h_med - h_min) * voxel_size**3
    totvol = (z_bot - z_top) * (h_max - h_min) * (h_max - h_min) * voxel_size**3
    
    print('Number of top boundary pores, untrimmed subnetworks')
    print( np.sum(pn_1['pore.left']), np.sum(pn_2['pore.left']),
           np.sum(pn_3['pore.left']), np.sum(pn_4['pore.left']))
    
    print('Number of bottom boundary pores, untrimmed subnetworks')
    print( np.sum(pn_1['pore.right']), np.sum(pn_2['pore.right']),
           np.sum(pn_3['pore.right']), np.sum(pn_4['pore.right']))
    
    print('Pore count, untrimmed subnetworks')
    print( np.sum(~pn_1['pore.boundary']),np.sum(~pn_2['pore.boundary']),
           np.sum(~pn_3['pore.boundary']),np.sum(~pn_4['pore.boundary']))
    
    print('Total pore volumes, untrimmed subnetworks')
    print( np.sum(pn_1['pore.volume'][~pn_1['pore.boundary']])/voxel_size**3,
           np.sum(pn_2['pore.volume'][~pn_2['pore.boundary']])/voxel_size**3,
           np.sum(pn_3['pore.volume'][~pn_3['pore.boundary']])/voxel_size**3,
           np.sum(pn_4['pore.volume'][~pn_4['pore.boundary']])/voxel_size**3)

    print('Porosities, untrimmed subnetworks')
    print( np.sum(pn_1['pore.volume'][~pn_1['pore.boundary']]) / (subvol*caf),
           np.sum(pn_2['pore.volume'][~pn_2['pore.boundary']]) / (subvol*caf),
           np.sum(pn_3['pore.volume'][~pn_3['pore.boundary']]) / (subvol*caf),
           np.sum(pn_4['pore.volume'][~pn_4['pore.boundary']]) / (subvol*caf))
    
    proj1w = proj1.copy(name = 'whole_subnetwork_1')
    proj2w = proj2.copy(name = 'whole_subnetwork_2')
    proj3w = proj3.copy(name = 'whole_subnetwork_3')
    proj4w = proj4.copy(name = 'whole_subnetwork_4')
    
    hech_1 = pn_1.check_network_health()
    op.topotools.trim(network=pn_1, pores=hech_1['trim_pores']) 
    
    volhist_1 = np.histogram(np.log10(volfact*pn_1['pore.volume'][~pn_1['pore.boundary']]), bins=20)

    hech_2 = pn_2.check_network_health()
    op.topotools.trim(network=pn_2, pores=hech_2['trim_pores']) 
    
    volhist_2 = np.histogram(np.log10(volfact*pn_2['pore.volume'][~pn_2['pore.boundary']]), bins=20)

    hech_3 = pn_3.check_network_health()
    op.topotools.trim(network=pn_3, pores=hech_3['trim_pores']) 
    
    volhist_3 = np.histogram(np.log10(volfact*pn_3['pore.volume'][~pn_3['pore.boundary']]), bins=20)
    
    hech_4 = pn_4.check_network_health()
    op.topotools.trim(network=pn_4, pores=hech_4['trim_pores']) 
    
    volhist_4 = np.histogram(np.log10(volfact*pn_4['pore.volume'][~pn_4['pore.boundary']]), bins=20)
    
    print('Porosity of the cylindrical image')
    print(ps.metrics.porosity(bin_img_nofloat[100:900,:,:])/caf)
    
    print('Top boundary pores, trimmed subnetworks')
    print( np.sum(pn_1['pore.left']), np.sum(pn_2['pore.left']),
           np.sum(pn_3['pore.left']), np.sum(pn_4['pore.left']))
    
    print('Bottom boundary pores, trimmed subnetworks')
    print( np.sum(pn_1['pore.right']), np.sum(pn_2['pore.right']),
           np.sum(pn_3['pore.right']), np.sum(pn_4['pore.right']))
    
    print('Pore counts, trimmed subnetworks')
    print( np.sum(~pn_1['pore.boundary']),np.sum(~pn_2['pore.boundary']),
           np.sum(~pn_3['pore.boundary']),np.sum(~pn_4['pore.boundary']))
    
    print('Total pore volumes, trimmed subnetworks')
    print( np.sum(pn_1['pore.volume'][~pn_1['pore.boundary']])/voxel_size**3,
           np.sum(pn_2['pore.volume'][~pn_2['pore.boundary']])/voxel_size**3,
           np.sum(pn_3['pore.volume'][~pn_3['pore.boundary']])/voxel_size**3,
           np.sum(pn_4['pore.volume'][~pn_4['pore.boundary']])/voxel_size**3)
    
    print('Porosities, trimmed subnetworks')
    print( np.sum(pn_1['pore.volume'][~pn_1['pore.boundary']]) / (subvol*caf),
           np.sum(pn_2['pore.volume'][~pn_2['pore.boundary']]) / (subvol*caf),
           np.sum(pn_3['pore.volume'][~pn_3['pore.boundary']]) / (subvol*caf),
           np.sum(pn_4['pore.volume'][~pn_4['pore.boundary']]) / (subvol*caf))
    
    print('Image region porosities')
    print( np.sum(bin_img_nofloat[z_top:z_bot,h_min:h_med,h_min:h_med]) / (subvol/voxel_size**3),
           np.sum(bin_img_nofloat[z_top:z_bot,h_min:h_med,h_med:h_max]) / (subvol/voxel_size**3),
           np.sum(bin_img_nofloat[z_top:z_bot,h_med:h_max,h_min:h_med]) / (subvol/voxel_size**3),
           np.sum(bin_img_nofloat[z_top:z_bot,h_med:h_max,h_med:h_max]) / (subvol/voxel_size**3))

    
    
    if domain_switch == 'cube':
        
        pn_5 = op.network.GenericNetwork(project=proj5)
        pn_5.update(psnet_5)
        geom_5 = op.geometry.GenericGeometry(network=pn_5, pores=pn_5.Ps, throats=pn_5.Ts)
        
        print('Number of top boundary pores, single untrimmed network')
        try:
            print( np.sum(pn_5['pore.left']))
        except Exception:
            pass
        
        print('Number of bottom boundary pores, single untrimmed network')
        try:
            print( np.sum(pn_5['pore.right']))
        except Exception:
            pass
        
        print('Pore count, single untrimmed network')
        print( np.sum(~pn_5['pore.boundary']))
        
        print('Throat count, single untrimmed network')
        print(np.sum(pn_5['throat.internal']))
        
        print('Total pore volume, single untrimmed network')
        print( np.sum(pn_5['pore.volume'][~pn_5['pore.boundary']]))
        
        print('Porosity, single untrimmed network (cubical)')
        print( np.sum(pn_5['pore.volume'][~pn_5['pore.boundary']]) / totvol)
        
        proj5w = proj5.copy(name = 'whole_network')
        
        pn_5w = proj5w['net_01']
        
        hech_5 = pn_5.check_network_health()
        op.topotools.trim(network=pn_5, pores=hech_5['trim_pores']) 
        
        volhist_5 = np.histogram(np.log10(volfact*pn_5['pore.volume'][~pn_5['pore.boundary']]), bins=20)
        
        print('Number of top boundary pores, single trimmed network')
        try:
            print( np.sum(pn_5['pore.left']))
        except Exception:
            pass
        
        print('Number of bottom boundary pores, single trimmed network')
        try:
            print( np.sum(pn_5['pore.right']))
        except Exception:
            pass
        
        print('Pore count, single trimmed network')
        print( np.sum(~pn_5['pore.boundary']))
        
        print('Throat count, single trimmed network')
        print(np.sum(pn_5['throat.internal']))
        
        print('Total pore volume, single trimmed network')
        print( np.sum(pn_5['pore.volume'][~pn_5['pore.boundary']]))
        
        print('Porosity, single trimmed network (cubical)')
        print( np.sum(pn_5['pore.volume'][~pn_5['pore.boundary']]) / totvol)
