# -*- coding: utf-8 -*-
"""
Created on Thu Dec  9 16:25:39 2021

@author: pkiuru
"""

import openpnm as op
import porespy as ps
import numpy as np
import matplotlib.pyplot as plt


#----------------------------------------------------
from matplotlib import rcParams

fontname = 'Arial'
fontsize = 8

rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial']
rcParams['xtick.direction'] = 'in'
rcParams['ytick.direction'] = 'in'

#rcParams['mathtext.fontset'] = 'cm'

rcParams['font.size'] = fontsize

rcParams['axes.titlesize'] = fontsize
rcParams['axes.labelsize'] = fontsize
rcParams['axes.titleweight'] = 'normal'

rcParams['xtick.labelsize'] = fontsize
rcParams['ytick.labelsize'] = fontsize
 
rcParams['legend.handlelength'] = 1.0

#----------------------------------------------------------

def load_diffusion_curves(filename):
    
    topfolder = '../IO/results_diffusion/'
    suffix = '.npz'
    
    dummy = np.load(topfolder+filename+suffix) 
    
    return dummy['press'], dummy['afp'], dummy['diffcoeff'] 

def limfunc(x, pad=0.15):
    
    lower = x[0] - 0.5 * pad * (x[-1]-x[0])
    upper = x[-1] + pad * (x[-1]-x[0])

    return [lower, upper]

#----------------------------------------------------------

load_diff = True

# Drainage & imbibition

if load_diff:
        
        caf = 0.785321
        cacorr = np.cos(np.radians(180)) / np.cos(np.radians(120))
        
        a = '0_5'
        press_1_2, afp_1_2, dc_1_2 = load_diffusion_curves('diff_'+a+'_2')
        press_imb_1_2, afp_imb_1_2, dc_imb_1_2 = load_diffusion_curves('diffimb_'+a+'_2')
        
        a = '20_25'
        press_2_1, afp_2_1, dc_2_1 = load_diffusion_curves('diff_'+a+'_1')
        press_imb_2_1, afp_imb_2_1, dc_imb_2_1 = load_diffusion_curves('diffimb_'+a+'_1')
        
        a = '40_45'
        press_3_1, afp_3_1, dc_3_1 = load_diffusion_curves('diff_'+a+'_1')
        press_imb_3_1, afp_imb_3_1, dc_imb_3_1 = load_diffusion_curves('diffimb_'+a+'_1')
        
        
        #Contact angle correction: 120 degrees -> 180 degrees
        
        press_1_2 = cacorr * press_1_2 / 1000
        press_imb_1_2 = cacorr * press_imb_1_2 / 1000
        
        press_2_1 = cacorr * press_2_1 / 1000
        press_imb_2_1 = cacorr * press_imb_2_1 / 1000

        press_3_1 = cacorr * press_3_1 / 1000
        press_imb_3_1 = cacorr * press_imb_3_1 / 1000
        
        
        # Area correction: square -> cylinder

        afp_1_2 /= caf
       
        afp_2_1 /= caf

        afp_3_1 /= caf

        afp_imb_1_2 /= caf
        
        afp_imb_2_1 /= caf
        
        afp_imb_3_1 /= caf

        # m2/s -> cm2/s

        dc_1_2 = dc_1_2 * 10000
        dc_imb_1_2 = dc_imb_1_2 * 10000
        
        dc_2_1 = dc_2_1 * 10000
        dc_imb_2_1 = dc_imb_2_1 * 10000

        dc_3_1 = dc_3_1 * 10000
        dc_imb_3_1 = dc_imb_3_1 * 10000




mplims = [0.03, 4.5]

bbdim = [0.245, 0.245]

bppos = [
        [0.090, 0.705, bbdim[0],bbdim[1]],
        [0.405, 0.705, bbdim[0],bbdim[1]],
        [0.725, 0.705, bbdim[0],bbdim[1]],
        [0.090, 0.385, bbdim[0],bbdim[1]],
        [0.405, 0.385, bbdim[0],bbdim[1]],
        [0.725, 0.385, bbdim[0],bbdim[1]],
        [0.090, 0.065, bbdim[0],bbdim[1]],
        [0.405, 0.065, bbdim[0],bbdim[1]],
        [0.725, 0.065, bbdim[0],bbdim[1]]
        ]


subtpos = [0.05,0.88]

subt = [('(a)'), ('(b)'), ('(c)'), ('(d)'),
        ('(e)'), ('(f)'), ('(g)'), ('(h)'), ('(i)')]

titletext = ['0-5 cm','20-25 cm','40-45 cm']
titlepos = [0.50, 1.08]

ylc = [-0.19, -0.19, -0.23,
       -0.18 ,-0.18, -0.23,
       -0.18, -0.18, -0.23
       ]
        
lw = 1.0 # line width
msize = 2 # marker size
linestyle = ['o-c', 'o--g']

fig = plt.figure(num=1)        
fig.set_size_inches(6,4.98)
fig.clf()

ax1 = fig.add_subplot(3,3,1)
ax1.plot(dc_1_2, press_1_2,linestyle[0], markersize=msize, lw=lw, label='Drainage')
ax1.plot(dc_imb_1_2, press_imb_1_2, linestyle[1], markersize=msize, lw=lw, label='Imbibition')
ax1.legend(loc='lower right')
ax1.set_xlabel('$D_\mathrm{pnm}$ [cm$^2$ s$^{-1}$]')
ax1.set_ylabel('Matric potential [kPa]')
ax1.set_xlim([-0.0005,1.1*np.max(dc_1_2)])
ax1.set_yscale('log')
ax1.set_yticks([0.1, 1])
ax1.set_yticklabels(['-0.1','-1'])
ax1.set_ylim(mplims)
ax1.yaxis.set_label_coords(ylc[0], 0.5)
ax1.text(titlepos[0], titlepos[1], titletext[0], transform=ax1.transAxes,
         horizontalalignment='center')
ax1.text(subtpos[0], subtpos[1], subt[0], transform=ax1.transAxes)
ax1.set_position(bppos[0])

ax2 = fig.add_subplot(3,3,2)
ax2.plot(afp_1_2,press_1_2,  linestyle[0], markersize=msize, lw=lw, label='Drainage')
ax2.plot(afp_imb_1_2,press_imb_1_2,  linestyle[1], markersize=msize, lw=lw, label='Imbibition')
#ax2.legend()
ax2.set_xlabel('Air-filled porosity')
ax2.set_ylabel('Matric potential [kPa]')
ax2.set_xlim([-0.01,1.1*np.max(afp_1_2)])
ax2.set_yscale('log')
ax2.set_yticks([0.1, 1])
ax2.set_yticklabels(['-0.1','-1'])
ax2.set_ylim(mplims)
ax2.yaxis.set_label_coords(ylc[1], 0.5)
ax2.text(subtpos[0], subtpos[1], subt[3], transform=ax2.transAxes)
ax2.set_position(bppos[3])

ax3 = fig.add_subplot(3,3,3)
ax3.plot(afp_1_2, dc_1_2, linestyle[0], markersize=msize, lw=lw, label='Drainage')
ax3.plot(afp_imb_1_2, dc_imb_1_2, linestyle[1], markersize=msize, lw=lw, label='Imbibition')
#ax3.legend()
ax3.set_xlabel('Air-filled porosity')
ax3.set_ylabel('$D_\mathrm{pnm}$ [cm$^2$ s$^{-1}$]')

#ax3.set_ylim([-0.0005,1.1*np.max(dc_1_2)])
ax3.set_xlim(limfunc(afp_1_2,pad=0.1))
ax3.set_ylim(limfunc(dc_1_2))
ax3.yaxis.set_label_coords(ylc[2], 0.5)
#ax3.yaxis.tick_right()
ax3.text(subtpos[0], subtpos[1], subt[6], transform=ax3.transAxes)
ax3.set_position(bppos[6])

ax4 = fig.add_subplot(3,3,4)
ax4.plot(dc_2_1, press_2_1,linestyle[0], markersize=msize, lw=lw, label='Drainage')
ax4.plot(dc_imb_2_1, press_imb_2_1, linestyle[1], markersize=msize, lw=lw, label='Imbibition')
ax4.set_xlabel('$D_\mathrm{pnm}$ [cm$^2$ s$^{-1}$]')
#ax4.set_ylabel('Matric potential [kPa]')
ax4.set_xlim([-0.0005,1.1*np.max(dc_2_1)])
ax4.set_yscale('log')
ax4.set_yticks([0.1, 1])
ax4.set_yticklabels(['-0.1','-1'])
ax4.set_ylim(mplims)
ax4.yaxis.set_label_coords(ylc[3], 0.5)
ax4.text(titlepos[0], titlepos[1], titletext[1], transform=ax4.transAxes,
         horizontalalignment='center')
ax4.text(subtpos[0], subtpos[1], subt[1], transform=ax4.transAxes)
ax4.set_position(bppos[1])


ax5 = fig.add_subplot(3,3,5)
ax5.plot(afp_2_1,press_2_1,  linestyle[0], markersize=msize, lw=lw, label='Drainage')
ax5.plot(afp_imb_2_1,press_imb_2_1,  linestyle[1], markersize=msize, lw=lw, label='Imbibition')
ax5.set_xlabel('Air-filled porosity')
#ax5.set_ylabel('Matric potential [kPa]')
ax5.set_xlim([-0.01,1.1*np.max(afp_2_1)])
ax5.set_yscale('log')
ax5.set_yticks([0.1, 1])
ax5.set_yticklabels(['-0.1','-1'])
ax5.set_ylim(mplims)
ax5.yaxis.set_label_coords(ylc[4], 0.5)
ax5.text(subtpos[0], subtpos[1], subt[4], transform=ax5.transAxes)
ax5.set_position(bppos[4])

ax6 = fig.add_subplot(3,3,6)
ax6.plot(afp_2_1, dc_2_1, linestyle[0], markersize=msize, lw=lw, label='Drainage')
ax6.plot(afp_imb_2_1, dc_imb_2_1, linestyle[1], markersize=msize, lw=lw, label='Imbibition')
ax6.set_xlabel('Air-filled porosity')
#ax6.set_ylabel('D [cm$^2$ s$^{-1}$]')
#ax6.set_ylim([-0.0005,1.1*np.max(dc_2_1)])
ax6.set_xlim(limfunc(afp_2_1,pad=0.1))
ax6.set_ylim(limfunc(dc_2_1))
ax6.yaxis.set_label_coords(ylc[5], 0.5)
#ax6.yaxis.tick_right()
ax6.text(subtpos[0], subtpos[1], subt[7], transform=ax6.transAxes)
ax6.set_position(bppos[7])

ax7 = fig.add_subplot(3,3,7)
ax7.plot(dc_3_1, press_3_1,linestyle[0], markersize=msize, lw=lw, label='Drainage')
ax7.plot(dc_imb_3_1, press_imb_3_1, linestyle[1], markersize=msize, lw=lw, label='Imbibition')
ax7.set_xlabel('$D_\mathrm{pnm}$ [cm$^2$ s$^{-1}$]')
#ax7.set_ylabel('Matric potential [kPa]')
ax7.set_xlim([-0.0005,1.1*np.max(dc_3_1)])
ax7.set_yscale('log')
ax7.set_yticks([0.1, 1])
ax7.set_yticklabels(['-0.1','-1'])
ax7.set_ylim(mplims)
ax7.yaxis.set_label_coords(ylc[6], 0.5)
ax7.text(titlepos[0], titlepos[1], titletext[2], transform=ax7.transAxes,
         horizontalalignment='center')
ax7.text(subtpos[0], subtpos[1], subt[2], transform=ax7.transAxes)
ax7.set_position(bppos[2])

ax8 = fig.add_subplot(3,3,8)
ax8.plot(afp_3_1,press_3_1,  linestyle[0], markersize=msize, lw=lw, label='Drainage')
ax8.plot(afp_imb_3_1,press_imb_3_1,  linestyle[1], markersize=msize, lw=lw, label='Imbibition')
ax8.set_xlabel('Air-filled porosity')
#ax8.set_ylabel('Matric potential [kPa]')
ax8.set_xlim([-0.01,1.1*np.max(afp_3_1)])
ax8.set_yscale('log')
ax8.set_yticks([0.1, 1])
ax8.set_yticklabels(['-0.1','-1'])
ax8.set_ylim(mplims)
ax8.yaxis.set_label_coords(ylc[7], 0.5)
ax8.text(subtpos[0], subtpos[1], subt[5], transform=ax8.transAxes)
ax8.set_position(bppos[5])

ax9 = fig.add_subplot(3,3,9)
ax9.plot(afp_3_1, dc_3_1, linestyle[0], markersize=msize, lw=lw, label='Drainage')
ax9.plot(afp_imb_3_1, dc_imb_3_1, linestyle[1], markersize=msize, lw=lw, label='Imbibition')
ax9.set_xlabel('Air-filled porosity')
#ax9.set_ylabel('D [cm$^2$ s$^{-1}$]')
#ax9.set_ylim([-0.0005,1.1*np.max(dc_3_1)])
ax9.set_xlim(limfunc(afp_3_1,pad=0.1))
ax9.set_ylim(limfunc(dc_3_1))
ax9.yaxis.set_label_coords(ylc[8], 0.5)
#ax9.yaxis.tick_right()
ax9.text(subtpos[0], subtpos[1], subt[8], transform=ax9.transAxes)
ax9.set_position(bppos[8])


ax1.xaxis.set_label_coords(0.5, -0.135)
ax2.xaxis.set_label_coords(0.5, -0.15)
ax3.xaxis.set_label_coords(0.5, -0.15)
ax4.xaxis.set_label_coords(0.5, -0.135)
ax5.xaxis.set_label_coords(0.5, -0.15)
ax6.xaxis.set_label_coords(0.5, -0.15)
ax7.xaxis.set_label_coords(0.5, -0.135)
ax8.xaxis.set_label_coords(0.5, -0.15)
ax9.xaxis.set_label_coords(0.5, -0.15)

'''
plt.figure(num=2)
plt.clf()
plt.plot(afp_imb_1_2/afp_imb_1_2[-1],label='0-5')
plt.plot(afp_imb_2_1/afp_imb_2_1[-1],label='20-25')
plt.plot(afp_imb_3_1/afp_imb_3_1[-1],label='40-45')
plt.legend()

plt.figure(num=3)
plt.clf()
plt.plot(dc_imb_1_2/dc_imb_1_2[-1],label='0-5')
plt.plot(dc_imb_2_1/dc_imb_2_1[-1],label='20-25')
plt.plot(dc_imb_3_1/dc_imb_3_1[-1],label='40-45')
plt.legend()
'''

#plt.savefig('fig02.pdf')

