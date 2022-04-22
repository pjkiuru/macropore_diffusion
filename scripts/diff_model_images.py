# -*- coding: utf-8 -*-
"""
Created on Mon Oct 11 13:01:45 2021

@author: pkiuru
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
try:
    import statsmodels.api as sm
    from statsmodels.formula.api import ols
    
    import statsmodels.stats.oneway as ao
    import statsmodels.stats.multicomp as mc
    import hypothetical as hp
    import scikit_posthocs as sp
except Exception:
    pass

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

#----------------------------------------------------

dataread = 1

draw_f_9 = 1

ttests = 0

variance_homogen = 0

anova = 0
anova_ols = 0

tukey = 0
games= 0
dunn = 0

boxpl = 0


if dataread:
    
    df = pd.read_excel('../Data_diffusion/Diffusion_data.xlsx', sheet_name='plotdata_final',
                     header=0)
    
    df_a = pd.read_excel('../data_diffusion/Diffusion_data.xlsx', sheet_name='plotdata_a_final',
                     header=0)

    Dc_1_1 = df['Dc_1kPa'][0:7].dropna()
    Dc_1_2 = df['Dc_1kPa'][7:14].dropna()
    Dc_1_3 = df['Dc_1kPa'][14:21].dropna()
    
    Dc_3_1 = df['Dc_3kPa'][0:7].dropna()
    Dc_3_2 = df['Dc_3kPa'][7:14].dropna()
    Dc_3_3 = df['Dc_3kPa'][14:23].dropna()
    
    Dc_6_1 = df['Dc_6kPa'][0:7].dropna()
    Dc_6_2 = df['Dc_6kPa'][7:14].dropna()
    Dc_6_3 = df['Dc_6kPa'][14:21].dropna() #!
    
    Dc_10_1 = df['Dc_10kPa'][0:7].dropna()
    Dc_10_2 = df['Dc_10kPa'][7:14].dropna()
    Dc_10_3 = df['Dc_10kPa'][14:21].dropna() #!

    Df_1_1 = df['Df_1kPa'][0:7].dropna()
    Df_1_2 = df['Df_1kPa'][7:14].dropna()
    Df_1_3 = df['Df_1kPa'][14:21].dropna()
    
    Df_3_1 = df['Df_3kPa'][0:7].dropna()
    Df_3_2 = df['Df_3kPa'][7:14].dropna()
    Df_3_3 = df['Df_3kPa'][14:21].dropna()
    
    Df_6_1 = df['Df_6kPa'][0:7].dropna()
    Df_6_2 = df['Df_6kPa'][7:14].dropna()
    Df_6_3 = df['Df_6kPa'][14:21].dropna()  #!
    
    Df_10_1 = df['Df_10kPa'][0:7].dropna()
    Df_10_2 = df['Df_10kPa'][7:14].dropna()
    Df_10_3 = df['Df_10kPa'][14:21].dropna()  #!
    
    try:
        Dpnm_1 = df['Dpnm'][0:7].dropna()
        Dpnm_2 = df['Dpnm'][7:14].dropna()
        Dpnm_3 = df['Dpnm'][14:21].dropna()  #!
    except Exception:
        pass
    
    Dc_1_mod = np.asarray(df['Dc_1kPa'].dropna())  #!
    Df_1_mod = np.asarray(df['Df_1kPa'].dropna())  #!
    Dc_3_mod = np.asarray(df['Dc_3kPa'].dropna())  #!
    Df_3_mod = np.asarray(df['Df_3kPa'].dropna())  #!
    Dc_6_mod = np.asarray(df['Dc_6kPa'].dropna())  #!
    Df_6_mod = np.asarray(df['Df_6kPa'].dropna())  #!
    Dc_10_mod = np.asarray(df['Dc_10kPa'].dropna())  #!
    Df_10_mod = np.asarray(df['Df_10kPa'].dropna())  #!
    #depths_mod_6 = np.array([1,1,1,1,1,1,1,2,2,2,2,2,2,2,3,3,3,3,3,3])
    #depths_mod_10 = np.array([1,1,1,1,1,1,1,2,2,2,2,2,2,2,3,3,3,3,3,3])
    '''
    depths_mod_1  = np.array([1,1,1,1,1,1,   2,2,2,2,2,2,2, 3,3,3,3,3,3  ])
    depths_mod_3  = np.array([1,1,1,1,1,1,   2,2,2,2,2,2,2, 3,3,3,3,3,3  ])
    depths_mod_6  = np.array([1,1,1,1,1,1,   2,2,2,2,2,2,2, 3,3,3,3,3    ])
    depths_mod_10 = np.array([1,1,1,1,1,1,1 ,2,2,2,2,2,2,2, 3,3,3,3,3    ])
    '''
    depths_mod_1  = np.array([1,1,1,1,1,1,1, 2,2,2,2,2,2,2, 3,3,3,3,3,3  ])
    depths_mod_3  = np.array([1,1,1,1,       2,2,2,2,       3,3,3,3      ])
    depths_mod_6  = np.array([1,1,1,1,1,1,   2,2,2,2,2,2,2, 3,3,3,3,3,3  ])
    depths_mod_10 = np.array([1,1,1,1,1,     2,2,2,2,2,2,   3,3,3,3,3,3  ])
        
    Dc_0_5_cm = np.concatenate([Dc_1_1, Dc_3_1, Dc_6_1, Dc_10_1])
    Df_0_5_cm = np.concatenate([Df_1_1, Df_3_1, Df_6_1, Df_10_1])
    Dc_20_25_cm = np.concatenate([Dc_1_2, Dc_3_2, Dc_6_2, Dc_10_2])
    Df_20_25_cm = np.concatenate([Df_1_2, Df_3_2, Df_6_2, Df_10_2])
    Dc_40_45_cm = np.concatenate([Dc_1_3, Dc_3_3, Dc_6_3, Dc_10_3])
    Df_40_45_cm = np.concatenate([Df_1_3, Df_3_3, Df_6_3, Df_10_3])
    
    a_1_1 = df_a['a_1kPa'][0:7]
    a_1_2 = df_a['a_1kPa'][7:14]
    a_1_3 = df_a['a_1kPa'][14:21]
    
    nonnans_1_1 = ~np.isnan(df_a['a_1kPa'][0:7])
    nonnans_1_2 = ~np.isnan(df_a['a_1kPa'][7:14])
    nonnans_1_3 = ~np.isnan(df_a['a_1kPa'][14:21])
    
    a_3_1 = df_a['a_3kPa'][0:7]
    a_3_2 = df_a['a_3kPa'][7:14]
    a_3_3 = df_a['a_3kPa'][14:23]
    
    nonnans_3_1 = ~np.isnan(df_a['a_3kPa'][0:7])
    nonnans_3_2 = ~np.isnan(df_a['a_3kPa'][7:14])
    nonnans_3_3 = ~np.isnan(df_a['a_3kPa'][14:21])
        
    a_6_1 = df_a['a_6kPa'][0:7]
    a_6_2 = df_a['a_6kPa'][7:14]
    a_6_3 = df_a['a_6kPa'][14:21]
    
    nonnans_6_1 = ~np.isnan(df_a['a_6kPa'][0:7])
    nonnans_6_2 = ~np.isnan(df_a['a_6kPa'][7:14])
    nonnans_6_3 = ~np.isnan(df_a['a_6kPa'][14:21])
    
    a_10_1 = df_a['a_10kPa'][0:7]
    a_10_2 = df_a['a_10kPa'][7:14]
    a_10_3 = df_a['a_10kPa'][14:21]
    
    a_10_1_all = df_a['a_all_10kPa'][0:7]
    a_10_2_all = df_a['a_all_10kPa'][7:14]
    a_10_3_all = df_a['a_all_10kPa'][14:21]
    
    nonnans_10_1 = ~np.isnan(df_a['a_10kPa'][0:7])
    nonnans_10_2 = ~np.isnan(df_a['a_10kPa'][7:14])
    nonnans_10_3 = ~np.isnan(df_a['a_10kPa'][14:21])
    
    fa_1 = df_a['f_a'][0:7].dropna()
    fa_2 = df_a['f_a'][7:14].dropna()
    fa_3 = df_a['f_a'][14:21].dropna()
    
    a_0_5_cm = np.concatenate([a_1_1[nonnans_1_1], a_3_1[nonnans_3_1], a_6_1[nonnans_6_1], a_10_1[nonnans_10_1]])
    a_20_25_cm = np.concatenate([a_1_2[nonnans_1_2], a_3_2[nonnans_3_2], a_6_2[nonnans_6_2], a_10_2[nonnans_10_2]])
    a_40_45_cm = np.concatenate([a_1_3[nonnans_1_3], a_3_3[nonnans_3_3], a_6_3[nonnans_6_3], a_10_3[nonnans_10_3]])
  

# Homogeneity of variance


if variance_homogen:

    
    print('')
    print('BARTLETT')
    print('')
    
    print('Dc_1_kPa')
    
    print(stats.bartlett(Dc_1_1, Dc_1_2, Dc_1_3))
    print([np.var(x, ddof=1) for x in [Dc_1_1, Dc_1_2, Dc_1_3]])
    
    print('Dc_3_kPa')
    
    print(stats.bartlett(Dc_3_1, Dc_3_2, Dc_3_3))
    print([np.var(x, ddof=1) for x in [Dc_3_1, Dc_3_2, Dc_3_3]])
    
    print('Dc_6_kPa')
    
    print(stats.bartlett(Dc_6_1, Dc_6_2, Dc_6_3))
    print([np.var(x, ddof=1) for x in [Dc_6_1, Dc_6_2, Dc_6_3]])
    
    print('Dc_10_kPa')
    
    print(stats.bartlett(Dc_10_1, Dc_10_2, Dc_10_3))
    print([np.var(x, ddof=1) for x in [Dc_10_1, Dc_10_2, Dc_10_3]])
    
    print('Df_1_kPa')
    
    print(stats.bartlett(Df_1_1, Df_1_2, Df_1_3))
    print([np.var(x, ddof=1) for x in [Df_1_1, Df_1_2, Df_1_3]])
    
    print('Df_3_kPa')
    
    print(stats.bartlett(Df_3_1, Df_3_2, Df_3_3))
    print([np.var(x, ddof=1) for x in [Df_3_1, Df_3_2, Df_3_3]])
    
    print('Df_6_kPa')
    
    print(stats.bartlett(Df_6_1, Df_6_2, Df_6_3))
    print([np.var(x, ddof=1) for x in [Df_6_1, Df_6_2, Df_6_3]])
    
    print('Df_10_kPa')
    
    print(stats.bartlett(Df_10_1, Df_10_2, Df_10_3))
    print([np.var(x, ddof=1) for x in [Df_10_1, Df_10_2, Df_10_3]])


    
    print('')
    print('LEVENE')
    print('')
    
    print('Dc_1_kPa')
    
    print(stats.levene(Dc_1_1, Dc_1_2, Dc_1_3))
    print('2 & 3')
    print(stats.levene(Dc_1_2, Dc_1_3))
    print([np.var(x, ddof=1) for x in [Dc_1_1, Dc_1_2, Dc_1_3]])
    
    print('Dc_3_kPa')
    
    print(stats.levene(Dc_3_1, Dc_3_2, Dc_3_3))
    print('2 & 3')
    print(stats.levene(Dc_3_2, Dc_3_3))
    print([np.var(x, ddof=1) for x in [Dc_3_1, Dc_3_2, Dc_3_3]])
    
    print('Dc_6_kPa')
    
    print(stats.levene(Dc_6_1, Dc_6_2, Dc_6_3))
    print('2 & 3')
    print(stats.levene(Dc_6_2, Dc_6_3))
    print([np.var(x, ddof=1) for x in [Dc_6_1, Dc_6_2, Dc_6_3]])
    
    print('Dc_10_kPa')
    
    print(stats.levene(Dc_10_1, Dc_10_2, Dc_10_3))
    print('2 & 3')
    print(stats.levene(Dc_10_2, Dc_10_3))
    print([np.var(x, ddof=1) for x in [Dc_10_1, Dc_10_2, Dc_10_3]])
    
    print('Df_1_kPa')
    
    print(stats.levene(Df_1_1, Df_1_2, Df_1_3))
    print('2 & 3')
    print(stats.levene(Df_1_2, Df_1_3))
    print([np.var(x, ddof=1) for x in [Df_1_1, Df_1_2, Df_1_3]])
    
    print('Df_3_kPa')
    
    print(stats.levene(Df_3_1, Df_3_2, Df_3_3))
    print('2 & 3')
    print(stats.levene(Df_3_2, Df_3_3))
    print([np.var(x, ddof=1) for x in [Df_3_1, Df_3_2, Df_3_3]])
    
    print('Df_6_kPa')
    
    print(stats.levene(Df_6_1, Df_6_2, Df_6_3))
    print('2 & 3')
    print(stats.levene(Df_6_2, Df_6_3))
    print([np.var(x, ddof=1) for x in [Df_6_1, Df_6_2, Df_6_3]])
    
    print('Df_10_kPa')
    
    print(stats.levene(Df_10_1, Df_10_2, Df_10_3))
    print('2 & 3')
    print(stats.levene(Df_10_2, Df_10_3))
    print([np.var(x, ddof=1) for x in [Df_10_1, Df_10_2, Df_10_3]])
    

# ANOVA

if anova:
    
    print('')
    print('Standard ANOVA, F-ONEWAY')
    print('')
    
    print('Dc_1_kPa')
    
    print(stats.f_oneway(Dc_1_1, Dc_1_2, Dc_1_3))
    
    print('Dc_3_kPa')
    
    print(stats.f_oneway(Dc_3_1, Dc_3_2, Dc_3_3))
    
    print('Dc_6_kPa')
    
    print(stats.f_oneway(Dc_6_1, Dc_6_2, Dc_6_3))
    
    print('Dc_10_kPa')
    
    print(stats.f_oneway(Dc_10_1, Dc_10_2, Dc_10_3))
    
    print('Df_1_kPa')
    
    print(stats.f_oneway(Df_1_1, Df_1_2, Df_1_3))
    
    print('Df_3_kPa')
    
    print(stats.f_oneway(Df_3_1, Df_3_2, Df_3_3))
    
    print('Betweenness centrality')
    
    print(stats.f_oneway(Df_6_1, Df_6_2, Df_6_3))
    
    print('Df_10_kPa')
    
    print(stats.f_oneway(Df_10_1, Df_10_2, Df_10_3))
    

if boxpl:

    bbdim = [0.16, 0.36]
    
    bppos = [
            [0.08, 0.57, bbdim[0],bbdim[1]],
            [0.325, 0.57, bbdim[0],bbdim[1]],
            [0.57, 0.57, bbdim[0],bbdim[1]],
            [0.815, 0.57, bbdim[0],bbdim[1]],
            [0.08, 0.095, bbdim[0],bbdim[1]],
            [0.325, 0.095, bbdim[0],bbdim[1]],
            [0.57, 0.095, bbdim[0],bbdim[1]],
            [0.815, 0.095, bbdim[0],bbdim[1]]
            ]
    
    subtpos = [0.01,1.06]
    
    subt = [('(a)'), ('(b)'), ('(c)'), ('(d)'),
            ('(e)'), ('(f)'), ('(g)'), ('(h)'), ('(i)')]
    
    ylabs = [
            'Ds -1 kPa',
            'Ds -3 kPa',
            'Ds -6 kPa',
            'Ds -10 kPa',
            'Closeness',
            'Betweenness',
            'Top',
            'xxx',
            '',
            '',
            '',
            'Horizon',
            ]
    
    ylc = [-0.29, -0.23, -0.23, -0.27 ,-0.23,
           -0.25, -0.27, -0.23, -0.00, -0.00, 
           -0.00, -0.00, -0.27]
    
    medianprops = dict(linestyle='-', linewidth=1, color='blue')
    flierprops = dict(marker='o', markersize=3,markerfacecolor='white',markeredgecolor='k')#, linestyle='none')
    
    fig = plt.figure(num=101)
    fig.set_size_inches(6,3.2)
    plt.clf()
    
    bporder = np.array([3,4,5,6,3,4,5,6])
    
    for j in range(0,8):
        
        ax = fig.add_subplot(2,4,j+1)
        
        i = bporder[j]
        
        if i == 3:
            factor = 1
        elif i == 8 or i == 9:
            factor = 1
        else:
            factor = 1
        
        if j < 4:
            data = [factor*df[df.columns[i]][df['Depth'] == 1].dropna(),
                factor*df[df.columns[i]][df['Depth'] == 2].dropna(),
                factor*df[df.columns[i]][df['Depth'] == 3].dropna()]
        else:
            data = [factor*df_a[df_a.columns[i]][df['Depth'] == 1].dropna(),
                factor*df_a[df_a.columns[i]][df['Depth'] == 2].dropna(),
                factor*df_a[df_a.columns[i]][df['Depth'] == 3].dropna()]
        
        ax.boxplot(data, labels= ['0-5', '20-25', '40-45'],
                   showmeans=False, medianprops=medianprops,
                   flierprops=flierprops)
        
        
        
        if i-2 > -2:
            ax.text(0.13, 0.9, "A", transform=ax.transAxes)
            ax.text(0.47, 0.9, "B", transform=ax.transAxes)
            ax.text(0.8, 0.9, "B", transform=ax.transAxes)
        
        #ax.set_title(subt[i-2], horizontalalignment='right')
        
        ax.text(subtpos[0], subtpos[1], subt[j], transform=ax.transAxes)
        
        if j < 4:
            ax.set_ylim([0, 1.2 * factor * np.max(df[df.columns[i]])])
        else:
            ax.set_ylim([0, 1.2 * factor * np.max(df_a[df_a.columns[i]])])
        
        if i-1 > 4:
            ax.set_xlabel("Depth (cm)")
            ax.xaxis.set_label_coords(0.5, -0.15)
         
        ax.xaxis.set_tick_params(length=0)
        
        ax.set_ylabel(ylabs[j], labelpad=0.9)
        #ax.yaxis.set_label_coords(ylc[j], 0.5)
        
        ax.set_position(bppos[j])
        
        #plt.savefig('metrics.pdf')


def probability_plot_ols(fignum, data_1):

    fig = plt.figure(num=fignum)
    fig.set_size_inches(4,4)
    plt.clf()
    ax1 = fig.add_subplot(111)
    stats.probplot(data_1, plot= plt, rvalue= True)
    ax1.set_title(str(fignum))
    plt.tight_layout()

if anova_ols:
        print('')
        print('Standard ANOVA ols fit')
        print('')
        
        for w in range(1,11):
            df.rename(columns={df.columns[w]: df.columns[w].replace(" ", "x")},inplace=True)
            df.rename(columns={df.columns[w]: df.columns[w].replace(".", "q")},inplace=True)
            df.rename(columns={df.columns[w]: df.columns[w].replace("-", "Q")},inplace=True)
        
        st_p_conn = ols('Dc_1kPa ~ C(Depth)', data=df).fit()
        print('-------- Porosity of connected cluster ----------')
        print(sm.stats.anova_lm(st_p_conn, typ=2))
        print('Residual Shapiro ' + str(stats.shapiro(st_p_conn.resid)))
        print(stats.anderson(st_p_conn.resid))
        probability_plot_ols(1, st_p_conn.resid)
        
        st_np = ols('Dc_3kPa ~ C(Depth)', data=df).fit()
        print('-------- Number of pores ----------')
        print(sm.stats.anova_lm(st_np, typ=2))
        print('Residual Shapiro ' + str(stats.shapiro(st_np.resid)))
        print(stats.anderson(st_np.resid))
        probability_plot_ols(2, st_np.resid)
        
        st_cn = ols('Dc_6kPa ~ C(Depth)', data=df).fit()
        print('-------- Coordination number ----------')
        print(sm.stats.anova_lm(st_cn, typ=2))
        print('Residual Shapiro ' + str(stats.shapiro(st_cn.resid)))
        print(stats.anderson(st_cn.resid))
        probability_plot_ols(3, st_cn.resid)
        
        st_cc = ols('Dc_10kPa ~ C(Depth)', data=df).fit()
        print('-------- Clustering coefficient ----------')
        print(sm.stats.anova_lm(st_cc, typ=2))
        print('Residual Shapiro ' + str(stats.shapiro(st_cc.resid)))
        print(stats.anderson(st_cc.resid))
        probability_plot_ols(4, st_cc.resid)
        
        st_gt = ols('Df_1kPa ~ C(Depth)', data=df).fit()
        print('-------- Geometric tortuosity ----------')
        print(sm.stats.anova_lm(st_gt, typ=2))
        print('Residual Shapiro ' + str(stats.shapiro(st_gt.resid)))
        print(stats.anderson(st_gt.resid))
        probability_plot_ols(5, st_gt.resid)
        
        st_ccntr = ols('Df_3kPa ~ C(Depth)', data=df).fit()
        print('-------- Closeness centrality ----------')
        print(sm.stats.anova_lm(st_ccntr, typ=2))
        print('Residual Shapiro ' + str(stats.shapiro(st_ccntr.resid)))
        print(stats.anderson(st_ccntr.resid))
        probability_plot_ols(6, st_ccntr.resid)
        
        
        st_bcntr = ols('Df_6kPa ~ C(Depth)', data=df).fit()
        print('-------- Betweenness centrality ----------')
        print(sm.stats.anova_lm(st_bcntr, typ=2))
        print('Residual Shapiro ' + str(stats.shapiro(st_bcntr.resid)))
        print(stats.anderson(st_bcntr.resid))
        probability_plot_ols(7, st_bcntr.resid)
        
        st_tdbc = ols('Df_10kPa ~ C(Depth)', data=df).fit()
        print('-------- Top-down betweenness centrality ----------')
        print(sm.stats.anova_lm(st_tdbc, typ=2))
        print('Residual Shapiro ' + str(stats.shapiro(st_tdbc.resid)))
        print(stats.anderson(st_tdbc.resid))
        probability_plot_ols(8, st_tdbc.resid)
        

        
        for w in range(1,11):
            df.rename(columns={df.columns[w]: df.columns[w].replace("x", " ")},inplace=True)
            df.rename(columns={df.columns[w]: df.columns[w].replace("q", ".")},inplace=True)
            df.rename(columns={df.columns[w]: df.columns[w].replace("Q", "-")},inplace=True)
  
        try:
            print('')
            print('Welch ANOVA')
            print('')
            
            welch = ao.anova_oneway(Dc_1_mod, depths_mod_1,
                               use_var='unequal', welch_correction=True)
            print('-------- Porosity of connected cluster ----------')
            print(welch)
            
            welch = ao.anova_oneway(Dc_3_mod, depths_mod_3,
                               use_var='unequal', welch_correction=True)
            print('-------- Number of pores ----------')
            print(welch)
            
            welch = ao.anova_oneway(Dc_6_mod, depths_mod_6,
                               use_var='unequal', welch_correction=True)
            print('-------- Coordination number ----------')
            print(welch)
            
            welch = ao.anova_oneway(Dc_10_mod, depths_mod_10,
                               use_var='unequal', welch_correction=True)
            print('-------- Clustering coefficient ----------')
            print(welch)
            
            welch = ao.anova_oneway(Df_1_mod, depths_mod_1,
                               use_var='unequal', welch_correction=True)
            print('-------- Geometric tortuosity ----------')
            print(welch)
            
            welch = ao.anova_oneway(Df_3_mod, depths_mod_3,
                               use_var='unequal', welch_correction=True)
            print('-------- Geomtorthor1 ----------')
            print(welch)
            
            welch = ao.anova_oneway(Df_6_mod, depths_mod_6,
                               use_var='unequal', welch_correction=True)
            print('-------- Geomtorthor2 ----------')
            print(welch)
            
            welch = ao.anova_oneway(Df_10_mod, depths_mod_10,
                               use_var='unequal', welch_correction=True)
            print('-------- Horizontal geometric tortuosity ----------')
            print(welch)
            
        
        except Exception:
        
            pass
    
    
        print('')
        print('KRUSKAL')
        print('')
        
        print('Porosity of connected cluster')
        
        print(stats.kruskal(Dc_1_1, Dc_1_2, Dc_1_3))
        
        print('Number of pores')
        
        print(stats.kruskal(Dc_3_1, Dc_3_2, Dc_3_3))
        
        print('Coordination number')
        
        print(stats.kruskal(Dc_6_1, Dc_6_2, Dc_6_3))
        
        print('Clustering coefficient')
        
        print(stats.kruskal(Dc_10_1, Dc_10_2, Dc_10_3))
        
        print('Geometric tortuosity')
        
        print(stats.kruskal(Df_1_1, Df_1_2, Df_1_3))
        
        print('Geometric tortuosity h1')
        
        print(stats.kruskal(Df_3_1, Df_3_2, Df_3_3))
        
        print('Geometric tortuosity h2')
        
        print(stats.kruskal(Df_6_1, Df_6_2, Df_6_3))
        
        print('Geometric tortuosity horizontal')
        
        print(stats.kruskal(Df_10_1, Df_10_2, Df_10_3))
        
        
if tukey:
    
    print('Tukey')
    print('')
    
    comp = mc.MultiComparison(Dc_1_mod, depths_mod_1)
    post_hoc_res = comp.tukeyhsd()
    print('Poros. of conn. cluster')
    print(post_hoc_res.summary())
    
    comp = mc.MultiComparison(Dc_3_mod, depths_mod_3)
    post_hoc_res = comp.tukeyhsd()
    print('Number of pores')
    print(post_hoc_res.summary())
    
    comp = mc.MultiComparison(Dc_6_mod, depths_mod_6)
    post_hoc_res = comp.tukeyhsd()
    print('Coordination number')
    print(post_hoc_res.summary())
    
    comp = mc.MultiComparison(Dc_10_mod, depths_mod_10)
    post_hoc_res = comp.tukeyhsd()
    print('Clustering coefficient')
    print(post_hoc_res.summary())
    
    comp = mc.MultiComparison(Df_1_mod, depths_mod_1)
    post_hoc_res = comp.tukeyhsd()
    print('Geometr. tortuosity')
    print(post_hoc_res.summary())
    
    comp = mc.MultiComparison(Df_3_mod, depths_mod_3)
    post_hoc_res = comp.tukeyhsd()
    print('Geomtorthor1')
    print(post_hoc_res.summary())
    
    comp = mc.MultiComparison(Df_6_mod, depths_mod_6)
    post_hoc_res = comp.tukeyhsd()
    print('Geomtorthor2')
    print(post_hoc_res.summary())
    
    comp = mc.MultiComparison(Df_10_mod, depths_mod_10)
    post_hoc_res = comp.tukeyhsd()
    print('Horizontal geometric tortuosity')
    print(post_hoc_res.summary())
    

if games:
    
    print('Games')
    print('')
    
    hp_games = hp.posthoc.GamesHowell(Dc_1_mod, group=depths_mod_1, alpha=0.05)
    print('Porosity of connected cluster')
    print(hp_games.test_result)
    print(hp_games.test_result['p_value'])
    
    hp_games = hp.posthoc.GamesHowell(Dc_3_mod, group=depths_mod_3, alpha=0.05)
    print('Number of pores')
    print(hp_games.test_result)
    print(hp_games.test_result['p_value'])
    
    hp_games = hp.posthoc.GamesHowell(Dc_6_mod, group=depths_mod_6, alpha=0.05)
    print('Coordination number')
    print(hp_games.test_result)
    print(hp_games.test_result['p_value'])
    
    hp_games = hp.posthoc.GamesHowell(Dc_10_mod, group=depths_mod_10, alpha=0.05)
    print('Clustering coefficient')
    print(hp_games.test_result)
    print(hp_games.test_result['p_value'])
    
    hp_games = hp.posthoc.GamesHowell(Df_1_mod, group=depths_mod_1, alpha=0.05)
    print('Geometric tortuosity')
    print(hp_games.test_result)
    print(hp_games.test_result['p_value'])
    
    hp_games = hp.posthoc.GamesHowell(Df_3_mod, group=depths_mod_3, alpha=0.05)
    print('Geomtorthor1')
    print(hp_games.test_result)
    print(hp_games.test_result['p_value'])
    
    hp_games = hp.posthoc.GamesHowell(Df_6_mod, group=depths_mod_6, alpha=0.05)
    print('Geomtorthor2')
    print(hp_games.test_result)
    print(hp_games.test_result['p_value'])
    
    hp_games = hp.posthoc.GamesHowell(Df_10_mod, group=depths_mod_10, alpha=0.05)
    print('Horizontal geometric tortuosity')
    print(hp_games.test_result)
    print(hp_games.test_result['p_value'])
    
    
if dunn:
    
    print('')
    print('Dunn-Bonferroni')
    print('')
    
    print('Dc 1 kPa')
    ph_dunn = sp.posthoc_dunn([Dc_1_1,Dc_1_2,Dc_1_3], p_adjust = 'holm')    
    print(ph_dunn)
    
    print('Dc 3')
    ph_dunn = sp.posthoc_dunn([Dc_3_1,Dc_3_2,Dc_3_3], p_adjust = 'holm')    
    print(ph_dunn)
    
    print('Dc 6')
    ph_dunn = sp.posthoc_dunn([Dc_6_1,Dc_6_2,Dc_6_3], p_adjust = 'holm')    
    print(ph_dunn)
    
    print('Dc 10')
    ph_dunn = sp.posthoc_dunn([Dc_10_1,Dc_10_2,Dc_10_3], p_adjust = 'holm')    
    print(ph_dunn)
    
    print('Df 1')
    ph_dunn = sp.posthoc_dunn([Df_1_1,Df_1_2,Df_1_3], p_adjust = 'holm')    
    print(ph_dunn)
    
    print('Df 3')
    ph_dunn = sp.posthoc_dunn([Df_3_1,Df_3_2,Df_3_3], p_adjust = 'holm')    
    print(ph_dunn)
    
    print('Df 6')
    ph_dunn = sp.posthoc_dunn([Df_6_1,Df_6_2,Df_6_3], p_adjust = 'holm')    
    print(ph_dunn)
    
    print('Df 10')
    ph_dunn = sp.posthoc_dunn([Df_10_1,Df_10_2,Df_10_3], p_adjust = 'holm')    
    print(ph_dunn)
    
    

if draw_f_9:
    
    def d_mq(diffcoeff, airfilledporos, porosity):
        
        return diffcoeff * airfilledporos**(10/3) / porosity**2
    
    def d_tpm(diffcoeff, airfilledporos, airfilledporos100, porosity):
        
        X = np.log10((2*airfilledporos100**3+0.04*airfilledporos100)/porosity**2) / \
        np.log10(airfilledporos100/porosity)
        
        return diffcoeff * porosity**2 * (airfilledporos/porosity)**(X)
    
    # 0-5 cm Df
    
    fig = plt.figure(num=109)
    
    fig.set_size_inches(7.,6.)
    plt.clf()
    ax = fig.add_subplot(1,1,1)
    
    D0_N2 = 1#0.2
    
    ax.scatter(a_1_1[nonnans_1_1], Df_1_1/D0_N2, c='b',s=10,label='1 kPa')
    ax.scatter(a_3_1[nonnans_3_1], Df_3_1/D0_N2, c='orange',s=10,label='3 kPa')
    ax.scatter(a_6_1[nonnans_6_1], Df_6_1/D0_N2, c='g',s=10,label='6 kPa')
    ax.scatter(a_10_1[nonnans_10_1], Df_10_1/D0_N2, c='r',s=10,label='10 kPa')
    
    por = np.mean(fa_1)#0.9
    
    #a_s = np.linspace(0,0.80,100)
    
    a_all_1 = np.concatenate([a_1_1[nonnans_1_1],a_3_1[nonnans_3_1],a_6_1[nonnans_6_1],a_10_1[nonnans_10_1]])
    args_1 = np.argsort(a_all_1)
    f_all_1_100 = np.concatenate([fa_1[nonnans_1_1 & nonnans_10_1],fa_1[nonnans_3_1 & nonnans_10_1],fa_1[nonnans_6_1 & nonnans_10_1],fa_1[nonnans_10_1]])
    f_all_1 = np.concatenate([fa_1[nonnans_1_1],fa_1[nonnans_3_1],fa_1[nonnans_6_1],fa_1[nonnans_10_1]])
    
    a_100s = np.concatenate([a_10_1[nonnans_1_1 & nonnans_10_1],a_10_1[nonnans_3_1 & nonnans_10_1],a_10_1[nonnans_6_1 & nonnans_10_1],a_10_1[nonnans_10_1]])
    X_1_all = np.log10((2*a_100s**3+0.04*a_100s)/f_all_1_100**2) / np.log10(a_100s/f_all_1_100)
    
    a_s = np.linspace(0,np.max(a_all_1[~np.isnan(a_all_1)]),100)
    
    D0 = 0.202#1#0.2
    
    D_MQ = D0 * a_s**(10/3) / por**2
    D_MQ_1_indiv = D0 * a_all_1[args_1]**(10/3) / f_all_1[args_1]**2
    
      
    #D_BBC_1 = D0 * a_s**(2+3/2.9) / por**2.9
    #D_BBC_1 = D0 * 0.66 * por * (a_s/por)**((12-3)/3)
    D_BBC_1 = D0 * a_s**(2.) / por**(2./3)
    
    D_BBC_1_indiv = D0 * a_all_1[args_1]**(2) / f_all_1[args_1]**(2/3)
        
    a_s_1 = np.copy(a_s)
    D_MQ_1 = np.copy(D_MQ)

    D_Cam = D0 * 0.9*a_s**2.3
    
    a100_1_av = np.mean(a_10_1_all[0:])
    fa_1_av = np.mean(fa_1[0:])
    
    X_1 = np.log10((2*a100_1_av**3+0.04*a100_1_av)/fa_1_av**2) / np.log10(a100_1_av/fa_1_av)
    
    D_TPM_1 = D0 * fa_1_av**2 * (a_s/fa_1_av)**(X_1)
    
    #D_TPM_indiv = d_tpm(D0,a_all_1[args_1],a_100s,f_all_1)
    #D_TPM_indiv = D0 * f_all_1[args_1]**2 * (a_all_1[args_1]/f_all_1[args_1])**(X_1_all[args_1])
    #import pdb
    #pdb.set_trace()
    ax.plot(a_s,D_MQ_1,'k-.',lw=.75,label='MQ model')
    ax.plot(a_s,D_TPM_1,'b-.',lw=.75,label='TPM model')
    ax.plot(a_s,D_Cam,'r-.',lw=.75,label='Campbell model')
    ax.plot(a_s,D_BBC_1,'r-.',lw=.75,label='BBC model')
    ax.plot(a_all_1[args_1],D_BBC_1_indiv,'g-.',lw=.75,label='BBC model individual')
    ax.plot(a_all_1[args_1],D_MQ_1_indiv,'g-.',lw=.75,label='MQ model individual')
    #ax.plot(a_all_1[args_1],D_TPM_indiv,'g-.',lw=.75,label='TPM model individual')
    
    ax.legend()
    
    ax.set_xlim([0.1, 0.7])
    ax.set_ylim([0.0, 0.145])
    
    # 0-5 cm Dc
    
    fig = plt.figure(num=1090)
    
    fig.set_size_inches(7.,6.)
    plt.clf()
    ax = fig.add_subplot(1,1,1)
    
    ax.scatter(a_1_1[nonnans_1_1], Dc_1_1/D0_N2, marker='s', c='b',s=10,label='1 kPa')
    ax.scatter(a_3_1[nonnans_3_1], Dc_3_1/D0_N2, marker='o', c='orange',s=10,label='3 kPa')
    ax.scatter(a_6_1[nonnans_6_1], Dc_6_1/D0_N2, marker='^', c='g',s=10,label='6 kPa')
    ax.scatter(a_10_1[nonnans_10_1], Dc_10_1/D0_N2, marker='D', c='k',s=10,label='10 kPa')
    
    ax.plot(a_s,D_MQ_1,'k-.',lw=.75,label='MQ model')
    ax.plot(a_s,D_TPM_1,'b-.',lw=.75,label='TPM model')
    ax.plot(a_s,D_Cam,'r-.',lw=.75,label='Campbell model')
    ax.plot(a_all_1[args_1],D_MQ_1_indiv,'g-.',lw=.75,label='MQ model individual')
    #ax.plot(a_all_1[args_1],D_TPM_indiv,'g-.',lw=.75,label='TPM model individual')
    
    ax.legend()
    plt.title('0-5 cm Dc')
    ax.set_xlim([0.14, 0.75])
    ax.set_ylim([0.0, 0.045])
    
    #--------------------------------------------------------------------------
    
    #0-5 cm

    
    fig = plt.figure(num=2090)
    
    fig.set_size_inches(9.,9.)
    plt.clf()
    ax = fig.add_subplot(1,1,1)
    
    ax.plot(a_s,D_MQ_1, c='gray', ls='--',lw=.75,label='MQ model')
    ax.plot(a_s,D_TPM_1,c='gray', ls=':',lw=.75,label='TPM model')
    ax.plot(a_s,D_Cam,c='gray', ls='-',lw=.75,label='Campbell model')
    ax.plot(a_s,D_BBC_1, c='gray',ls='-.',lw=.75,label='BBC model')
    ax.scatter(a_all_1[args_1],D_MQ_1_indiv,marker='^', c='gray',s=8)
    ax.scatter(a_all_1[args_1],D_BBC_1_indiv,marker='^', c='gray',s=8)
    #ax.plot(a_all_1[args_1],D_TPM_indiv,'g-.',lw=.75,label='TPM model individual')
    ax.scatter(a_1_1[nonnans_1_1], Df_1_1/D0_N2, marker='^', c='gray',s=8)
    ax.scatter(a_3_1[nonnans_3_1], Df_3_1/D0_N2, marker='^', c='gray',s=8)
    ax.scatter(a_6_1[nonnans_6_1], Df_6_1/D0_N2, marker='^', c='gray',s=8)
    ax.scatter(a_10_1[nonnans_10_1], Df_10_1/D0_N2, marker='^', c='gray',s=8)
    
    ax.scatter(a_1_1[nonnans_1_1], Dc_1_1/D0_N2, marker='o', c='b',s=14,label='1 kPa')
    ax.scatter(a_3_1[nonnans_3_1], Dc_3_1/D0_N2, marker='o', c='orange',s=14,label='3 kPa')
    ax.scatter(a_6_1[nonnans_6_1], Dc_6_1/D0_N2, marker='o', c='g',s=14,label='6 kPa')
    ax.scatter(a_10_1[nonnans_10_1], Dc_10_1/D0_N2, marker='o', c='k',s=14,label='10 kPa')
    
    ax.legend()
    plt.title('TAMA')
    ax.set_xlim([0.14, 0.65])
    ax.set_ylim([0.0, 0.075])
    
    
    
    #--------------------------------------------------------------------------
    
    # 20-25 cm Df
    
    fig = plt.figure(num=110)
    
    fig.set_size_inches(7.,6.)
    plt.clf()
    ax = fig.add_subplot(1,1,1)
    
    D0_N2 = 1#0.2
    
    ax.scatter(a_1_2[nonnans_1_2], Df_1_2/D0_N2, c='b',s=10,label='1 kPa')
    ax.scatter(a_3_2[nonnans_3_2], Df_3_2/D0_N2, c='orange',s=10,label='3 kPa')
    ax.scatter(a_6_2[nonnans_6_2], Df_6_2/D0_N2, c='g',s=10,label='6 kPa')
    ax.scatter(a_10_2[nonnans_10_2], Df_10_2/D0_N2, c='r',s=10,label='10 kPa')
    
    por = np.mean(fa_2)#0.9
    
    a_s = np.linspace(0,0.80,100)
    
    a_all_2 = np.concatenate([a_1_2[nonnans_10_2],a_3_2[nonnans_10_2],a_6_2[nonnans_10_2],a_10_2[nonnans_10_2]])
    args_2 = np.argsort(a_all_2)
    f_all_2 = np.concatenate([fa_2[nonnans_10_2],fa_2[nonnans_10_2],fa_2[nonnans_10_2],fa_2[nonnans_10_2]])
    
    a_100s_2 = np.concatenate([a_10_2[nonnans_10_2],a_10_2[nonnans_10_2],a_10_2[nonnans_10_2],a_10_2[nonnans_10_2]])
    X_2_all = np.log10((2*a_100s_2**3+0.04*a_100s_2)/f_all_2**2) / np.log10(a_100s_2/f_all_2)
    
    a_s = np.linspace(0,np.max(a_all_2[~np.isnan(a_all_2)]),100)
    
    D0 = 0.202#1#0.2
    
    D_MQ = D0 * a_s**(10/3) / por**2
    D_MQ_2_indiv = D0 * a_all_2[args_2]**(10/3) / f_all_2[args_2]**2
    
    D_BBC_2 = D0 * a_s**(2+3/2.9) / por**2.9
    #D_BBC_2 = D0 * 0.66 * por * (a_s/por)**((12-3)/3)
    D_BBC_2 = D0 * a_s**(2.) / por**(2./3)
    
    
    D_BBC_2_indiv = D0 * a_all_2[args_2]**(2) / f_all_2[args_2]**(2/3)
    
    a_s_2 = np.copy(a_s)
    D_MQ_2 = np.copy(D_MQ)

    D_Cam_2 = D0 * 0.9*a_s**2.3
    
    a100_2_av = np.mean(a_10_2_all)
    fa_2_av = np.mean(fa_2)
    
    X_2 = np.log10((2*a100_2_av**3+0.04*a100_2_av)/fa_2_av**2) / np.log10(a100_2_av/fa_2_av)
    
    D_TPM_2 = d_tpm(D0,a_s,a100_2_av,fa_2_av)#    D0 * fa_1_av**2 * (a_s/fa_1_av)**(X_1)
    
    D_TPM_2_indiv = d_tpm(D0,a_all_2[args_2],a_100s_2[args_2],f_all_2[args_2])
    #D0 * f_all_1[args_1]**2 * (a_all_1[args_1]/f_all_1[args_1])**(X_1_all[args_1])
    
    ax.plot(a_s,D_MQ_2,'k-.',lw=.75,label='MQ model')
    ax.plot(a_s,D_TPM_2,'b-.',lw=.75,label='TPM model')
    ax.plot(a_s,D_Cam_2,'r-.',lw=.75,label='Campbell model')
    ax.plot(a_s,D_BBC_2,'r-.',lw=.75,label='BBC model')
    ax.plot(a_all_2[args_2],D_BBC_2_indiv,'g-.',lw=.75,label='BBC model individual')
    ax.plot(a_all_2[args_2],D_MQ_2_indiv,'g-.',lw=.75,label='MQ model individual')
    ax.plot(a_all_2[args_2],D_TPM_2_indiv,'g-.',lw=.75,label='TPM model individual')
    
    ax.legend()
    
    ax.set_xlim([0.051, 0.36])
    ax.set_ylim([0.0, 0.009])
    
    # 20-25 cm Dc
    
    fig = plt.figure(num=1091)
    
    fig.set_size_inches(7.,6.)
    plt.clf()
    ax = fig.add_subplot(1,1,1)
    
    ax.scatter(a_1_2[nonnans_1_2], Dc_1_2/D0_N2, marker='s', c='b',s=10,label='1 kPa')
    ax.scatter(a_3_2[nonnans_3_2], Dc_3_2/D0_N2, marker='o', c='orange',s=10,label='3 kPa')
    ax.scatter(a_6_2[nonnans_6_2], Dc_6_2/D0_N2, marker='^', c='g',s=10,label='6 kPa')
    ax.scatter(a_10_2[nonnans_10_2], Dc_10_2/D0_N2, marker='D', c='k',s=10,label='10 kPa')
    
    ax.plot(a_s,D_MQ_2,'k-.',lw=.75,label='MQ model')
    ax.plot(a_s,D_TPM_2,'b-.',lw=.75,label='TPM model')
    ax.plot(a_s,D_Cam_2,'r-.',lw=.75,label='Campbell model')
    #ax.plot(a_all_2[args_2],D_MQ_indiv,'g-.',lw=.75,label='MQ model individual')
    #ax.plot(a_all_2[args_2],D_TPM_indiv,'g-.',lw=.75,label='TPM model individual')
    
    ax.legend()
    plt.title('20-25 cm Dc')
    ax.set_xlim([0.05, 0.35])
    ax.set_ylim([-0.0005, 0.015])
    
    #--------------------------------------------------------------------------
    
    #20-25 cm

    
    fig = plt.figure(num=3090)
    
    fig.set_size_inches(9.,9.)
    plt.clf()
    ax = fig.add_subplot(1,1,1)
    
    ax.plot(a_s,D_MQ_2, c='gray', ls='--',lw=.75,label='MQ model')
    ax.plot(a_s,D_TPM_2,c='gray', ls=':',lw=.75,label='TPM model')
    ax.plot(a_s,D_Cam_2,c='gray', ls='-',lw=.75,label='Campbell model')
    ax.plot(a_s,D_BBC_2, c='gray',ls='-.',lw=.75,label='BBC model')
    ax.scatter(a_all_2[args_2],D_MQ_2_indiv,marker='^', c='gray',s=8)
    ax.scatter(a_all_2[args_2],D_BBC_2_indiv,marker='^', c='gray',s=8)
    #ax.plot(a_all_1[args_1],D_TPM_indiv,'g-.',lw=.75,label='TPM model individual')
    ax.scatter(a_1_2[nonnans_1_2], Df_1_2/D0_N2, marker='^', c='gray',s=8,label='')
    ax.scatter(a_3_2[nonnans_3_2], Df_3_2/D0_N2, marker='^', c='gray',s=8,label='')
    ax.scatter(a_6_2[nonnans_6_2], Df_6_2/D0_N2, marker='^', c='gray',s=8,label='')
    ax.scatter(a_10_2[nonnans_10_2], Df_10_2/D0_N2, marker='^', c='gray',s=8,label='')
    
    ax.scatter(a_1_2[nonnans_1_2], Dc_1_2/D0_N2, marker='o', c='b',s=14,label='1 kPa')
    ax.scatter(a_3_2[nonnans_3_2], Dc_3_2/D0_N2, marker='o', c='orange',s=14,label='3 kPa')
    ax.scatter(a_6_2[nonnans_6_2], Dc_6_2/D0_N2, marker='o', c='g',s=14,label='6 kPa')
    ax.scatter(a_10_2[nonnans_10_2], Dc_10_2/D0_N2, marker='o', c='k',s=14,label='10 kPa')
    
    ax.legend()
    plt.title('TAMA 20-25')
    ax.set_xlim([0.05, 0.35])
    ax.set_ylim([-0.0005, 0.015])
    
    
    
    #--------------------------------------------------------------------------
    
    # 40-45 cm Df
    
    fig = plt.figure(num=111)
    
    fig.set_size_inches(7.,6.)
    plt.clf()
    ax = fig.add_subplot(1,1,1)
    
    D0_N2 = 1#0.2
    
    #Tämän pitäisi toimia näin, koska a:sta ja Df:stä puuttuvat samat arvot.
    ax.scatter(a_1_3[nonnans_1_3], Df_1_3/D0_N2, c='b',s=10,label='1 kPa')
    ax.scatter(a_3_3[nonnans_3_3], Df_3_3/D0_N2, c='orange',s=10,label='3 kPa')
    ax.scatter(a_6_3[nonnans_6_3], Df_6_3/D0_N2, c='g',s=10,label='6 kPa')
    ax.scatter(a_10_3[nonnans_10_3], Df_10_3/D0_N2, c='r',s=10,label='10 kPa')
    
    por = np.mean(fa_3)#0.9
    
    a_all_3 = np.concatenate([a_1_3[nonnans_10_3],a_3_3[nonnans_10_3],a_6_3[nonnans_10_3],a_10_3[nonnans_10_3]])
    args_3 = np.argsort(a_all_3)
    f_all_3_100 = np.concatenate([fa_3[nonnans_10_3],fa_3[nonnans_10_3],fa_3[nonnans_10_3],fa_3[nonnans_10_3]])
    f_all_3 = np.concatenate([fa_3[nonnans_10_3],fa_3[nonnans_10_3],fa_3[nonnans_10_3],fa_3[nonnans_10_3]])
    
    a_100s_3 = np.concatenate([a_10_3[nonnans_10_3],a_10_3[nonnans_10_3],a_10_3[nonnans_10_3],a_10_3[nonnans_10_3]])
    X_3_all = np.log10((2*a_100s_3**3+0.04*a_100s_3)/f_all_3_100**2) / np.log10(a_100s_3/f_all_3_100)
    
    a_s = np.linspace(0,np.max(a_all_3[~np.isnan(a_all_3)]),100)
    
    D0 = 0.202#1#0.2
    
    D_MQ = D0 * a_s**(10/3) / por**2
    D_MQ_3_indiv = D0 * a_all_3[args_3]**(10/3) / f_all_3[args_3]**2
      
    D_BBC_3 = D0 * a_s**(2+3/2.9) / por**2.9
    
    #D_BBC_3 = D0 * 0.66 * por * (a_s/por)**((12-3)/3)
    D_BBC_3 = D0 * a_s**(2.) / por**(2./3)
    
    D_BBC_3_indiv = D0 * a_all_3[args_3]**(2.)/ f_all_2[args_3]**(2./3)
    
    a_s_3 = np.copy(a_s)
    D_MQ_3 = np.copy(D_MQ)

    D_Cam_3 = D0 * 0.9*a_s**2.3
    
    a100_3_av = np.mean(a_10_3_all)
    fa_3_av = np.mean(fa_3)
    
    X_3 = np.log10((2*a100_3_av**3+0.04*a100_3_av)/fa_3_av**2) / np.log10(a100_3_av/fa_3_av)
    
    D_TPM_3 = d_tpm(D0,a_s,a100_3_av,fa_3_av)#    D0 * fa_1_av**2 * (a_s/fa_1_av)**(X_1)
    
    D_TPM_3_indiv = d_tpm(D0,a_all_3[args_3],a_100s_3[args_3],f_all_3[args_3])
    #D0 * f_all_1[args_1]**2 * (a_all_1[args_1]/f_all_1[args_1])**(X_1_all[args_1])
    
    ax.plot(a_s,D_MQ_3,'k-.',lw=.75,label='MQ model')
    ax.plot(a_s,D_TPM_3,'b-.',lw=.75,label='TPM model')
    ax.plot(a_s,D_Cam_3,'r-.',lw=.75,label='Campbell model')
    ax.plot(a_s,D_BBC_3,'r-.',lw=.75,label='BBC model')
    ax.plot(a_all_3[args_3],D_BBC_3_indiv,'g-.',lw=.75,label='BBC model individual')
    ax.plot(a_all_3[args_3],D_MQ_3_indiv,'g-.',lw=.75,label='MQ model individual')
    ax.plot(a_all_3[args_3],D_TPM_3_indiv,'g-.',lw=.75,label='TPM model individual')
    plt.title('40-45 cm Df')
    ax.legend()
    
    ax.set_xlim([0.001, 0.26])
    #ax.set_ylim([0.0, 0.009])

    # 40-45 cm Dc

    fig = plt.figure(num=1092)
    
    fig.set_size_inches(9.,8.)
    plt.clf()
    ax = fig.add_subplot(1,1,1)
    
    ax.scatter(a_1_3[nonnans_1_3], Dc_1_3/D0_N2, marker='s', c='b',s=10,label='1 kPa')
    ax.scatter(a_3_3[nonnans_3_3], Dc_3_3/D0_N2, marker='o', c='orange',s=10,label='3 kPa')
    ax.scatter(a_6_3[nonnans_6_3], Dc_6_3/D0_N2, marker='^', c='g',s=10,label='6 kPa')
    ax.scatter(a_10_3[nonnans_10_3], Dc_10_3/D0_N2, marker='D', c='k',s=10,label='10 kPa')
    
    ax.plot(a_s,D_MQ_3,'k-.',lw=.75,label='MQ model')
    ax.plot(a_s,D_TPM_3,'b-.',lw=.75,label='TPM model')
    ax.plot(a_s,D_Cam_3,'r-.',lw=.75,label='Campbell model')
    #ax.plot(a_s,D0*params[0][0] * a_s**params[0][1], 'g-', label='Own fit')
    #ax.plot(a_all_3[args_3],D_MQ_indiv,'g-.',lw=.75,label='MQ model individual')
    #ax.plot(a_all_3[args_3],D_TPM_indiv,'g-.',lw=.75,label='TPM model individual')
    
    ax.legend()
    plt.title('40-45 cm Dc')
    #ax.set_xlim([0.0, 0.12])
    #ax.set_ylim([-0.0005, 0.014])    
    
    
    #--------------------------------------------------------------------------
    
    #40-45 cm

    
    fig = plt.figure(num=4090)
    
    fig.set_size_inches(9.,9.)
    plt.clf()
    ax = fig.add_subplot(1,1,1)
    
    ax.plot(a_s,D_MQ_3, c='gray', ls='--',lw=.75,label='MQ model')
    ax.plot(a_s,D_TPM_3,c='gray', ls=':',lw=.75,label='TPM model')
    ax.plot(a_s,D_Cam_3,c='gray', ls='-',lw=.75,label='Campbell model')
    ax.plot(a_s,D_BBC_3, c='gray',ls='-.',lw=.75,label='BBC model')
    ax.scatter(a_all_3[args_3],D_MQ_3_indiv,marker='^', c='gray',s=8)
    ax.scatter(a_all_3[args_3],D_BBC_3_indiv,marker='^', c='gray',s=8)
    #ax.plot(a_all_1[args_1],D_TPM_indiv,'g-.',lw=.75,label='TPM model individual')
    ax.scatter(a_1_3[nonnans_1_3], Df_1_3/D0_N2, marker='^', c='gray',s=8)
    ax.scatter(a_3_3[nonnans_3_3], Df_3_3/D0_N2, marker='^', c='gray',s=8)
    ax.scatter(a_6_3[nonnans_6_3], Df_6_3/D0_N2, marker='^', c='gray',s=8)
    ax.scatter(a_10_3[nonnans_10_3], Df_10_3/D0_N2, marker='^', c='gray',s=8)
    
    ax.scatter(a_1_3[nonnans_1_3], Dc_1_3/D0_N2, marker='o', c='b',s=14,label='1 kPa')
    ax.scatter(a_3_3[nonnans_3_3], Dc_3_3/D0_N2, marker='o', c='orange',s=14,label='3 kPa')
    ax.scatter(a_6_3[nonnans_6_3], Dc_6_3/D0_N2, marker='o', c='g',s=14,label='6 kPa')
    ax.scatter(a_10_3[nonnans_10_3], Dc_10_3/D0_N2, marker='o', c='k',s=14,label='10 kPa')
    
    ax.legend()
    plt.title('TAMA 40-45')
    ax.set_xlim([0.05, 0.35])
    ax.set_ylim([0.0, 0.015])
    
    
    #''''''''''''''''''''''''''''''''''
    #''''''''''''''''''''''''''''''''''
    # FINAL IMAGES
    #''''''''''''''''''''''''''''''''''
    #''''''''''''''''''''''''''''''''''
    
    #colors = list([(0,0,0), (230/255,159/255,0), (86/255,180/255,233/255),
    #           (0,158/255,115/255), (240/255,228/255,66/255),
    #           (0,84/255,238/255), (213/255,94/255,0),(204/255,121/255,167/255)])
    
    #colors = list([(0,0,0), (0/255,0/255,185/255), (0/255,0/255,0/255),
    #           (135/255,206/255,255/255), (0/255,0/255,0/255),
    #           (100/255,149/255,237/255), (0/255,0/255,0),(0/255,0/255,0/255)])
    
    #https://github.com/gka/chroma.js/issues/200
    colors = list(['#14181C', '#08569A', '',
               '#80BFF7', '',
               '#007FEF'])
    
    
    bbdim = [0.22, 0.80]
    
    ds = ['0\u20135 cm', '20\u201325 cm', '40\u201345 cm']
    
    bppos = [
            [0.07, 0.15, bbdim[0],bbdim[1]],
            [0.35, 0.15, bbdim[0],bbdim[1]],
            [0.63, 0.15, bbdim[0],bbdim[1]],
            ]
    
    subt = [('(a)'), ('(b)'), ('(c)')]
    
    subtpos = [0.06, 0.92]
    
    size_small = 4
    
    D0_N2 = 0.202
    
    fig = plt.figure(num=5091)
    fig.set_size_inches(6,2.0)
    plt.clf()
        
    
    ax1 = fig.add_subplot(1,3,1)
    
    a_s = np.linspace(0,np.max(a_all_1[~np.isnan(a_all_1)]),100)
    
    ax1.plot(a_s,D_MQ_1/D0_N2, c='gray', ls='--',lw=.75,label='MQ')
    ax1.plot(a_s,D_TPM_1/D0_N2,c='gray', ls=':',lw=.75,label='TPM')
    ax1.plot(a_s,D_Cam/D0_N2,c='gray', ls='-',lw=.75,label='Campbell')
    ax1.plot(a_s,D_BBC_1/D0_N2, c='gray',ls='-.',lw=.75,label='BBC')
    ax1.scatter(a_all_1[args_1],D_MQ_1_indiv/D0_N2,marker='^', c='gray',s=size_small)
    ax1.scatter(a_all_1[args_1],D_BBC_1_indiv/D0_N2,marker='^', c='gray',s=size_small)
    ax1.scatter(a_1_1[nonnans_1_1], Df_1_1/D0_N2, marker='^', c='gray',s=size_small)
    ax1.scatter(a_3_1[nonnans_3_1], Df_3_1/D0_N2, marker='^', c='gray',s=size_small)
    ax1.scatter(a_6_1[nonnans_6_1], Df_6_1/D0_N2, marker='^', c='gray',s=size_small)
    ax1.scatter(a_10_1[nonnans_10_1], Df_10_1/D0_N2, marker='^', c='gray',s=size_small)
    
    ax1.scatter(a_1_1[nonnans_1_1], Dc_1_1/D0_N2, marker='o', color=colors[0],s=14,label='1 kPa')
    ax1.scatter(a_3_1[nonnans_3_1], Dc_3_1/D0_N2, marker='o', color=colors[1],s=14,label='3 kPa')
    ax1.scatter(a_6_1[nonnans_6_1], Dc_6_1/D0_N2, marker='o', color=colors[5],s=14,label='6 kPa')
    ax1.scatter(a_10_1[nonnans_10_1], Dc_10_1/D0_N2, marker='o', color=colors[3],s=14,label='10 kPa')
    
    ax1.set_xlabel('Air-filled porosity (m$^3$ m$^{-3}$)')
    ax1.set_ylabel('$D_{\mathrm{s}}/D_0$')
    ax1.yaxis.set_label_coords(-0.18, 0.5, transform=ax1.transAxes)
    ax1.xaxis.set_label_coords(0.5, -0.09, transform=ax1.transAxes)
    
    ax1.set_xlim([0.14, 0.67])
    #ax1.set_ylim([0.0/D0_N2, 0.075/D0_N2])
    ax1.set_ylim([-0.01, 0.67])
    
    ax1.text(subtpos[0], subtpos[1], subt[0], transform=ax1.transAxes)
    
    ax1.set_position(bppos[0])
    #ax1.set_position([0.08, 0.10, bbdim[0],0.6])
    '''
    ax1b = fig.add_subplot(2,3,4)#, sharex=ax1)
    
    ax1b.scatter(a_1_1[nonnans_1_1], Df_1_1/D0_N2, marker='^', c='gray',s=size_small)
    
    ax1b.set_xlim([0.14, 0.65])
    ax1b.set_ylim([0.50, 0.64])
    
    ax1.spines["top"].set_visible(False)
    ax1b.spines["bottom"].set_visible(False)
    #ax1b.xaxis.tick_bottom()
    #ax1b.tick_params(labeltop=False)  # don't put tick labels at the top
    ax1b.xaxis.tick_top()
    ax1b.tick_params(labeltop=False)  # don't put tick labels at the top
    ax1b.set_xticks([])
    
    ax1b.set_position([0.08, 0.75, bbdim[0],0.2])
    '''
    ax2 = fig.add_subplot(1,3,2)
      
    
    a_s = np.linspace(0,np.max(a_all_2[~np.isnan(a_all_2)]),100)
    
    ax2.plot(a_s,D_MQ_2/D0_N2, c='gray', ls='--',lw=.75,label='MQ61')
    ax2.plot(a_s,D_TPM_2/D0_N2,c='gray', ls=':',lw=.75,label='TPM')
    ax2.plot(a_s,D_Cam_2/D0_N2,c='gray', ls='-',lw=.75,label='CC')
    ax2.plot(a_s,D_BBC_2/D0_N2, c='gray',ls='-.',lw=.75,label='MQ60')
    ax2.scatter(a_all_2[args_2],D_MQ_2_indiv/D0_N2,marker='^', c='gray',s=size_small)
    ax2.scatter(a_all_2[args_2],D_BBC_2_indiv/D0_N2,marker='^', c='gray',s=size_small)
    ax2.scatter(a_1_2[nonnans_1_2], Df_1_2/D0_N2, marker='^', c='gray',s=size_small,label='')
    ax2.scatter(a_3_2[nonnans_3_2], Df_3_2/D0_N2, marker='^', c='gray',s=size_small,label='')
    ax2.scatter(a_6_2[nonnans_6_2], Df_6_2/D0_N2, marker='^', c='gray',s=size_small,label='')
    ax2.scatter(a_10_2[nonnans_10_2], Df_10_2/D0_N2, marker='^', c='gray',s=size_small,label='')
    
    ax2.scatter(a_1_2[nonnans_1_2], Dc_1_2/D0_N2, marker='o', color=colors[0],s=14,label='1 kPa')
    ax2.scatter(a_3_2[nonnans_3_2], Dc_3_2/D0_N2, marker='o', color=colors[1],s=14,label='3 kPa')
    ax2.scatter(a_6_2[nonnans_6_2], Dc_6_2/D0_N2, marker='o', color=colors[5],s=14,label='6 kPa')
    ax2.scatter(a_10_2[nonnans_10_2], Dc_10_2/D0_N2, marker='o', color=colors[3],s=14,label='10 kPa')
    
    #ax2.set_xlim([0.05, 0.35])
    #ax2.set_ylim([-0.0005/D0_N2, 0.015/D0_N2])
    

    
    ax2.set_xlabel('Air-filled porosity (m$^3$ m$^{-3}$)')
    ax2.xaxis.set_label_coords(0.5, -0.09, transform=ax2.transAxes)
    
    ax2.set_xlim([0.0, 0.38])
    #ax2.set_ylim([-0.0005/D0_N2, 0.019/D0_N2])
    ax2.set_ylim([-0.0015, 0.095])
    
    ax2.text(subtpos[0], subtpos[1], subt[1], transform=ax2.transAxes)
    
    ax2.set_position(bppos[1])
    
    ax3 = fig.add_subplot(1,3,3)
    
    a_s = np.linspace(0,np.max(a_all_3[~np.isnan(a_all_3)]),100)
    
    ax3.plot(a_s,D_MQ_3/D0_N2, c='gray', ls='--',lw=.75,label='MQ61')
    ax3.plot(a_s,D_TPM_3/D0_N2,c='gray', ls=':',lw=.75,label='TPM')
    ax3.plot(a_s,D_Cam_3/D0_N2,c='gray', ls='-',lw=.75,label='CC')
    ax3.plot(a_s,D_BBC_3/D0_N2, c='gray',ls='-.',lw=.75,label='MQ60')
    ax3.scatter(a_all_3[args_3],D_MQ_3_indiv/D0_N2,marker='^', c='gray',s=size_small)
    ax3.scatter(a_all_3[args_3],D_BBC_3_indiv/D0_N2,marker='^', c='gray',s=size_small)
    
    ax3.scatter(a_1_3[nonnans_1_3], Df_1_3/D0_N2, marker='^', c='gray',s=size_small,label='')
    ax3.scatter(a_3_3[nonnans_3_3], Df_3_3/D0_N2, marker='^', c='gray',s=size_small,label='')
    ax3.scatter(a_6_3[nonnans_6_3], Df_6_3/D0_N2, marker='^', c='gray',s=size_small,label='')
    ax3.scatter(a_10_3[nonnans_10_3], Df_10_3/D0_N2, marker='^', c='gray',s=size_small,label='')
    
    ax3.scatter(a_1_3[nonnans_1_3], Dc_1_3/D0_N2, marker='o', color=colors[0],s=14,label='-1 kPa')
    ax3.scatter(a_3_3[nonnans_3_3], Dc_3_3/D0_N2, marker='o', color=colors[1],s=14,label='-3 kPa')
    ax3.scatter(a_6_3[nonnans_6_3], Dc_6_3/D0_N2, marker='o', color=colors[5],s=14,label='-6 kPa')
    ax3.scatter(a_10_3[nonnans_10_3], Dc_10_3/D0_N2, marker='o', color=colors[3],s=14,label='-10 kPa')

    handles, labels = ax2.get_legend_handles_labels()
    order = [3,2,1,0]
    ax2.legend([handles[idx] for idx in order],[labels[idx] for idx in order],
               ncol = 1, fontsize=fontsize,  bbox_to_anchor=(0.97, 0.92),
               bbox_transform=plt.gcf().transFigure,
               columnspacing=1, handletextpad=0.45)
   
    handles, labels = ax3.get_legend_handles_labels()
    order = [4,5,6,7]
    ax3.legend([handles[idx] for idx in order],[labels[idx] for idx in order],
               ncol = 1, fontsize=fontsize,  bbox_to_anchor=(0.86, 0.50),
               bbox_transform=plt.gcf().transFigure,
               columnspacing=1, handletextpad=0.35)
    
    
    ax3.set_xlabel('Air-filled porosity (m$^3$ m$^{-3}$)')
    ax3.yaxis.set_label_coords(-0.19, 0.5, transform=ax3.transAxes)
    ax3.xaxis.set_label_coords(0.5, -0.09, transform=ax3.transAxes)
    
    ax3.set_xlim([0.0, 0.27])
    #ax3.set_ylim([-0.0005/D0_N2, 0.01625/D0_N2])
    ax3.set_ylim([-0.0015, 0.095])
    
    ax3.text(subtpos[0], subtpos[1], subt[2], transform=ax3.transAxes)
    
    ax3.text(1.22, 0.95, 'Models',transform=ax3.transAxes, fontsize=fontsize,ha='center')
    ax3.text(1.36, 0.43, 'Measurements',transform=ax3.transAxes, fontsize=fontsize,ha='center')
    
    ax3.set_position(bppos[2])
    
    plt.savefig('fig03.pdf')
    
    #--------------------------------------------------------------------------
    
    colors = list([(0,0,0), (230/255,159/255,0), (86/255,180/255,233/255),
               (0,158/255,115/255), (240/255,228/255,66/255),
               (0,84/255,238/255), (213/255,94/255,0),(204/255,121/255,167/255)])
    
    fig = plt.figure(num=5092)
    fig.set_size_inches(3.0,3.0)
    plt.clf()
        
    ax = fig.add_subplot(1,1,1)
    
    a_s = np.linspace(0,0.55,100)
    
    ax.plot(a_s, 2*a_s**3+0.04*a_s, c='k', ls='-',lw=.75,label='')
    ax.scatter(a_10_1[nonnans_10_1], Dc_10_1/D0_N2, marker='o', color=colors[0],s=14,label='0-5 cm')
    ax.scatter(a_10_2[nonnans_10_2], Dc_10_2/D0_N2, marker='o', color=colors[2],s=14,label='20-25 cm')
    ax.scatter(a_10_3[nonnans_10_3], Dc_10_3/D0_N2, marker='o', color=colors[3],s=14,label='40-45 cm')
    
    ax.legend(bbox_to_anchor=(0.43, 0.97),
               bbox_transform=ax.transAxes,
               handletextpad=0.45)
    
    ax.set_xlim([-0.002,0.59])
    ax.set_ylim([-0.001,0.39])
    
    ax.set_xlabel('Air-filled porosity at -10 kPa (m$^3$ m$^{-3}$)')
    ax.set_ylabel('$D_{100}/D_{0}$')
    ax.xaxis.set_label_coords(0.5, -0.07)
    ax.yaxis.set_label_coords(-0.12, 0.5)
    
    ax.annotate("", xy=(0.35, 0.12), xytext=(0.22, 0.21),
            arrowprops=dict(arrowstyle="->"))
    
    ax.text(0.14, 0.56, r'$D_{100}/D_0=2a_{100}^{3}+0.04a_{100}$',
            transform=ax.transAxes, fontsize=fontsize)
    #ax.text(0.21, 0.45, '$=2a_{100}^{3}+0.04a_{100}$',
    #        transform=ax.transAxes, fontsize=fontsize)
    
    ax.set_position([0.16, 0.14, 0.80, 0.80])
    
    #plt.savefig('fig04.pdf')
    
    #--------------------------------------------------------
    
    
    # all in the same figure
    if False:
    
        fig = plt.figure(num=1093)
        
        fig.set_size_inches(7.,6.)
        plt.clf()
        ax = fig.add_subplot(1,1,1)
        
        ax.scatter(a_1_3[nonnans_1_3], Dc_1_3/D0_N2, marker='s', c='b',s=10,label='1 kPa')
        ax.scatter(a_3_3[nonnans_3_3], Dc_3_3/D0_N2, marker='o', c='orange',s=10,label='3 kPa')
        ax.scatter(a_6_3[nonnans_6_3], Dc_6_3/D0_N2, marker='^', c='g',s=10,label='6 kPa')
        ax.scatter(a_10_3[nonnans_10_3], Dc_10_3/D0_N2, marker='D', c='k',s=10,label='10 kPa')
        
        ax.scatter(a_1_2[nonnans_1_2], Dc_1_2/D0_N2, marker='s', c='b',s=10,label='')
        ax.scatter(a_3_2[nonnans_3_2], Dc_3_2/D0_N2, marker='o', c='orange',s=10,label='')
        ax.scatter(a_6_2[nonnans_6_2], Dc_6_2/D0_N2, marker='^', c='g',s=10,label='')
        ax.scatter(a_10_2[nonnans_10_2], Dc_10_2/D0_N2, marker='D', c='k',s=10,label='')
        
        ax.scatter(a_1_1[nonnans_1_1], Dc_1_1/D0_N2, marker='s', c='b',s=10,label='')
        ax.scatter(a_3_1[nonnans_3_1], Dc_3_1/D0_N2, marker='o', c='orange',s=10,label='')
        ax.scatter(a_6_1[nonnans_6_1], Dc_6_1/D0_N2, marker='^', c='g',s=10,label='')
        ax.scatter(a_10_1[nonnans_10_1], Dc_10_1/D0_N2, marker='D', c='k',s=10,label='')
        
        a_s = np.linspace(0,0.80,100)
        
        D_Cam = D0 * 0.9*a_s**2.3
        
        
        #ax.plot(a_s,D_MQ,'k-.',lw=.75,label='MQ model')
        #ax.plot(a_s,D_TPM,'b-.',lw=.75,label='TPM model')
        ax.plot(a_s,D_Cam/D0_N2,'r-.',lw=.75,label='Campbell model')
        ax.plot(a_s_1,D_MQ_1/D0_N2,'c-.',lw=.75,label='MQ model 0-5')
        ax.plot(a_s_2,D_MQ_2/D0_N2,'m-.',lw=.75,label='MQ model 20-25')
        ax.plot(a_s_3,D_MQ_3/D0_N2,'g-.',lw=.75,label='MQ model 40-45')
        
        ax.legend()
        plt.title('All depths Dc')
        ax.set_xlim([0.0, 0.65])
        ax.set_ylim([-0.0005, 0.14])   


#------------------------------------------------------------------

if ttests:


    # Dc -1 kPa 20-25 vs. 40-45 cm
    
    print('Independent samples t-test for Dc at -1 kPa between middle and bottom layers')
    try:
        ttest_1 = hp.hypothesis.tTest(Dc_1_2, Dc_1_3, paired=False, var_equal=True)
        print(ttest_1.test_summary)
    except Exception:
        print(stats.ttest_ind(Dc_1_2, Dc_1_3))
    
    print('Bartlett')
    print(stats.bartlett(Dc_1_2, Dc_1_3))
    print([np.var(x, ddof=1) for x in [Dc_1_2, Dc_1_3]])
    print('Levene')
    print(stats.levene(Dc_1_2, Dc_1_3))
    print('Shapiro middle and bottom')
    print(stats.shapiro(Dc_1_2))
    print(stats.shapiro(Dc_1_3))

    print('----------')
    
    # Dc -1 kPa 20-25 vs. 40-45 cm
    
    print('Independent samples t-test for a at -1 kPa between middle and bottom layers')
    try:
        ttest_1 = hp.hypothesis.tTest(a_1_2, a_1_3[~np.isnan(a_1_3)], paired=False, var_equal=True)
        print(ttest_1.test_summary)
    except Exception:
        print(stats.ttest_ind(a_1_2, a_1_3[~np.isnan(a_1_3)]))
    
    print('Bartlett')
    print(stats.bartlett(a_1_2, a_1_3[~np.isnan(a_1_3)]))
    print([np.var(x, ddof=1) for x in [a_1_2, a_1_3[~np.isnan(a_1_3)]]])
    print('Levene')
    print(stats.levene(a_1_2, a_1_3[~np.isnan(a_1_3)]))
    print('Shapiro middle and bottom')
    print(stats.shapiro(a_1_2))
    print(stats.shapiro(a_1_3[~np.isnan(a_1_3)]))