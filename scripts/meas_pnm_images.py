# -*- coding: utf-8 -*-
"""
Created on Mon Nov 22 16:45:06 2021

@author: pkiuru
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, FormatStrFormatter, MaxNLocator
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

#import my_models as mm

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


    
pnmdata = np.loadtxt('../Data/D_PNM_comparison_3kPa.txt', skiprows=1, usecols= (0,1))
#pnmdata = np.loadtxt('../Data/a_PNM_comparison_3kPa.txt', skiprows=1, usecols= (0,1))

pndata = np.loadtxt('../Data/a_PNM_comparison_3kPa.txt', skiprows=1, usecols= (0,1))

Dc = pnmdata[:,0]
Dpnm = pnmdata[:,1]

#Dc = np.log10(Dc)
#Dpnm = np.log10(Dpnm)

ameas = pndata[:,0]
apnm = pndata[:,1]
    
def probability_plot(fignum, data_1, data_2, data_3):
    
    fig = plt.figure(num=fignum)
    fig.set_size_inches(10,4)
    plt.clf()
    ax1 = fig.add_subplot(131)
    stats.probplot(data_1, plot= plt, rvalue= True)
    ax1.set_title(str(fignum))
    ax2 = fig.add_subplot(132)
    stats.probplot(data_2, plot= plt, rvalue= True)
    ax2.set_title(str(fignum))
    ax2.set_ylabel('')
    ax3 = fig.add_subplot(133)
    stats.probplot(data_3, plot= plt, rvalue= True)
    ax3.set_title(str(fignum))
    ax3.set_ylabel('')
    plt.tight_layout()



print('Measured vs. PNM ' + str(stats.shapiro(Dc-Dpnm)))
print(stats.anderson(Dc-Dpnm))
   
probability_plot(1, Dc-Dpnm, Dc-Dpnm, Dc-Dpnm)

print('')
print('Paired t-test Measured vs. PNM')
print(stats.ttest_rel(Dc, Dpnm))
try:
    print('HYPOTHETICAL')
    gt_paired_ttest_1 = hp.hypothesis.tTest(Dc, Dpnm, paired=True, var_equal=True)
    print(gt_paired_ttest_1.test_summary)
except Exception:
    pass
print('')

print('')
print('Independent t-test Measured vs. PNM')
print(stats.ttest_ind(Dc, Dpnm,equal_var=True))
print('')
print('')
print('Independent t-test Measured vs. PNM, Welch')
print(stats.ttest_ind(Dc, Dpnm,equal_var=False))
print('')

print('')
print('Wilcoxon signed rank test Measured vs. PNM')
print(stats.wilcoxon(Dc, Dpnm))
#print('Hypothetical')
#w = hp.nonparametric.WilcoxonTest(Dc, Dpnm, paired=True)
#print(w.test_summary)
print('')

print('')
print('Sign test Measured vs. PNM')
print(stats.binom_test(np.sum(Dc-Dpnm > 0), len(Dc), p=0.5, alternative='two-sided'))
print('')

print('')
print('Variance equivalence: Measured vs. PNM')
print(stats.bartlett(Dc, Dpnm))
print(stats.levene(Dc, Dpnm))
print([np.var(x, ddof=1) for x in [Dc, Dpnm]])


# D meas vs. D pnm - 4 subplots

slope, intercept, r_value, p_value, std_err = stats.linregress(Dc, Dpnm)

fig = plt.figure(num=888)
fig.set_size_inches(6.2,5.6)
plt.clf()
ax = fig.add_subplot(2,2,1)
ax.scatter(Dc[0:4], Dpnm[0:4], marker='o', s=14, c='k', label='0\u20135 cm',zorder=3)
ax.scatter(Dc[4:8], Dpnm[4:8], marker='s', s=14, c='b', label='20\u201325 cm',zorder=4)
ax.scatter(Dc[8:], Dpnm[8:], marker='^', s=16, c='r', label='40\u201345 cm',zorder=5)
ax.plot(np.linspace(np.min(Dc),np.max(Dc),100), np.linspace(np.min(Dc),
                    np.max(Dc),100), 'k:', lw=1, label='1:1 line',zorder=2)
ax.plot(Dc, intercept + slope * Dc, 'k', lw=1, label = 'Linear fit',zorder=1)

formatter = FormatStrFormatter('%1.3f')
locator = MaxNLocator(nbins=8, steps=[4])
ax.yaxis.set_major_locator(locator)
ax.yaxis.set_major_formatter(formatter)
ax.xaxis.set_major_locator(locator)
ax.xaxis.set_major_formatter(formatter)

ax.set_xlabel(r'D$_{\mathrm{s}}$ (cm$^2$ s$^{-1}$)')
ax.set_ylabel(r'D$_{\mathrm{pnm}}$ (cm$^2$ s$^{-1}$)')


ax.text(0.48 ,0.87, '$y = $' + np.format_float_positional(slope, 4) + '$x + $'
        + np.format_float_positional(intercept, 4), transform=ax.transAxes)
ax.text(0.48, 0.81, '$R^2 = $' + np.format_float_positional(r_value**2, 2),
        transform=ax.transAxes)

lims = [-0.0005,0.022]
ax.set_xlim(lims)
ax.set_ylim(lims)


handles, labels = ax.get_legend_handles_labels()
order = [2,3,4,1,0]
ax.legend([handles[idx] for idx in order],[labels[idx] for idx in order],
           fontsize=fontsize, loc='lower right', borderaxespad=1.00)

ax.yaxis.set_label_coords(-0.12, 0.5)
ax.xaxis.set_label_coords(0.5, -0.065)

ax.set_position([0.08, 0.54, 0.40, 0.42])

# Bland-Altmann for D

count = len(Dc)

means = np.mean([Dc, Dpnm], axis=0)
diff = Dc-Dpnm
mean_diff = np.mean(diff)
std_diff = np.std(diff, axis=0)
sd_limit = 1.96

sterr = std_diff / np.sqrt(count)
t_crit = stats.t.ppf(q=1-0.025,df=count-1)
ci = sterr * t_crit

sterr_limit = np.sqrt(1/count+sd_limit**2/(2*(count-1)))*std_diff
ci_limit = sterr_limit * t_crit

#fig = plt.figure(num=89)
#fig.set_size_inches(3.2,3.2)
#plt.clf()
ax2 = fig.add_subplot(2,2,2)
ax2.scatter(np.mean([Dc[0:4], Dpnm[0:4]], axis=0), Dc[0:4]-Dpnm[0:4], marker='o', s=14, c='k')
ax2.scatter(np.mean([Dc[4:8], Dpnm[4:8]], axis=0), Dc[4:8]-Dpnm[4:8], marker='s', s=14, c='b')
ax2.scatter(np.mean([Dc[8:], Dpnm[8:]], axis=0), Dc[8:]-Dpnm[8:], marker='^', s=16, c='r')
ax2.axhline(mean_diff,ls='-',c='k',lw=1.)
ax2.annotate('mean diff:\n{}'.format(np.round(mean_diff, 3)),
            xy=(0.98, 0.60),
            horizontalalignment='right',
            verticalalignment='center',
            fontsize=fontsize+0,
            xycoords='axes fraction')

half_ylim = (1.75 * sd_limit) * std_diff
ax2.set_ylim(mean_diff - half_ylim,
            mean_diff + half_ylim)
#ax2.yaxis.tick_right()

xmin = 0
xmax = 1.15*np.max(means)
ax2.set_xlim(xmin, xmax)

limit_of_agreement = sd_limit * std_diff
lower = mean_diff - limit_of_agreement
upper = mean_diff + limit_of_agreement
for j, lim in enumerate([lower, upper]):
    ax2.axhline(lim,ls='--',c='k',lw=0.5)
ax2.annotate('-{}sd: {}'.format(sd_limit, np.round(lower, 3)),
            xy=(0.04, 0.07),
            horizontalalignment='left',
            verticalalignment='bottom',
            fontsize=fontsize+0,
            xycoords='axes fraction')
ax2.annotate('+{}sd: {}'.format(sd_limit, np.round(upper, 3)),
            xy=(0.04, 0.91),
            horizontalalignment='left',
            fontsize=fontsize+0,
            xycoords='axes fraction')

x = np.linspace(xmin, xmax)
ax2.fill_between(x, mean_diff-ci, mean_diff+ci, color='silver', alpha=0.25)
ax2.fill_between(x, mean_diff+limit_of_agreement-ci_limit, mean_diff+limit_of_agreement+ci_limit, color='silver', alpha=0.1)
ax2.fill_between(x, mean_diff-limit_of_agreement-ci_limit, mean_diff-limit_of_agreement+ci_limit, color='silver', alpha=0.1)

ax2.yaxis.set_label_coords(-0.14, 0.5)
ax2.xaxis.set_label_coords(0.5, -0.065)

ax2.set_xlabel(r'mean(D$_{\mathrm{s}}$,D$_{\mathrm{pnm}}$) (cm$^2$ s$^{-1}$)')
ax2.set_ylabel(r'D$_{\mathrm{s}}$-D$_{\mathrm{pnm}}$ (cm$^2$ s$^{-1}$)')

ax2.set_position([0.58, 0.54, 0.40, 0.42])

# a meas vs a pnm

slope_a, intercept_a, r_value_a, p_value_a, std_err_a = stats.linregress(ameas, apnm)

ax3 = fig.add_subplot(2,2,3)
ax3.scatter(ameas[0:4], apnm[0:4], marker='o', s=14, c='k', label='0\u20135 cm',zorder=3)
ax3.scatter(ameas[4:8], apnm[4:8], marker='s', s=14, c='b', label='20\u201325 cm',zorder=4)
ax3.scatter(ameas[8:], apnm[8:], marker='^', s=16, c='r', label='40\u201345 cm',zorder=5)
ax3.plot(np.linspace(np.min(ameas),np.max(ameas),100), np.linspace(np.min(ameas),
                    np.max(ameas),100), 'k:', lw=1, label='1:1 line',zorder=2)
ax3.plot(ameas, intercept_a + slope_a * ameas, 'k', lw=1, label = 'Linear fit',zorder=1)

formatter = FormatStrFormatter('%1.1f')
locator = MaxNLocator(nbins=8, steps=[4])
ax3.yaxis.set_major_locator(locator)
ax3.yaxis.set_major_formatter(formatter)
ax3.xaxis.set_major_locator(locator)
ax3.xaxis.set_major_formatter(formatter)

ax3.set_xlabel(r'a$_{\mathrm{meas}}$ (m$^3$ m$^{-3}$)')
ax3.set_ylabel(r'a$_{\mathrm{pnm}}$ (m$^3$ m$^{-3}$)')


ax3.text(0.48 ,0.27, '$y = $' + np.format_float_positional(slope_a, 3) + '$x + $'
        + np.format_float_positional(intercept_a, 3), transform=ax3.transAxes)
ax3.text(0.48, 0.21, '$R^2 = $' + np.format_float_positional(r_value_a**2, 2),
        transform=ax3.transAxes)

lims = [-0.02,0.61]
ax3.set_xlim(lims)
ax3.set_ylim(lims)


#handles, labels = ax3.get_legend_handles_labels()
#order = [2,3,4,1,0]
#ax3.legend([handles[idx] for idx in order],[labels[idx] for idx in order],
#           fontsize=fontsize, loc='upper right', borderaxespad=1.00)

ax3.yaxis.set_label_coords(-0.08, 0.5)
ax3.xaxis.set_label_coords(0.5, -0.065)

ax3.set_position([0.08, 0.06, 0.40, 0.42])


# Bland-Altmann for a

#ameas = np.log10(ameas)
#apnm = np.log10(apnm)

count = len(ameas)

means = np.mean([ameas, apnm], axis=0)
diff = ameas-apnm
mean_diff = np.mean(diff)
std_diff = np.std(diff, axis=0)
sd_limit = 1.96

sterr = std_diff / np.sqrt(count)
t_crit = stats.t.ppf(q=1-0.025,df=count-1)
ci = sterr * t_crit

sterr_limit = np.sqrt(1/count+sd_limit**2/(2*(count-1)))*std_diff
ci_limit = sterr_limit * t_crit

ax4 = fig.add_subplot(2,2,4)
ax4.scatter(np.mean([ameas[0:4], apnm[0:4]], axis=0), ameas[0:4]-apnm[0:4], marker='o', s=14, c='k')
ax4.scatter(np.mean([ameas[4:8], apnm[4:8]], axis=0), ameas[4:8]-apnm[4:8], marker='s', s=14, c='b')
ax4.scatter(np.mean([ameas[8:], apnm[8:]], axis=0), ameas[8:]-apnm[8:], marker='^', s=16, c='r')
ax4.axhline(mean_diff,ls='-',c='k',lw=1.)
ax4.annotate('mean diff:\n{}'.format(np.round(mean_diff, 2)),
            xy=(0.98, 0.60),
            horizontalalignment='right',
            verticalalignment='center',
            fontsize=fontsize+0,
            xycoords='axes fraction')

half_ylim = (1.7 * sd_limit) * std_diff
ax4.set_ylim(mean_diff - half_ylim,
            mean_diff + half_ylim)
#ax2.yaxis.tick_right()



limit_of_agreement = sd_limit * std_diff
lower = mean_diff - limit_of_agreement
upper = mean_diff + limit_of_agreement
for j, lim in enumerate([lower, upper]):
    ax4.axhline(lim,ls='--',c='k',lw=0.5)
ax4.annotate('-{}sd: {}'.format(sd_limit, np.round(lower, 2)),
            xy=(0.04, 0.07),
            horizontalalignment='left',
            verticalalignment='bottom',
            fontsize=fontsize+0,
            xycoords='axes fraction')
ax4.annotate('+{}sd: {}'.format(sd_limit, np.round(upper, 2)),
            xy=(0.04, 0.91),
            horizontalalignment='left',
            fontsize=fontsize+0,
            xycoords='axes fraction')


xmax = 0.
xmin = 1.15*np.min(means)
ax4.set_xlim(xmin, xmax)

x = np.linspace(xmin, xmax)
ax4.fill_between(x, mean_diff-ci, mean_diff+ci, color='silver', alpha=0.25)
ax4.fill_between(x, mean_diff+limit_of_agreement-ci_limit, mean_diff+limit_of_agreement+ci_limit, color='silver', alpha=0.1)
ax4.fill_between(x, mean_diff-limit_of_agreement-ci_limit, mean_diff-limit_of_agreement+ci_limit, color='silver', alpha=0.1)

ax4.yaxis.set_label_coords(-0.09, 0.5)
ax4.xaxis.set_label_coords(0.5, -0.065)

ax4.set_xlabel(r'mean$(\log_{10}(a_{\mathrm{meas}})$,$\log_{10}(a_{\mathrm{pnm}}))$')
ax4.set_ylabel(r'$\log_{10}(a_{\mathrm{meas}})-\log_{10}(a_{\mathrm{pnm}})$')

ax4.set_position([0.58, 0.06, 0.40, 0.42])

#------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------
# D meas vs. D pnm - 3 subplots

slope, intercept, r_value, p_value, std_err = stats.linregress(Dc, Dpnm)

fig = plt.figure(num=88)
fig.set_size_inches(6.0,2.0)
plt.clf()
ax = fig.add_subplot(1,3,1)
ax.scatter(Dc[0:4], Dpnm[0:4], marker='o', s=14, c='k', label='Top',zorder=3)
ax.scatter(Dc[4:8], Dpnm[4:8], marker='s', s=14, c='b', label='Middle',zorder=4)
ax.scatter(Dc[8:], Dpnm[8:], marker='^', s=16, c='r', label='Bottom',zorder=5)
ax.plot(np.linspace(np.min(Dc),np.max(Dc),100), np.linspace(np.min(Dc),
                    np.max(Dc),100), 'k:', lw=1, label='1:1 line',zorder=2)
ax.plot(Dc, intercept + slope * Dc, 'k', lw=1, label = 'Linear fit',zorder=1)

formatter = FormatStrFormatter('%1.2f')
locator = MaxNLocator(nbins=8, steps=[10])
ax.yaxis.set_major_locator(locator)
ax.yaxis.set_major_formatter(formatter)
ax.xaxis.set_major_locator(locator)
ax.xaxis.set_major_formatter(formatter)

ax.set_xlabel(r'$D_{\mathrm{s}}$ (cm$^2$ s$^{-1}$)')
ax.set_ylabel(r'$D_{\mathrm{pnm}}$ (cm$^2$ s$^{-1}$)')


ax.text(0.24 ,0.87, '$y = $' + np.format_float_positional(slope, 4) + '$x + $'
        + np.format_float_positional(intercept, 4), transform=ax.transAxes)
ax.text(0.24, 0.77, '$R^2 = $' + np.format_float_positional(r_value**2, 2),
        transform=ax.transAxes)

lims = [-0.0005,0.032]
ax.set_xlim(lims)
ax.set_ylim(lims)


handles, labels = ax.get_legend_handles_labels()
order = [2,3,4,1,0]
ax.legend([handles[idx] for idx in order],[labels[idx] for idx in order],
           fontsize=fontsize, loc='lower right', borderaxespad=0.50)

ax.yaxis.set_label_coords(-0.17, 0.5)
ax.xaxis.set_label_coords(0.5, -0.095)

ax.set_position([0.40, 0.16, 0.25, 0.78])

# Bland-Altmann for D

count = len(Dc)

means = np.mean([Dc, Dpnm], axis=0)
diff = Dc-Dpnm
mean_diff = np.mean(diff)
std_diff = np.std(diff, axis=0)
sd_limit = 1.96

sterr = std_diff / np.sqrt(count)
t_crit = stats.t.ppf(q=1-0.025,df=count-1)
ci = sterr * t_crit

sterr_limit = np.sqrt(1/count+sd_limit**2/(2*(count-1)))*std_diff
ci_limit = sterr_limit * t_crit

#fig = plt.figure(num=89)
#fig.set_size_inches(3.2,3.2)
#plt.clf()
ax2 = fig.add_subplot(1,3,2)
ax2.scatter(np.mean([Dc[0:4], Dpnm[0:4]], axis=0), Dc[0:4]-Dpnm[0:4], marker='o', s=14, c='k')
ax2.scatter(np.mean([Dc[4:8], Dpnm[4:8]], axis=0), Dc[4:8]-Dpnm[4:8], marker='s', s=14, c='b')
ax2.scatter(np.mean([Dc[8:], Dpnm[8:]], axis=0), Dc[8:]-Dpnm[8:], marker='^', s=16, c='r')
ax2.axhline(mean_diff,ls='-',c='k',lw=1.)
ax2.annotate('mean diff.:\n{}'.format(np.round(mean_diff, 3)),
            xy=(0.98, 0.62),
            horizontalalignment='right',
            verticalalignment='center',
            fontsize=fontsize+0,
            xycoords='axes fraction')

half_ylim = (1.75 * sd_limit) * std_diff
ax2.set_ylim(mean_diff - half_ylim,
            mean_diff + half_ylim)
#ax2.yaxis.tick_right()

xmin = 0
xmax = 1.15*np.max(means)
ax2.set_xlim(xmin, xmax)

limit_of_agreement = sd_limit * std_diff
lower = mean_diff - limit_of_agreement
upper = mean_diff + limit_of_agreement
for j, lim in enumerate([lower, upper]):
    ax2.axhline(lim,ls='--',c='k',lw=0.5)
ax2.annotate('-{}sd: {}'.format(sd_limit, np.round(lower, 3)),
            xy=(0.19, 0.11),
            horizontalalignment='left',
            verticalalignment='bottom',
            fontsize=fontsize+0,
            xycoords='axes fraction')
ax2.annotate('+{}sd: {}'.format(sd_limit, np.round(upper, 3)),
            xy=(0.19, 0.83),
            horizontalalignment='left',
            fontsize=fontsize+0,
            xycoords='axes fraction')

x = np.linspace(xmin, xmax)
ax2.fill_between(x, mean_diff-ci, mean_diff+ci, color='silver', alpha=0.25)
ax2.fill_between(x, mean_diff+limit_of_agreement-ci_limit, mean_diff+limit_of_agreement+ci_limit, color='silver', alpha=0.1)
ax2.fill_between(x, mean_diff-limit_of_agreement-ci_limit, mean_diff-limit_of_agreement+ci_limit, color='silver', alpha=0.1)

ax2.yaxis.set_label_coords(-0.18, 0.5)
ax2.xaxis.set_label_coords(0.5, -0.095)

ax2.set_xlabel(r'mean($D_{\mathrm{s}}$,$D_{\mathrm{pnm}}$) (cm$^2$ s$^{-1}$)')
ax2.set_ylabel(r'$D_{\mathrm{s}}-D_{\mathrm{pnm}}$ (cm$^2$ s$^{-1}$)')

ax2.set_position([0.73, 0.16, 0.25, 0.78])

# a meas vs a pnm

slope_a, intercept_a, r_value_a, p_value_a, std_err_a = stats.linregress(ameas, apnm)

ax3 = fig.add_subplot(1,3,3)
ax3.scatter(ameas[0:4], apnm[0:4], marker='o', s=14, c='k', label='0\u20135 cm',zorder=3)
ax3.scatter(ameas[4:8], apnm[4:8], marker='s', s=14, c='b', label='20\u201325 cm',zorder=4)
ax3.scatter(ameas[8:], apnm[8:], marker='^', s=16, c='r', label='40\u201345 cm',zorder=5)
ax3.plot(np.linspace(np.min(ameas),np.max(ameas),100), np.linspace(np.min(ameas),
                    np.max(ameas),100), 'k:', lw=1, label='1:1 line',zorder=2)
ax3.plot(ameas, intercept_a + slope_a * ameas, 'k', lw=1, label = 'Linear fit',zorder=1)

formatter = FormatStrFormatter('%1.1f')
locator = MaxNLocator(nbins=8, steps=[4])
ax3.yaxis.set_major_locator(locator)
ax3.yaxis.set_major_formatter(formatter)
ax3.xaxis.set_major_locator(locator)
ax3.xaxis.set_major_formatter(formatter)

ax3.set_xlabel(r'$a_{\mathrm{meas}}$ (m$^3$ m$^{-3}$)')
ax3.set_ylabel(r'$a_{\mathrm{pnm}}$ (m$^3$ m$^{-3}$)')


ax3.text(0.32 ,0.15, '$y = $' + np.format_float_positional(slope_a, 3) + '$x + $'
        + np.format_float_positional(intercept_a, 3), transform=ax3.transAxes)
ax3.text(0.32, 0.05, '$R^2 = $' + np.format_float_positional(r_value_a**2, 2),
        transform=ax3.transAxes)

lims = [-0.02,0.68]
ax3.set_xlim(lims)
ax3.set_ylim(lims)


#handles, labels = ax3.get_legend_handles_labels()
#order = [2,3,4,1,0]
#ax3.legend([handles[idx] for idx in order],[labels[idx] for idx in order],
#           fontsize=fontsize, loc='upper right', borderaxespad=1.00)

ax3.yaxis.set_label_coords(-0.14, 0.5)
ax3.xaxis.set_label_coords(0.5, -0.095)

ax3.set_position([0.07, 0.16, 0.25, 0.78])

ax.text(0.05, 0.9, '(b)', transform=ax.transAxes)
ax2.text(0.05, 0.9, '(c)', transform=ax2.transAxes)
ax3.text(0.05, 0.9, '(a)', transform=ax3.transAxes)

