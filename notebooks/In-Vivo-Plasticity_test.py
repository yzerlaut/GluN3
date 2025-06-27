# %%

import sys, os
sys.path.append(os.path.join(os.path.expanduser('~'), 
                             'work', 'physion', 'src'))
import physion
from physion.utils import plot_tools as pt
# pt.set_style('dark-notebook')
pt.set_style('manuscript')
import numpy as np

# %%
DATASET = physion.analysis.read_NWB.scan_folder_for_NWBfiles('./data')

# %%
f = DATASET['files'][0]
data = physion.analysis.read_NWB.Data(f)
data.build_dFoF()
data.build_Deconvolved(Tau=0.8)
Ep = physion.analysis.process_NWB.EpisodeData(data,
                                              prestim_duration=3,
                                              quantities=['dFoF', 'Deconvolved'])
# %%

nMax = 6
fig, AX = pt.figure(axes=(nMax, int(data.nROIs/nMax)+1), 
                    wspace=0.4, hspace=0.4, figsize=(1,1))
nTrials = Ep.dFoF.shape[0]
colors = [pt.autumn(r) for r in np.random.uniform(0, 1, data.nROIs)]
for i, ax in enumerate(pt.flatten(AX)):
    if i<data.nROIs:
        for r in range(nTrials):
            ax.plot(Ep.t[::10], Ep.dFoF[r, i, ::10], lw=0.1, 
                    color=colors[i])
        """
        pt.plot(Ep.t, Ep.dFoF[i, :, :].mean(axis=0),
                sy = Ep.dFoF[i,:,:].std(axis=0),
                color=colors[i], ax=ax,lw=0.5)
        """
        pt.shaded_window(ax, xlim=[0,5])
        pt.annotate(ax, 'roi #%i' % (i+1), (1,1), ha='right', color=colors[i])
        pt.draw_bar_scales(ax, Xbar=1, Xbar_label='1s', 
                           Ybar=1, Ybar_label='1$\\Delta$F/F')

    ax.axis('off')

# %%

nTrials = Ep.dFoF.shape[0]

fig, ax = pt.figure(figsize=(2,2))
#ax.plot(Ep.t, Ep.dFoF[:,:int(nTrials/2),:].mean(axis=(0,1)),
ax.plot(Ep.t, Ep.dFoF[:,10:15,:].mean(axis=(0,1)),
        color='tab:blue')
ax.plot(Ep.t, Ep.dFoF[:,int(nTrials/2):,:].mean(axis=(0,1)),
        color='tab:red')

# %%

resps = {'first':[], 'last':[]}

# RESP, sRESP = 'Deconvolved', 'Deconv.'
RESP, sRESP = 'dFoF', '$\Delta$F/F'

for f in DATASET['files'][1:]:

    data = physion.analysis.read_NWB.Data(f)
    data.build_dFoF()
    data.build_Deconvolved(Tau=0.8)
    Ep = physion.analysis.process_NWB.EpisodeData(data,
                                                  prestim_duration=3,
                                                  quantities=['dFoF',
                                                              'Deconvolved',
                                                              'running_speed'])
    
    cond = Ep.running_speed.mean(axis=1)<0.05

    nTrials = Ep.dFoF.shape[0]

    fig, ax = pt.figure(axes=(2,1), figsize=(1.2,1), 
                        wspace=0.7, right=5., top=0.8)
    pt.annotate(fig, f.split('/')[-1], (0.5,1))

    Cond = cond & (np.arange(nTrials)<nTrials/2)
    resps['first'].append(\
        getattr(Ep, RESP)[Cond,:,:].mean(axis=(0,1)))
    ax[0].plot(Ep.t, resps['first'][-1],
            color='tab:blue')
    Cond = cond & (np.arange(nTrials)>nTrials/2)
    resps['last'].append(\
        getattr(Ep, RESP)[Cond,:,:].mean(axis=(0,1)))
    ax[0].plot(Ep.t, resps['last'][-1],
            color='tab:red')
    for i in np.arange(nTrials)[cond]:
        ax[1].plot(Ep.t, 
                   getattr(Ep, RESP)[i,:,:].mean(axis=0),
                   lw=0.1)
    for x in ax:
        pt.shaded_window(x, xlim=[0,5])
        pt.set_plot(x, 
                    xlabel='time (s)', 
                    ylabel='$\langle$ %s $\\rangle_{ROIs}$' % sRESP)
    pt.annotate(ax[0], '1-%i' % (int(nTrials/2)), (0,1), color='tab:blue')
    pt.annotate(ax[0], '<-trials->', (0.5, 1), ha='center')
    pt.annotate(ax[0], '%i-%i' % (int(nTrials/2)+1, nTrials), (1,1), 
                color='tab:red', ha='right')
    ax[1].set_title('all trials (n=%i)' % nTrials)

    inset = pt.inset(ax[1], (1.3,0.,0.5,1))
    physion.dataviz.imaging.show_CaImaging_FOV(data, ax=inset,
                                               roiIndices=range(data.nROIs))
resps['t'] = Ep.t
# 

# %%
from scipy import stats
fig, ax = pt.figure(right=4)

pre = (resps['t']>-1) & (resps['t']<0)
post = (resps['t']>0) & (resps['t']<1)

for key in ['first', 'last']:
        resps['%s_norm' % key] = np.array([\
                r-np.mean(np.array(r)[pre])\
                        for r in resps[key]])
        resps['%s_norm' % key] = np.array([\
                r/np.max(r[post])\
                        for r in resps['%s_norm' % key]])

pt.plot(resps['t'], np.mean(resps['first_norm'],axis=0),
        sy=stats.sem(resps['first_norm'], axis=0),
        color='tab:blue', ax=ax)
pt.plot(resps['t'], np.mean(resps['last_norm'],axis=0),
        sy=stats.sem(resps['last_norm'], axis=0),
        color='tab:red', ax=ax)

pt.set_plot(ax, xlabel='time (s)', ylabel='%s\n(peak norm.)' % sRESP,
            yticks=[0,1])
pt.annotate(ax, 'first', (1,1), va='top', ha='right', color='tab:blue')
pt.annotate(ax, '\nlast', (1,1), va='top', ha='right', color='tab:red')

inset = pt.inset(ax, [1.7,0.1,0.2,0.8])

window = (resps['t']>0) & (resps['t']<5)
pt.bar([np.mean(100*np.mean(resps['last_norm'][:,window]-\
                            resps['first_norm'][:,window], axis=1))],
       sy=[stats.sem(100*np.mean(resps['last_norm'][:,window]-\
                                 resps['first_norm'][:,window], axis=1))],
       ax=inset)
pt.set_plot(inset, ['left'], 
            title='N=%i' % len(resps['last']),
            ylabel='% change\n $\langle$ last-first $\\rangle$')

# %%
