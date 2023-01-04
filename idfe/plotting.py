import joypy
import matplotlib.pyplot as plt
import seaborn as sns

from idfe.utils import *

def plot_ridgeline_datasets(df_diam_2darr, ndatasets, mring_mode, xlabelpos, ylabelpos, skip=5, by=None, column=None, color=None, fontsize=20):
    '''
    Plot one histogram per dataset per IDFE method, for all datasets found in the base directory
    '''
    labels = [y if y%skip==0 else None for y in np.arange(ndatasets)]

    # plot 'rex' first and then 'vida'
    joypy.joyplot(df_diam_2darr, by=by, column=column, legend=True, color=color, fill=True, labels=labels, linewidth=0.1, range_style='own', overlap=2)

    # fine-tune the plots
    plt.xlabel(r'diameter $d_0$ ($\mu$as)', fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.text(xlabelpos[0], xlabelpos[1], f'template={mring_mode[0]}{mring_mode[1]} (m-ring mode {mring_mode[1]})', size=fontsize)
    plt.text(ylabelpos[0], ylabelpos[1], f'Dataset id', rotation=90, size=fontsize)
    plt.savefig(f'IDFE_d0_template_{mring_mode[0]}{mring_mode[1]}.png', bbox_inches='tight')
    info(f'Ridgeline plot saved to IDFE_d0_template_{mring_mode[0]}{mring_mode[1]}.png')


def plot_hist_pars(df_output_pars, parlist, parlabellist, dataset, bins=10, fontsize=20):
    '''
    Plot overlapped Rex and VIDA histograms per parameter for a given dataset
    '''

    npars = len(parlist)

    fig, ax = plt.subplots(1, npars, figsize=(npars*5,5))

    for ii in np.arange(npars):
        if npars > 1:
            sns.histplot(data=df_output_pars, x=f'REx_{parlist[ii]}_{dataset}', element='step', color='m', ax=ax[ii], kde=True, bins=bins, alpha=0.3, stat='probability', label='REx')
            sns.histplot(data=df_output_pars, x=f'VIDA_{parlist[ii]}_{dataset}', element='step', color='c', ax=ax[ii], kde=True, bins=bins, alpha=0.3, stat='probability', label='VIDA')
            ax[ii].set_xlabel(f'{parlabellist[ii]}', fontsize=fontsize)
            if ii==0:
                ax[ii].legend()

        else:
            sns.histplot(data=df, x=f'REx_{parlist[ii]}_{dataset}', element='step', color='m', ax=ax, kde=True, bins=bins, alpha=0.3, stat='probability', label='REx')
            sns.histplot(data=df, x=f'VIDA_{parlist[ii]}_{dataset}', element='step', color='c', ax=ax, kde=True, bins=bins, alpha=0.3, stat='probability', label='VIDA')
            ax.set_xlabel(f'{parlabellist[ii]}', fontsize=fontsize)
            if ii==0: ax.legend()

    fig.suptitle(f'{dataset}', fontsize=fontsize)
    fig.tight_layout()
        
    plt.savefig(f'IDFE_{dataset}_histograms.png')
    plt.close(fig)
    info(f'Histogram plots saved to IDFE_{dataset}_histograms.png')


def plot_images_vs_pars(df_output_pars, parlist, parlabellist, dataset, fontsize=20):
    '''
    Plot images vs REx/VIDA estimates
    '''

    npars = len(parlist)
    nimages = np.array(df_output_pars['id']).shape[0]

    fig, ax = plt.subplots(npars, 1, figsize=(10,npars*5))

    for ii in np.arange(npars):

        if npars > 1:
            ax[ii].plot(np.arange(nimages), df_output_pars[f'REx_{parlist[ii]}_{dataset}'], 'm.', label='REx')
            ax[ii].plot(np.arange(nimages), df_output_pars[f'VIDA_{parlist[ii]}_{dataset}'], 'c.', label='VIDA')

            ax[ii].set_ylabel(f'{parlabellist[ii]}')

            if ii == 0:
                ax[ii].legend()

        else:
            ax.plot(np.arange(nimages), df_output_pars[f'REx_{parlist[ii]}_{dataset}'], 'm.', label='REx')
            ax.plot(np.arange(nimages), df_output_pars[f'VIDA_{parlist[ii]}_{dataset}'], 'c.', label='VIDA')

            ax.legend()
            ax.set_ylabel(f'{parlabellist[ii]}')

    plt.tight_layout()
    plt.savefig(f'IDFE_{dataset}_images_vs_pars.png')
    info(f'Comparison plots saved to IDFE_{dataset}_images_vs_pars.png')


def plot_scatter_hists(df_output_pars, parlist, parlabellist, dataset, fontsize=20, bins=10):
    '''
    Plot scatterplots and the corresponding histograms in the same plot
    '''

    npars = len(parlist)
    nimages = np.array(df_output_pars['id']).shape[0]

    fig = plt.figure(figsize=(14, npars*5))

    # define plot margins
    leftmargin=0.12
    rightmargin=0.01
    topmargin=0.02
    bottommargin=0.03
    scatterwidth = 0.75-leftmargin
    histwidth = 0.25-rightmargin
    scatterheight = histheight = (1-(topmargin+bottommargin))/npars

    ax = []
    for ii in np.arange(npars):
        # scatterplot
        ax.append(fig.add_axes([leftmargin, bottommargin+(ii*scatterheight), scatterwidth, scatterheight]))
        if ii == 0:
            legend = True
        else:
            legend = False
        sns.scatterplot(x=np.arange(nimages), y=df_output_pars[f'REx_{parlist[ii]}_{dataset}'], color='m', marker='.', ax=ax[-1], label='REx', legend=legend)
        sns.scatterplot(x=np.arange(nimages), y=df_output_pars[f'VIDA_{parlist[ii]}_{dataset}'], color='c', marker='.', ax=ax[-1], label='VIDA', legend=legend)

        # customize scatterplot
        ax[-1].axes.xaxis.set_visible(False)

        ax[-1].tick_params(axis='y', which='major', labelsize=fontsize)
        ax[-1].tick_params(axis='y', which='minor', labelsize=fontsize/2.)

        ax[-1].set_ylabel(f'{parlabellist[ii]}', fontsize=fontsize)
        if ii == 0:
            ax[-1].legend(loc='upper left', fontsize=fontsize, markerscale=5)

        # histplot
        ax.append(fig.add_axes([leftmargin+scatterwidth, bottommargin+(ii*scatterheight), histwidth, histheight], sharey=ax[-1]))
        sns.histplot(data=df_output_pars, y=df_output_pars[f'REx_{parlist[ii]}_{dataset}'], bins=bins, element='step', kde=True, color='m', ax=ax[-1])
        sns.histplot(data=df_output_pars, y=df_output_pars[f'VIDA_{parlist[ii]}_{dataset}'], bins=bins, element='step', kde=True, color='c', ax=ax[-1])

        # customize histplot
        ax[-1].axes.xaxis.set_visible(False)
        ax[-1].axes.yaxis.set_visible(False)

    plt.savefig(f'IDFE_{dataset}_scatter_histograms.png')
    info(f'Comparison plots saved to IDFE_{dataset}_scatter_histograms.png')
