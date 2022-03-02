
from dimod.vartypes import Vartype
import dimod
import matplotlib.pyplot as plt
from collections import Counter
import pandas as pd
import numpy as np
import functools
from typing import List, Union
import matplotlib.colors as mcolors

# %%


def plotEnergyValuesDwaveSampleSet(
    results: dimod.SampleSet,
    title: str = None,
):
    '''
    Plots the energy values of the samples in a bar plot using as an impit a Dmid.sampleset.

    Args:
        results: A dimod.SampleSet object.
        title: A string to use as the plot title.

    Returns:
        A matplotlib.pyplot.Figure object.
    '''

    _, ax = plt.subplots()

    energies = [datum.energy for datum in results.data(
        ['energy'], sorted_by='energy')]

    if results.vartype == dimod.Vartype.BINARY:
        samples = [''.join(c for c in str(datum.sample.values()).strip(
            ', ') if c.isdigit()) for datum in results.data(['sample'], sorted_by=None)]
        ax.set(xlabel='bitstring for solution')
    else:
        samples = np.arange(len(energies))
        ax.set(xlabel='solution')

    ax.bar(samples, energies)
    ax.tick_params(axis='x', rotation=90)
    ax.set_ylabel('Energy')
    if title:
        ax.set_title(str(title))
    print("minimum energy:", min(energies))
    return ax


def plotBarValues(
    df: pd.DataFrame,
    column_name: str,
    sorted: bool = True,
    skip: int = 1,
    xlabel: str = None,
    ylabel: str = None,
    title: str = None,
    save_fig: bool = False,
    **kwargs,
) -> plt.Figure:
    '''
    Plots the values of a column in a bar plot.

    Args:
        df: A pandas dataframe.
        column_name: A string of the column name.
        sorted: A boolean to sort the dataframe.
        skip: An integer of the number of rows to skip.
        xlabel: A string of the xlabel.
        ylabel: A string of the ylabel.
        title: A string of the title.
        save_fig: A boolean to save the figure.

    Returns:
        A matplotlib.pyplot.Figure object.
    '''
    _, ax = plt.subplots()
    if sorted:
        df = df.sort_values(by=column_name)
    df.plot(y=column_name, kind='bar', ax=ax, **kwargs)
    ax.figure.tight_layout()
    new_ticks = np.arange(0, len(df.index) // skip)*skip
    # positions of each tick, relative to the indices of the x-values
    if column_name == 'sample':
        new_ticks
    ax.set_xticks(new_ticks)
    # labels
    ax.set_xticklabels(new_ticks)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    if save_fig:
        plt.savefig(title+'.png')
    return ax


def plotBarCounts(
    df: pd.DataFrame,
    column_name: str,
    normalized: bool = False,
    sorted: bool = True,
    ascending: bool = False,
    skip: int = 1,
    xlabel: str = None,
    title: str = None,
    save_fig: bool = False,
    **kwargs,
) -> plt.Figure:
    '''
    Plots the counts of a column in a bar plot.

    Args:
        df: A pandas dataframe.
        column_name: A string of the column name.
        normalized: A boolean to normalize the counts.
        sorted: A boolean to sort the dataframe.
        ascending: A boolean to sort the plot in ascending order.
        skip: An integer of the number of rows to skip.
        xlabel: A string of the xlabel.
        title: A string of the title.
        save_fig: A boolean to save the figure.

    Returns:
        A matplotlib.pyplot.Figure object.
    '''
    _, ax = plt.subplots()
    series = df[column_name].value_counts(normalize=normalized)
    if sorted:
        series = series.sort_values(ascending=ascending)
    series.plot(kind='bar', ax=ax, **kwargs)
    ax.figure.tight_layout()
    if column_name == 'sample':
        new_ticks = [str(list(state.values())).replace(', ', '')
                     for i, state in enumerate(series.keys()) if not i % skip]
    else:
        new_ticks = [t.get_text()[:7] for i, t in enumerate(ax.get_xticklabels()) if not i %
                     skip]
    ax.set_xticks(ax.get_xticks()[::skip])
    # labels
    ax.set_xticklabels(new_ticks)
    if xlabel:
        ax.set_xlabel(xlabel)
    if normalized:
        ax.set_ylabel('Probability')
    else:
        ax.set_ylabel('Count')
    if title:
        ax.set_title(title)
    if save_fig:
        plt.savefig(title+'.png')
    return ax


def plotSamplesDwaveSampleSet(
    results: dimod.SampleSet,
    title: str = None,
    skip: int = 1,
):
    '''
    Plots the samples of the samples in a histogram.

    Args:
        results: A dimod.SampleSet object.
        title: A string to use as the plot title.
        skip: An integer to skip every nth sample.

    Returns:
        A matplotlib.pyplot.Figure object.
    '''
    _, ax = plt.subplots()
    energies = results.data_vectors['energy']
    if results.vartype == dimod.Vartype.BINARY:
        samples = [''.join(c for c in str(datum.sample.values()).strip(
            ', ') if c.isdigit()) for datum in results.data(['sample'], sorted_by=None)]
        ax.set_xlabel('bitstring for solution')
    else:
        samples = np.arange(len(results))
        ax.set_xlabel('solution')

    counts = Counter(samples)
    total = len(samples)
    for key in counts:
        counts[key] /= total
    df = pd.DataFrame.from_dict(counts, orient='index').sort_index()
    df.plot(kind='bar', legend=None, ax=ax)

    ax.tick_params(axis='x', rotation=80)
    ax.set_xticklabels([t.get_text()[:7] if not i %
                       skip else "" for i, t in enumerate(ax.get_xticklabels())])
    ax.set_ylabel('Probabilities')
    if title:
        ax.set_title(str(title))
    print("minimum energy:", min(energies))
    return ax


def plotEnergyCFDDwaveSampleSet(
    results: dimod.SampleSet,
    title: str = None,
    skip: int = 1,
):
    '''
    Plots the energy values of the samples in a cumulative distribution function.

    Args:
        results: A dimod.SampleSet object.
        title: A string to use as the plot title.
        skip: An integer to skip every nth sample.

    Returns:
        A matplotlib.pyplot.Figure object.
    '''
    _, ax = plt.subplots()
    # skip parameter given to avoid putting all xlabels
    energies = results.data_vectors['energy']
    occurrences = results.data_vectors['num_occurrences']
    counts = Counter(energies)
    total = sum(occurrences)
    counts = {}
    for index, energy in enumerate(energies):
        if energy in counts.keys():
            counts[energy] += occurrences[index]
        else:
            counts[energy] = occurrences[index]
    for key in counts:
        counts[key] /= total
    df = pd.DataFrame.from_dict(counts, orient='index').sort_index()
    df.plot(kind='bar', legend=None, ax=ax)
    ax.set_xticklabels([t.get_text()[:7] if not i %
                       skip else "" for i, t in enumerate(ax.get_xticklabels())])

    ax.set_xlabel('Energy')
    ax.set_ylabel('Probabilities')
    if title:
        ax.set_title(str(title))
    print("minimum energy:", min(energies))
    return ax


def plot_1d_singleinstance(
    df: pd.DataFrame,
    x_axis: str,
    y_axis: str,
    label_plot: str,
    dict_fixed: dict,
    ax: plt.Axes,
    log_x: bool = False,
    log_y: bool = False,
    save_fig: bool = False,
    labels: dict = None,
    prefix: str = '',
    plots_path: str = './plots/',
    default_dict: dict = None,
    color: str = None,
    fail_value: float = None,
    **kwargs,
) -> plt.Axes:
    '''
    Function to plot 1D dependance figures

    Args:
        df: Dataframe with the data to plot
        x_axis: Name of the x axis
        y_axis: Name of the y axis
        label_plot: Name of the plot
        dict_fixed: Dictionary with the fixed values
        ax: Axes to plot on
        log_x: Logarithmic x axis
        log_y: Logarithmic y axis
        save_fig: Save the figure
        labels: Dictionary with the labels
        prefix: Prefix to add to the title
        plots_path: Path to save the figure
        default_dict: Dictionary with the default values
        color: Color of the plot
        fail_value: Value to use when run has failed
        **kwargs: Additional arguments to pass to the plot function

    Returns:
        ax: Axes with the plot
    '''
    if fail_value is None:
        fail_value = np.inf
    
    if color is None:
        color = 'grey'

    # Condition to enforce dict_fix conditions
    if dict_fixed is not None:
        cond = [df[k].apply(lambda k: k == v).astype(bool)
                for k, v in dict_fixed.items()]
        cond_total = functools.reduce(lambda x, y: x & y, cond)
        working_df = df[cond_total]
    else:
        working_df = df.copy()
    if y_axis == 'tts' or y_axis == 'tts_scaled':
        working_df.mask(
            working_df[y_axis] == fail_value).sort_values(
            x_axis
        ).plot(
            ax=ax,
            x=x_axis,
            y=y_axis,
            label=label_plot,
            style='-',
            color=color,
            **kwargs,)
    else:
        working_df.sort_values(
            x_axis
        ).plot(
            ax=ax,
            x=x_axis,
            y=y_axis,
            label=label_plot,
            style='-*',
            color=color,
            **kwargs,)

    if y_axis + '_conf_interval_lower' in df.columns and y_axis + '_conf_interval_upper' in df.columns:
        ax.fill_between(
            working_df.sort_values(x_axis)[x_axis],
            working_df.sort_values(x_axis)[y_axis + '_conf_interval_lower'],
            working_df.sort_values(x_axis)[y_axis + '_conf_interval_upper'],
            alpha=0.2,
        )

    if default_dict is not None:
        cond_default = [df[k].apply(lambda k: k == v).astype(bool)
                        for k, v in default_dict.items()]
        cond_default_total = functools.reduce(lambda x, y: x & y, cond_default)
        default_val = df[cond_default_total][y_axis].values
        if default_val.size != 0:
            default_val = default_val[0]
            ax.axhline(default_val,
                       label='Default',
                       linestyle='--')

    ax.legend()
    if y_axis in labels.keys():
        ax.set(ylabel=labels[y_axis])
    else:
        ax.set(ylabel=y_axis)
    if x_axis in labels.keys():
        ax.set(xlabel=labels[x_axis])
    else:
        ax.set(xlabel=x_axis)
    if log_x:
        ax.set(xscale='log')
    if log_y:
        ax.set(yscale='log')
    ax.set(title='Plot ' + prefix +
           '\n' + y_axis + ' dependance with ' + x_axis + ', \n' +
                 ', '.join(str(key) + '=' + str(value) for key, value in dict_fixed.items()))

    if save_fig:
        plt.savefig(
            plots_path + y_axis + '_' + x_axis + '_fixed_' + '_'.join(str(key)
                                                                      for key in dict_fixed.keys())
            + '.png')

    return ax


def plot_1d_singleinstance_list(
    df: pd.DataFrame,
    x_axis: str,
    y_axis: str,
    list_dicts: List[dict],
    ax: plt.Axes,
    dict_fixed: dict = None,
    log_x: bool = False,
    log_y: bool = False,
    use_colorbar: bool = False,
    save_fig: bool = False,
    labels: dict = None,
    prefix: str = '',
    plots_path: str = './plots/',
    default_dict: dict = None,
    colors: List[str] = None,
    colormap: plt.cm = None,
    fail_value: float = None,
    **kwargs,
) -> plt.Axes:
    '''
    Function to plot 1D dependance figures

    Args:
        df: Dataframe with the data to plot
        x_axis: Name of the x axis
        y_axis: Name of the y axis
        list_dicts: List of dictionaries with the values to plot
        dict_fixed: Dictionary with the fixed values
        ax: Axes to plot on
        log_x: Logarithmic x axis
        log_y: Logarithmic y axis
        use_colorbar: Use colorbar
        save_fig: Save the figure
        labels: Dictionary with the labels to use
        prefix: Prefix to add to the title
        plots_path: Path to save the figure
        default_dict: Dictionary with the default values
        colormap: Colormap to use
        fail_value: Value to use when run has failed
        **kwargs: Additional arguments to pass to the plot function

    Returns:
        ax: Axes with the plot
    '''
    single_instance = False
    if fail_value is None:
        fail_value = np.inf
    if 'alpha' in kwargs:
        single_instance = True
    if colormap is None:
        colormap = plt.cm.rainbow
    if colors is None:
        # Default colors for matplotlib
        colors = [u'#1f77b4', u'#ff7f0e', u'#2ca02c', u'#d62728', u'#9467bd',
                  u'#8c564b', u'#e377c2', u'#7f7f7f', u'#bcbd22', u'#17becf']
    elif colors == ['colormap']:
        colors = colormap(np.linspace(0, 1, len(list_dicts)))
    title_str = 'Plot ' + \
        prefix + '\n' + \
        y_axis + ' dependance with ' + x_axis

    # Condition to enforce dict_fix conditions
    fixed_instance = False
    if dict_fixed is not None:
        if 'instance' not in dict_fixed.keys():
            title_str += ', \n Ensemble'
        else:
            fixed_instance = True
        cond = [df[k].apply(lambda k: k == v).astype(bool)
                for k, v in dict_fixed.items()]
        cond_total = functools.reduce(lambda x, y: x & y, cond)
        working_df = df[cond_total]
        title_str += ', \n' + \
            ', '.join(str(key) + '=' + str(value)
                      for key, value in dict_fixed.items())
    else:
        working_df = df.copy()
    for i, line in enumerate(list_dicts):
        cond_part = [working_df[k].apply(lambda k: k == v).astype(bool)
                     for k, v in line.items()]
        cond_partial = functools.reduce(lambda x, y: x & y, cond_part)
        label_plot = ', '.join(str(key) + '=' + str(value)
                               for key, value in line.items())
        if 'instance' not in line.keys() and not fixed_instance:
            label_plot = 'Ensemble, ' + label_plot
        if len(working_df[cond_partial]) == 0:
            continue
        else:
            if y_axis == 'tts' or y_axis == 'tts_scaled':
                working_df[cond_partial].mask(
                    working_df[cond_partial][y_axis] == fail_value).sort_values(
                    x_axis
                ).plot(
                    ax=ax,
                    x=x_axis,
                    y=y_axis,
                    label=label_plot if not single_instance else '',
                    color=colors[i],
                    style='-*' if not single_instance else '-',
                    **kwargs,)
            else:
                working_df[cond_partial].sort_values(
                    x_axis
                ).plot(
                    ax=ax,
                    x=x_axis,
                    y=y_axis,
                    label=label_plot if not single_instance else '',
                    color=colors[i],
                    style='-*' if not single_instance else '-',
                    **kwargs,)

        if y_axis + '_conf_interval_lower' in df.columns and \
            y_axis + '_conf_interval_upper' in df.columns and \
                not single_instance and \
        working_df[cond_partial][y_axis + '_conf_interval_lower'].values.shape[0] > 1:
            ax.fill_between(
                working_df[cond_partial].sort_values(x_axis)[x_axis],
                working_df[cond_partial].sort_values(
                    x_axis)[y_axis + '_conf_interval_lower'],
                working_df[cond_partial].sort_values(
                    x_axis)[y_axis + '_conf_interval_upper'],
                alpha=0.2,
                color=colors[i],
            )

    if default_dict is not None:
        cond_default = [df[k].apply(lambda k: k == v).astype(bool)
                        for k, v in default_dict.items()]
        cond_default_total = functools.reduce(lambda x, y: x & y, cond_default)
        default_val = df[cond_default_total][y_axis].values
        if default_val.size != 0:
            default_val = default_val[0]
            ax.axhline(default_val,
                       label='Default',
                       linestyle='--')

    if not single_instance:
        if use_colorbar:
            ax.get_legend().remove()
            # We assign the colormap according to the first key of the list_dicts, which we assume is sorted
            list_colorbar = [list(i.values())[0] for i in list_dicts]
            # Setup the colorbar
            normalize = mcolors.Normalize(
                vmin=min(list_colorbar), vmax=max(list_colorbar))
            scalarmappaple = plt.cm.ScalarMappable(
                norm=normalize, cmap=colormap)
            scalarmappaple.set_array(list_colorbar)
            plt.colorbar(scalarmappaple, ax=ax, label=', '.join(
                labels[key] for key in line.keys()))
        else:
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    if y_axis in labels.keys():
        ax.set(ylabel=labels[y_axis])
    else:
        ax.set(ylabel=y_axis)
    if x_axis in labels.keys():
        ax.set(xlabel=labels[x_axis])
    else:
        ax.set(xlabel=x_axis)
    if log_x:
        ax.set(xscale='log')
    if log_y:
        ax.set(yscale='log')

    ax.set(title=title_str)

    if save_fig:
        plt.savefig(
            plots_path + y_axis + '_' + x_axis + '_fixed_' + '_'.join(str(key)
                                                                      for key in dict_fixed.keys())
            + '_list_' + '_'.join(str(key) for key in list_dicts[0].keys())
            + '.png')

    return ax
