import re
from datetime import datetime

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.cm as cmx
from matplotlib import colors as mcolors


from ilurl.utils.aux_tools import TIMESTAMP, snakefy

def plot_times(times, series, series_labels, xy_labels, title):
    """ Makes an hourly plot of series

    PARAMETERS:
    ----------
    * times: list of strings
        those are the times that will represent the x-axis

    * series: list or list of lists
        those are the values that will be plotted over y-axis, if
        it's containts are also lists than is a multi-series

    * series_labels: string or list of strings
        use string for single series or list of strings for multiple
        series.

    * xy_labels: None or list of strings
       If present expects to have two values xlabel and ylabel respectively

    * title: None or string
       A glorious title

   USAGE:
   -----
   > times = ["00:00:00", "00:15:00", "00:30:00", ..., "23:45:00"]
   > series =[[20, 300, 327.5, ... 20], [10, 45, 27, ..., 5]]
   > series_labels = ["means", "std"]
   > xy_labels = ["Times", "# vehicles"]
   > title = "Induction loop reading"
   > plot_times(times, series, series_labels, xy_labels, title)

   REFS:
   ----
   * Time plots
     See https://stackoverflow.com/questions/

     13515471/matplotlib-how-to-prevent-x-axis-labels-from-overlapping-each-other#13521621
     1574088/plotting-time-in-python-with-matplotlib#1574146
     14946371/editing-the-date-formatting-of-x-axis-tick-labels-in-matplotlib

    """
    num_series = len(series_labels)
    if num_series > 2:
        raise ValueError("Only 1 or 2 series are supported")

    time_only_format = mdates.DateFormatter("%H:%M")
    fig, ax = plt.subplots()
    ax.xaxis.set_major_formatter(time_only_format)

    ax.plot(times, series[0], label=series_labels[0])
    if num_series == 2:
        ax.plot(times, series[1], marker='o', linestyle='--', color='r', label=series_labels[1])

    ax.set_xlabel('Time')
    ax.set_ylabel('# vehicles')

    if title:
        ax.set_title(title)

    ax.xaxis_date()
    fig.autofmt_xdate()
    plt.legend()
    plt.show()

def scatter_phases(series, label, categories={}, save_path=None,
                    network=None, rewards=[], reward_is_penalty=False):
    """Scatter e.g delay 0 vs delay 1

        * Given a feature plot phases
    Params:
        series: dict<tuple<dict<str,dict<int, <list<float>>>>>
            nested dictionary
            ('delay', 'delay') --> '247123161' --> 0 --> [0.0, ...]

        label: str
            e.g 'delay', 'count'

        category: dict<str, list<float>>
            {'delay': [0.0, 0.06, 0.4, 0.82, 1.31]}

        rewards: list<dict<str, float>> optional
            e.g [{'247123161': -0.1118}, ..., {'247123161': -0.2283}]

        network: str
            e.g intersection, grid, grid_6

        save_path: pathlib.Path
            20200923143127.894361/
                intersection_20200923-1431271600867887.8995893/
                    log_plots
    """
    for labels, tls in series.items():
        labelid = labels.index(label)

        for tl, phases in tls.items():
            title_label = snakefy(label)
            data = []
            for n, points in phases.items():
                # filter data
                data.append(points)
            _rewards = [r[tl] for r in rewards]

            fig, ax = plt.subplots()
            # Phase iterations

            ax.set_xlabel(f'{title_label} 0')
            ax.set_ylabel(f'{title_label} 1')
            set_categories(ax, [label], categories[tl])

            # Plot the gradient colors
            clr = make_colors(ax, data, _rewards, reward_is_penalty)

            ax.scatter(*data, c=clr, label=f'{title_label} 0 x {title_label} 1')
            ax.legend()
            ax.grid(True)

            save_scatter(fig, tl, (label,), network=network, save_path=save_path)
            plt.show()


def scatter_states(series, categories={}, save_path=None, network=None,
                   rewards=[], reward_is_penalty=False, reward_function=None):
    """States e.g delay 0 vs delay 1

        * Given a dual plot of phases 0-1 two different 

    Params:
        series: dict<tuple<dict<str,dict<int, <list<float>>>>>
            ('speed', 'count') --> '247123161' --> 0 --> [[0.81, 0.92], ..]

        category: dict<str, list<float>>
            {'delay': [0.0, 0.06, 0.4, 0.82, 1.31]}

        save_path: pathlib.Path
            20200923143127.894361/
                intersection_20200923-1431271600867887.8995893/
                    log_plots

        network: str
            e.g intersection, grid, grid_6
    """
    fnc = reward_function
    absolutes = []
    partials = []
    for labels, tls in series.items():
        xlabel, ylabel = labels
        for tl, phases in tls.items():
            fig, axs = plt.subplots(1, 2)
            colors = ['tab:blue', 'tab:red']
            # Phase iterations
            for n, points in phases.items():
                ax = axs[n]
                x, y = zip(*points)
                data = [x, y]
                pr, ar = partial_rewards(fnc, tl, rewards, points)
                partials += pr
                absolutes += ar
                clr = make_colors(ax, data, pr, reward_is_penalty=reward_is_penalty)
                ax.set_xlabel(snakefy(xlabel))
                ax.set_ylabel(snakefy(ylabel))
                
                set_categories(ax, labels, categories)

                ax.scatter(*data, c=clr, label=f'Phase {n}')

                ax.legend()
                ax.grid(True)

            totr = sum([rr[tl] for rr in rewards])
            try:
                assert abs(sum(absolutes) - totr) / totr < 1e-3
                assert abs(sum(partials) - len(rewards)) / len(rewards) < 1e-3
            except AssertionError:
                print('Warning: Assertion Error: Rounding errors overflow')
            save_scatter(fig, tl, labels, network=network, save_path=save_path)
        plt.show()

def set_categories(ax, labels, categories):
    """Applies gradient coloring for rewards

    Params:
        * ax: matplotlib.axes._subplots.AxesSubplot
            Axis object
        * labels: array-like
            One or Two sized array with x and y labels.
        * categories: dict<tuple(str, str), list<float>>
            Vertical and Horizontal categories
    """
    if any(categories):
        xlabel, ylabel = labels[0], labels[-1]
        ax.vlines(categories[xlabel]['0'], 0, 1,
                  transform=ax.get_xaxis_transform(),
                  colors='tab:purple', label=f'category {snakefy(xlabel)}')

        ax.hlines(categories[ylabel]['1'], 0, 1,
                  transform=ax.get_yaxis_transform(),
                  colors='tab:cyan', label=f'category {snakefy(ylabel)}')

def make_colors(ax, data, rewards, reward_is_penalty=False):
    """Applies gradient coloring for rewards
        Params:
            * ax: matplotlib.axes._subplots.AxesSubplot
                Axis object
            * data: list<?>
                List with series.
 
            * rewards: array-like
                List with either rewards or penalties.

            * reward_is_penalty: bool
                if True then rewards are actually a penalty  

        Returns:
            * clr: str or numpy.ndarray
                'tab:blue' or array of gradient
    """
    # TODO: FIX gradient
    # if any(rewards):
    #     rnorm = normalize_rewards(rewards, reward_is_penalty)
    #     cmap = plt.get_cmap('RdPu')
    #     cnorm = mcolors.Normalize(vmin=min(rnorm), vmax=max(rnorm))
    #     smap = cmx.ScalarMappable(norm=cnorm, cmap=cmap)

    #     clr = smap.to_rgba(rnorm)
    #     data.append(rnorm)
    #     cbar = plt.colorbar(mappable=smap, ax=ax)
    # else:
    clr = 'tab:blue'
    return clr

def save_scatter(fig, tl, labels, network=None, save_path=None):
    """Creates subtitle and save scatter plot

    Params:
        * fig: matplotlib.figure.Figure
            Two axis figure

        * tl: str
            The index of the traffic light e.g '247123161'

        * labels: array-like
            One or Two sized array with x and y labels.

        * network: str
            map name e.g intersection, grid, grid_6

        * save_path: pathlib.Path
            The path to save
 
    """
    xlabel, ylabel = snakefy(labels[0]), snakefy(labels[-1])
    filename = f'{xlabel}x{ylabel}'
    suptitle = f'{xlabel} x {ylabel}'
    if save_path is not None:
        # with timestamp
        # timestamp = get_timestamp(save_path)
        # suptitle = f'{suptitle}\n{timestamp}'
        if network is not None:
            suptitle = f'{snakefy(network)}-{tl}\n{suptitle}'
        fig.suptitle(suptitle)
        plt.savefig(save_path / f'{tl}-{filename}-Scatter.png')
    else:
        fig.suptitle(suptitle)

def get_timestamp(save_path):
    """Extract timestamp from path
    
    Params:
        * save_path: pathlib.Path
 
    Returns:
        * timestamp: str
            e.g '20200923-1448311600868911.0831687'
    """
    timestamp = ''
    if save_path is not None:
        expid = save_path.parts[-2]
        result = re.search(TIMESTAMP, expid)
        if result:
            timestamp = result.group(0,)
    return timestamp

def partial_rewards(reward_function, tl, total_rewards, phase_states):
    """Computes the reward with respect to 1 phase of the state
    
    Params:
        * reward_function: function
            Reward used during training.

        * tl: str
            Id for traffic light agents.

        * total_rewards: list<dict<str, list<float>>>
            Rewards saved during training.

        * phase_states: list<list<float>>
            Phase states each element of the outer list
            is the phase state.

    Returns:

        * partials: list<float> 
            List of the fraction of the reward due to phase.

        * absolutes: list<float>
            List of the reward function evaluated on the phase.
    """
    partials = []
    absolutes = []
    if any(total_rewards):
        for state, reward in zip(phase_states, total_rewards):
            pr = reward_function({tl: state})
            absolutes.append(pr[tl])
            partials.append(pr[tl] / reward[tl] if bool(reward[tl]) else 0)
    return partials, absolutes
        
    

def normalize_rewards(rewards, reward_is_penalty=False):
    """Corrects rewards to be in the interval 0-1
   
        * If reward is actually a penalty it inverts 
    Params:
        * rewards: list
            list with the rewards from a tls
 
    Returns:
        * reward_is_penalty: bool
            If True then the reward is actually a penalty 
            default False
    """
    # Switch signals
    if reward_is_penalty:
        _rewards = [-r for r in rewards]
    else:
        _rewards = rewards
    rmin = min(_rewards)
    rmax = max(_rewards)
    return [(rwr - rmin) / (rmax - rmin) for rwr in _rewards]

