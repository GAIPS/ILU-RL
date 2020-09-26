__author__ = 'Guilherme Varela'
__date__ = '2019-10-24'
import re
from datetime import datetime

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from ilurl.utils.aux import TIMESTAMP, snakefy

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

def scatter_phases(series, label, categories={}, save_path=None):
    """Scatter phases"""

    data = []
    for labels, tls in series.items():
        labelid = labels.index(label)

        for tl, phases in tls.items():
            title_label = snakefy(label)

            for n, points in phases.items():
                # filter data
                data.append(points)

            fig, ax = plt.subplots()
            # Phase iterations
            clr = 'tab:blue'
            ax.set_xlabel(f'{title_label} 0')
            ax.set_ylabel(f'{title_label} 1')
            if any(categories):
                ax.vlines(categories[label], 0, 1,
                          transform=ax.get_xaxis_transform(),
                          colors='tab:green', label=f'Category {title_label}')

                ax.hlines(categories[label], 0, 1,
                          transform=ax.get_yaxis_transform(),
                          colors='tab:cyan', label=f'Category {title_label}')

            ax.scatter(*data, c=clr, label=f'{title_label} 0 x {title_label} 1')
            ax.legend()
            ax.grid(True)
            filename = f'{title_label}x{title_label}'
            suptitle = filename
            if save_path is not None:
                expid = save_path.parts[-2]
                result = re.search(TIMESTAMP, expid)
                if result:
                    timestamp = result.group(0,)
                else:
                    timetamp = expid

                suptitle = f'{filename}\n{timestamp}'
                fig.suptitle(suptitle)
                plt.savefig(save_path / f'{tl}-{filename}-Scatter.png')
            else:
                fig.suptitle(suptitle)
            plt.show()


def scatter_states(series, categories={}, save_path=None):
    """Scatter phases"""
    # xlabel, ylabel = xylabels

    for labels, tls in series.items():
        xlabel, ylabel = labels
        for tl, phases in tls.items():
            fig, axs = plt.subplots(1, 2)
            colors = ['tab:blue', 'tab:red']
            # Phase iterations
            for n, points in phases.items():
                ax = axs[n]
                clr = colors[n]
                ax.set_xlabel(snakefy(xlabel))
                ax.set_ylabel(snakefy(ylabel))


                ax.vlines(categories[xlabel], 0, 1,
                          transform=ax.get_xaxis_transform(),
                          colors='tab:green', label=snakefy(xlabel))

                ax.hlines(categories[ylabel], 0, 1,
                          transform=ax.get_yaxis_transform(),
                          colors='tab:cyan', label=snakefy(ylabel))

                x, y = zip(*points)
                ax.scatter(x, y, c=clr, label=f'Phase {n}')

                ax.legend()
                ax.grid(True)

        filename = f'{snakefy(xlabel)}x{snakefy(ylabel)}'
        suptitle = f'{filename}\n{tl}'

        fig.suptitle(suptitle)
        if save_path is not None:
            plt.savefig(save_path / f'{tl}-{filename}-Scatter.png')
        plt.show()

def get_timestamp(save_path):
    timestamp = ''
    if save_path is not None:
        expid = save_path.parts[-1]
        result = re.search(TIMESTAMP, expid)
        if result:
            timestamp = result.group(0,)
    return timestamp
