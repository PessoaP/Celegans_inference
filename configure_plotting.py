import matplotlib.pyplot as plt
import seaborn as sns
from cycler import cycler

def configure_plotting():
    params = {
        'legend.fontsize': 8,
        'figure.figsize': (6.5, 3),
        'axes.labelsize': 8,
        'axes.titlesize': 8,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'lines.markersize': 5,
        'lines.linewidth': 1
    }
    plt.rcParams.update(params)

    custom_palette = [
        "#E69F00", "#009E73", "#CC79A7",
        "#0072B2", "#D55E00", "#56B4E9", "#F0E442"
    ]
    linestyles = ['-', '-', '--', '--', '-.', ':', '-']

    assert len(custom_palette) == len(linestyles), "Color and line style lists must match"

    sns.set_palette(custom_palette)
    plt.rcParams['axes.prop_cycle'] = cycler(color=custom_palette) + cycler(linestyle=linestyles)
