"""
Description:
    This module provides utility functions for ipcc style plots

Author:
    Mabere Kodero

Available Functions and Constants:

- Visualization:
    - `set_plot_theme`: Sets the IPCC plot theme for matplotlib.
    - `ipcc_cmap`: Provides colormaps with IPCC color schemes.
    - `get_text_color`: Determines an appropriate text color based on the background.
    - `spine_off`: Removes spines from matplotlib plots.
    - `get_cbar_axes`: Retrieves axes for colorbars in matplotlib plots.

Example Usage:

    Importing the module:
        >>> from pathlib import Path

from ipccutils import *
from tools import *  # Imports all functions and constants

    Setting the plot theme:
        >>> set_plot_theme(font_scale=1.5, line_width=1.5, latex=False)

    Working with colormaps:
        >>> ipcc_cmap.balance  # Access the 'balance' colormap
        >>> ipcc_cmap["balance_r"].adjust(N=10, split=0.5)  # Adjust the reversed 'balance' colormap
        >>> ipcc_cmap["#000000", "#ff0000", "#ffffff"].blend(N=60, discrete=True)  # Create a custom colormap

    Managing plots:
        >>> cax = get_cbar_axes()  # Retrieve colorbar axes
        >>> use like this : cb = plt.colorbar(cf,cax=cax,ax=ax,orientation="vertical",drawedges=True)
            Where cf is the contourf object or pcolormesh object

        >>> spine_off()  # Remove top and right spines from the current plot

"""

from .colors import ipcc_cmap, set_plot_theme, spine_off
from .plot import cartplot, create_map_figure, get_cbar_axes

__all__ = [
    "set_plot_theme",
    "ipcc_cmap",
    "spine_off",
    "get_cbar_axes",
    "create_map_figure",
    "cartplot",
]
