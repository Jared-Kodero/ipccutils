# ipccutils
This module provides utility functions for IPCC style plots including color maps
Description: Utility functions for creating high-quality, IPCC-style scientific visualizations using matplotlib.

This package standardizes and enhances plot aesthetics for climate-related data visualization, especially for figures styled in accordance with IPCC conventions. It includes pre-defined colormaps, themes, and layout tools.

```
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
        >>> ipcc_cmap["balance"].preview()  # Preview the 'balance' colormap
        >>> ipcc_cmap.balance  # Access the 'balance' colormap
        >>> ipcc_cmap["balance_r"].adjust(N=10, split=0.5)  # Adjust the reversed 'balance' colormap
        >>> ipcc_cmap["#000000", "#ff0000", "#ffffff"].blend(N=60, discrete=True)  # Create a custom colormap

    Managing plots:
        >>> cax = get_cbar_axes()  # Retrieve colorbar axes
        >>> use like this : cb = plt.colorbar(cf,cax=cax,ax=ax,orientation="vertical",drawedges=True)
            Where cf is the contourf object or pcolormesh object

        >>> spine_off()  # Remove top and right spines from the current plot

"""

```

The ipcccutils library I am calling a “package” in quotes because it currently has the core structure of any package you would install using conda or pip; there is an __init__.py file that allows you to access all of the library’s modules and the functions within, using a single from pathlib import Path

from ipccutils import * command. However, this “package” is not available through conda or pip yet. In the meantime, you can get the package by cloning the repo.
