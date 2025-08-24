"""
ipccutils — IPCC-Style Plotting Utilities
============================================

Description:
------------
This module provides utility functions to generate scientific plots styled
according to IPCC (Intergovernmental Panel on Climate Change) visual standards.
It includes predefined color themes, colormap management tools, spine trimming,
colorbar utilities, and high-level Cartopy-based map plotting routines.

Author:
-------
Jared M. Kodero

Dependencies:
-------------
- matplotlib
- seaborn
- numpy
- xarray
- cartopy

Main Features:
--------------

1. Plot Styling
---------------
>>> set_plot_theme(font_scale=1.5, line_width=1.5, latex=False)
    # Sets global plotting style for consistent appearance.
    # - font_scale: scaling for all fonts (default=1.5)
    # - line_width: width for plot lines (default=1.5)
    # - latex: enable LaTeX text rendering (default=False)

>>> spine_off()
    # Removes top and right plot spines for a cleaner look.

>>> color = get_text_color("#dddddd")
    # Returns a contrasting color ('black' or 'white') suitable for text over the given background.

>>> cax = get_cbar_axes(ax=None, orientation='vertical', shrink=0.9, pad=0.05)
    # Returns axes for placing a colorbar, using the current or provided axis.
    # Use as:
    >>> cb = plt.colorbar(cf, cax=cax, ax=ax, orientation="vertical", drawedges=True)

2. Colormaps
------------
>>> ipcc_cmap.balance
    # Access the default 'balance' diverging colormap

>>> ipcc_cmap["balance_r"].adjust(N=10, split=0.5)
    # Adjust reversed 'balance' colormap to 10 levels with 50% midpoint

>>> ipcc_cmap["#000000", "#ff0000", "#ffffff"].blend(N=60, discrete=True)
    # Create a discrete custom colormap from 3 colors

3. Map Creation
---------------
>>> fig, ax = create_map_figure(figsize=(10, 6))
    # Create a matplotlib figure suitable for geospatial plotting


- All other arguments are optional and can be used for extensive customization.
- Designed to produce figures consistent with IPCC WG-style reports.
- Use the provided colormap methods (`adjust`, `blend`) to match your data scale and color symmetry needs.

"""

from .colors import ipcc_cmap, set_plot_theme, spine_off

__all__ = [
    "ipcc_cmap",
    "set_plot_theme",
    "spine_off",
]
