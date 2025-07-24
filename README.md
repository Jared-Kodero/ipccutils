# ipccutils
This module provides utility functions for IPCC style plots including color maps
Description: Utility functions for creating high-quality, IPCC-style scientific visualizations using matplotlib.

This package standardizes and enhances plot aesthetics for climate-related data visualization, especially for figures styled in accordance with IPCC conventions. It includes pre-defined colormaps, themes, and layout tools.

```
"""
ipccutils.py — IPCC-Style Plotting Utilities
============================================

Description:
------------
This module provides utility functions to generate scientific plots styled 
according to IPCC (Intergovernmental Panel on Climate Change) visual standards.
It includes predefined color themes, colormap management tools, spine trimming, 
colorbar utilities, and high-level Cartopy-based map plotting routines.

Author:
-------
Mabere Kodero

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

4. CartoPy Plotting
-------------------

>>> import numpy as np
>>> import xarray as xr
>>> da = xr.DataArray(
        np.random.rand(10, 10),
        dims=["lat", "lon"],
        coords={"lat": np.linspace(-90, 90, 10), "lon": np.linspace(-180, 180, 10)}
    )

>>> cartplot(
        data=da,
        plot_type="default",            # or 'pcolormesh', 'contour', 'contourf'
        projection="PlateCarree",       # Cartopy projection name or object
        central_longitude=0.0,          # Longitude for central meridian
        global_extent=True,             # Whether to use a global extent
        figsize=(12, 6),                # Figure size
        cmap="balance",                 # Colormap
        vmin=None,                      # Min value for colormap
        vmax=None,                      # Max value for colormap
        levels=None,                    # Contour levels
        robust=False,                   # Whether to ignore extreme percentiles
        gridlines=True,                 # Toggle gridlines
        orientation="horizontal",       # Orientation of colorbar
        drawedges=False,                # Draw colorbar edges
        cbar_label="Example Units",     # Colorbar label
        states=True,                    # Draw internal state borders
        borders=True,                   # Draw country borders
        facecolor="lightgrey",          # Background color of the map
        edgecolor="face",               # Edge color for map patches
        bbox=None,                      # Tuple: (lon_min, lon_max, lat_min, lat_max)
        coastlines=True,                # Draw coastlines
        ocean=True,                     # Show ocean features
        land=True                       # Show land features
    )

# Alternatively, use method from xarray.DataArray directly:
>>> da.cartopy.plot(
        plot_type="default",
        projection="PlateCarree",
        central_longitude=0.0,
        global_extent=True,
        figsize=(12, 6),
        cmap="balance",
        vmin=None,
        vmax=None,
        levels=None,
        robust=False,
        gridlines=True,
        orientation="horizontal",
        drawedges=False,
        cbar_label="Example Units",
        states=True,
        borders=True,
        facecolor="lightgrey",
        edgecolor="face",
        bbox=None,
        coastlines=True,
        ocean=True,
        land=True
    )

Notes:
------
- `data` is required in `cartplot` and must be a 2D `xarray.DataArray` with latitude and longitude dimensions.
- All other arguments are optional and can be used for extensive customization.
- Designed to produce figures consistent with IPCC WG-style reports.
- Use the provided colormap methods (`adjust`, `blend`) to match your data scale and color symmetry needs.

"""

```

The ipcccutils library I am calling a “package” in quotes because it currently has the core structure of any package you would install using conda or pip; there is an __init__.py file that allows you to access all of the library’s modules and the functions within, using a single from pathlib import Path

from ipccutils import * command. However, this “package” is not available through conda or pip yet. In the meantime, you can get the package by cloning the repo.
