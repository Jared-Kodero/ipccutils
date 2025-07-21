from typing import Literal, Union

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


def get_cbar_axes(
    *,
    fig: plt.Figure = None,
    axes: plt.Axes = None,
    subplots=False,
    orientation: Literal["vertical", "horizontal"] = "vertical",
    pad=0.04,
) -> plt.Axes:
    """
    Create a new set of axes for a colorbar by stealing space from the current axes.
    This is useful for adding a colorbar to a plot without overlapping the existing axes.

    Parameters
    ----------
    fig : matplotlib.figure.Figure, optional
        The figure to which the colorbar axes will be added. If None, uses the current figure.
    ax : matplotlib.axes.Axes, optional
        The axes from which space will be stolen. If None, uses the current axes.
    pad : float, optional
        The padding between the colorbar axes and the existing axes. Default is 0.04. Try 0.04 and 0.05
    subplots : bool, optional
        If True, the function will adjust the colorbar position based on the subplots in the figure.
        This is useful when the figure has multiple subplots and you want to ensure the colorbar does not overlap with them.
        if True, the axes and fig must be provided and will be used to determine the position of the colorbar.

    orientation : str, optional
        The orientation of the colorbar. Can be either "vertical" or "horizontal". Default is "vertical".

    Returns
    -------
    matplotlib.axes.Axes
        The new axes for the colorbar.
    """

    if subplots and axes is None:
        raise ValueError("If subplots is True, axes and fig must be provided.")

    if fig is None:
        fig = plt.gcf()
    if axes is None:
        axes = plt.gca()

    plt.tight_layout()

    fig_width, fig_height = fig.get_size_inches()

    def _create_cax(y0, x0, y1, x1, x_len, y_len):
        if orientation == "vertical":

            bottommost = y0
            height = y_len
            rightmost = pad * height + x1
            norm = pad if fig_height < 5 else 0.05
            width = norm * height

            cax = fig.add_axes([rightmost, bottommost, width, height])

        elif orientation == "horizontal":

            rightmost = x0
            width = x_len
            bottommost = y0 - (0.12 * width)
            norm = pad if fig_width < 5 else 0.05
            height = norm * width

            cax = fig.add_axes([rightmost, bottommost, width, height])

        return cax

    if not subplots:
        pos = axes.get_position()
        fig_x_len = pos.x1 - pos.x0
        fig_y_len = pos.y1 - pos.y0
        cax = _create_cax(pos.y0, pos.x0, pos.y1, pos.x1, fig_x_len, fig_y_len)

    elif subplots:

        nrows, ncols = 1, 1

        if isinstance(axes, plt.Axes):
            nrows, ncols = 1, 1

        elif axes.ndim == 2:
            nrows, ncols = axes.shape
        elif axes.ndim == 1:
            # Need to ask figure
            last_ax = fig.axes[-1]
            nrows = last_ax.get_subplotspec().rowspan.stop
            ncols = last_ax.get_subplotspec().colspan.stop

        axes = np.reshape(axes, (nrows, ncols))
        right_axes = axes[:, -1]  # All rows, last column
        bottom_axes = axes[-1, :]  # Last row, all columns

        top_right_ax = right_axes[0].get_position()
        bot_right_ax = right_axes[-1].get_position()
        left_bot_ax = bottom_axes[0].get_position()
        right_bot_ax = bottom_axes[-1].get_position()

        fig_x_len = right_bot_ax.x1 - left_bot_ax.x0
        fig_y_len = top_right_ax.y1 - bot_right_ax.y0

        cax = _create_cax(
            bot_right_ax.y0,
            left_bot_ax.x0,
            top_right_ax.y1,
            right_bot_ax.x1,
            fig_x_len,
            fig_y_len,
        )

    return cax


def create_map_figure(
    *,
    projection: Literal[
        "PlateCarree",
        "Mercator",
        "Robinson",
        "Mollweide",
        "Orthographic",
        "LambertConformal",
        "AlbersEqualArea",
        "Stereographic",
        "NorthPolarStereo",
        "SouthPolarStereo",
    ] = "PlateCarree",
    figsize: tuple[float, float] = None,
    global_extent: bool = False,
    central_longitude: float = 0.0,
    bbox: tuple[float, float, float, float] = None,
    only_ocean: bool = False,
    only_land: bool = False,
    states: bool = True,
    borders: bool = True,
    facecolor: str = "grey",
    edgecolor: str = "face",
):
    """
    Create a Cartopy map figure using a specified map projection and extent.

    Parameters
    ----------
    projection : {"PlateCarree", "Mercator", "Robinson", "Mollweide", "Orthographic",
                  "LambertConformal", "AlbersEqualArea", "Stereographic",
                  "NorthPolarStereo", "SouthPolarStereo"}, default "PlateCarree"
        The Cartopy map projection to use. Selects from common Cartopy projections.

    figsize : tuple of float, optional
        Matplotlib figure size in inches as (width, height). If None, uses the default size.

    global_extent : bool, default False
        If True, sets the extent of the map to the full globe.

    central_longitude : float, default 0.0
        Central longitude of the projection. Used in projections where applicable.

    central_latitude : float, default 0.0
        Central latitude of the projection. Relevant for Orthographic and some regional projections.

    bbox : tuple of float, optional
        Bounding box for the map extent in the form (min_lon, min_lat, max_lon, max_lat).
        Ignored if `global_extent=True`.

    only_ocean : bool, default False
        If True, shades ocean areas with a default image and hides land.

    states : bool, default True
        If True, overlays U.S. state boundaries (visible in North America extent).

    borders : bool, default True
        If True, overlays international country borders.

    facecolor : str, default "grey"
        Fill color for continents (if `only_ocean=False`).

    edgecolor : str, default "face"
        Edge color for coastlines, borders, and other map features.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The created Matplotlib figure.

    ax : matplotlib.axes.Axes
        The Cartopy-aware map axes.
    """

    projections = {
        "PlateCarree": ccrs.PlateCarree,
        "Mercator": ccrs.Mercator,
        "Robinson": ccrs.Robinson,
        "Mollweide": ccrs.Mollweide,
        "Orthographic": ccrs.Orthographic,
        "LambertConformal": ccrs.LambertConformal,
        "AlbersEqualArea": ccrs.AlbersEqualArea,
        "Stereographic": ccrs.Stereographic,
        "NorthPolarStereo": ccrs.NorthPolarStereo,
        "SouthPolarStereo": ccrs.SouthPolarStereo,
    }

    crt_projection = projections.get(projection, ccrs.PlateCarree)
    crt_projection = crt_projection(central_longitude=central_longitude)

    fig, ax = plt.subplots(subplot_kw={"projection": crt_projection}, figsize=figsize)

    if global_extent:
        ax.set_global()

    ax.add_feature(cfeature.COASTLINE)

    if only_ocean:

        ax.add_feature(
            cfeature.NaturalEarthFeature(
                "physical",
                "land",
                "50m",
                edgecolor=edgecolor,
                facecolor=facecolor,
                alpha=0.5,
            )
        )

    if states:
        ax.add_feature(cfeature.STATES, linestyle="-", alpha=0.3)
    if borders:
        ax.add_feature(cfeature.BORDERS, linestyle="-", alpha=0.3)

    if bbox:
        ax.set_extent([bbox[0], bbox[2], bbox[1], bbox[3]], crs=ccrs.PlateCarree())

    return fig, ax


def cartplot(
    data: xr.DataArray,
    *,
    figsize: tuple[float, float] = None,
    map_type: Literal["pcolormesh", "contourf", "contour", "imshow"] = None,
    projection: Literal[
        "PlateCarree",
        "Mercator",
        "Robinson",
        "Mollweide",
        "Orthographic",
        "LambertConformal",
        "AlbersEqualArea",
        "Stereographic",
        "NorthPolarStereo",
        "SouthPolarStereo",
    ] = "PlateCarree",
    central_longitude: float = 0.0,
    cmap: Union[str, mcolors.Colormap] = None,
    vmin: float = None,
    vmax: float = None,
    levels: Union[int, list] = None,
    robust: bool = False,
    gridlines: bool = False,
    cbar_orientation: Literal["vertical", "horizontal"] = "vertical",
    cbar_label: str = None,
    global_extent: bool = False,
    bbox: tuple[float, float, float, float] = None,
    only_ocean: bool = False,
    only_land: bool = False,
    states: bool = True,
    borders: bool = True,
    facecolor: str = "grey",
    edgecolor: str = "face",
    return_plot: bool = False,
):
    """
    Plot an xarray DataArray on a Cartopy map using the specified projection and plot type.

    Parameters
    ----------
    data : xr.DataArray
        The 2D data to plot. Should contain spatial dimensions (e.g., lat/lon or x/y).

    figsize : tuple of float, optional
        Figure size in inches, as (width, height).

    map_type : {"pcolormesh", "contourf", "contour", "imshow"}, optional
        The type of plot to generate. If None, defaults to `DataArray.plot()` behavior.

    projection : str, default "PlateCarree"
        Cartopy CRS projection name. One of:
        {"PlateCarree", "Mercator", "Robinson", "Mollweide", "Orthographic",
         "LambertConformal", "AlbersEqualArea", "Stereographic",
         "NorthPolarStereo", "SouthPolarStereo"}.

    central_longitude : float, default 0.0
        Central longitude of the projection.

    central_latitude : float, default 0.0
        Central latitude for projections that require it (e.g., Orthographic).

    cmap : str or matplotlib.colors.Colormap, optional
        Colormap to use for the plot.

    vmin, vmax : float, optional
        Color limits for the plot. If not specified, determined automatically.

    levels : int or list of float, optional
        Number of contour levels or specific contour values. Used for contour and contourf plots.

    robust : bool, default False
        If True and `vmin`/`vmax` are not specified, uses the 2nd and 98th percentiles for color limits.

    cbar_orientation : {"vertical", "horizontal"}, default "vertical"
        Orientation of the colorbar.

    cbar_label : str, optional
        Label for the colorbar.

    global_extent : bool, default False
        If True, sets extent to show the entire globe.

    gridlines : bool, default False
        If True, adds gridlines to the map.

    bbox : tuple of float, optional
        Geographic extent to display in (min_lon, min_lat, max_lon, max_lat) format.
        Ignored if `global_extent=True`.

    only_ocean : bool, default False
        If True, shades only ocean areas and hides land using

    only_land : bool, default False
        If True, shades only land areas and hides ocean.

    states : bool, default True
        If True, overlays U.S. state boundaries (only visible in appropriate extents).

    borders : bool, default True
        If True, overlays international borders.

    facecolor : str, default "grey"
        Fill color for continents.

    edgecolor : str, default "face"
        Edge color for coastlines and borders.

    return_plot : bool, default False
        If True, returns the figure and axes objects along with the plot object.
        If False, only returns the figure and axes.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The Matplotlib Figure object.

    ax : matplotlib.axes._subplots.AxesSubplot
        The Cartopy-aware Axes with the plotted map.

    p : matplotlib.collections.QuadMesh or ContourSet
        The plotted data object.

    Notes
    -----
    This function requires `cartopy` and `matplotlib`. It is intended for 2D spatial
    `xarray.DataArray` objects.
    """

    # capture local kwargs
    kwargs = locals()
    # Define which kwargs go to map setup
    map_keys = {
        "projection",
        "figsize",
        "global_extent",
        "bbox",
        "only_ocean",
        "only_land",
        "states",
        "borders",
        "facecolor",
        "edgecolor",
        "central_longitude",
        "central_latitude",
    }

    if only_ocean and only_land:
        raise ValueError(
            "Cannot use both `only_ocean` and `only_land` at the same time."
        )

    map_kwargs = {k: v for k, v in kwargs.items() if k in map_keys}
    plot_kwargs = {k: v for k, v in kwargs.items() if k not in map_keys}

    # Setup figure and axis
    fig, ax = create_map_figure(**map_kwargs)

    plot_funcs = {
        "pcolormesh": data.plot.pcolormesh,
        "contourf": data.plot.contourf,
        "contour": data.plot.contour,
        "imshow": data.plot.imshow,
        None: data.plot,
    }

    map_type = plot_kwargs.pop("map_type", None)
    cbar_orientation = plot_kwargs.pop("cbar_orientation", "vertical")
    cbar_label = plot_kwargs.pop("cbar_label", None)
    return_plot = plot_kwargs.pop("return_plot", False)
    gridlines = plot_kwargs.pop("gridlines", False)

    if map_type not in plot_funcs:
        raise ValueError(
            f"Invalid map_type '{map_type}'. Choose from {list(plot_funcs)}."
        )

    plot_func = plot_funcs[map_type]

    # Plot the data
    p = plot_func(
        ax=ax, add_colorbar=False, transform=ccrs.PlateCarree(), **plot_kwargs
    )

    if only_ocean:
        ax.add_feature(cfeature.LAND, facecolor="white", zorder=1)
    elif only_land:
        ax.add_feature(cfeature.OCEAN, facecolor="white", zorder=1)

    if gridlines:
        gl = ax.gridlines(
            draw_labels=True, linewidth=0.5, color="gray", alpha=0.5, linestyle="--"
        )

        # Enable only left and bottom labels
        gl.top_labels = False
        gl.right_labels = False
        gl.bottom_labels = True
        gl.left_labels = True

    cax = get_cbar_axes(fig=fig, axes=ax, orientation=cbar_orientation)

    cb = plt.colorbar(
        p,
        cax=cax,
        ax=ax,
        orientation=cbar_orientation,
        drawedges=True,
    )

    if cbar_label:
        cb.set_label(cbar_label)

    res = (fig, ax) if not return_plot else (fig, ax, p)

    return res
