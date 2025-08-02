import functools
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
    subplots: bool = False,
    orientation: Literal["vertical", "horizontal"] = "vertical",
    pad: float = 0.04,
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
    states: bool = True,
    borders: bool = True,
    facecolor: str = "grey",
    edgecolor: str = "face",
    bbox: tuple[float, float, float, float] = None,
    coastlines: bool = True,
    ocean: bool = True,
    land: bool = True,
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

    coastlines : bool, default True
        If True, adds coastlines to the map.

    ocean : bool, default False
        If True, shades ocean areas with a default image and hides land.

    land : bool, default True
        If True, shades land areas with a default image and hides ocean.

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

    if coastlines:
        ax.add_feature(cfeature.COASTLINE)

    if ocean and not land:

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


def plot_p_values(
    ax: plt.Axes,
    data: xr.DataArray,
    level: float = 0.05,
    color: str = "grey",
    alpha: float = 1,
    s: float = 1,
):
    """
    Plot p-values on a Cartopy axis.

    Parameters
    ----------
    ax : cartopy.mpl.geoaxes.GeoAxesSubplot`
        The Cartopy axis to plot on.
    data : xarray.DataArray
        The data array containing p-values.
    level : float, optional
        The significance level to use for plotting. Points with p-values below this level will be plotted
    color : str, optional
        Color of the points to plot. Default is "grey".
    alpha : float, optional
        Alpha transparency of the points. Default is 0.05.
    s : float, optional
        Size of the points to plot. Default is 1.
    """

    if "lon" not in data.dims or "lat" not in data.dims:
        raise ValueError("DataArray must contain 'lon' and 'lat' dimensions.")

    # p_values = xr.where(data > level, 1, np.nan)

    p_values = data.to_dataframe(name="p_values").reset_index()
    p_values = p_values.query("p_values < @level")

    # replace where p_values < 1with NaN pandas

    p_values = p_values.dropna()

    ax.scatter(
        p_values["lon"],
        p_values["lat"],
        transform=ccrs.PlateCarree(),
        color=color,
        alpha=alpha,
        s=s,
    )

    return ax


def cartplot(
    data: xr.DataArray,
    *,
    plot_type: Literal[
        "default", "pcolormesh", "contourf", "contour", "imshow"
    ] = "default",
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
    global_extent: bool = False,
    figsize: tuple[float, float] = None,
    cmap: Union[str, mcolors.Colormap] = None,
    vmin: float = None,
    vmax: float = None,
    levels: Union[int, list] = None,
    robust: bool = False,
    gridlines: bool = False,
    add_colorbar: bool = True,
    orientation: Literal["vertical", "horizontal"] = "vertical",
    drawedges: bool = False,
    cbar_label: str = None,
    states: bool = True,
    borders: bool = True,
    facecolor: str = "#d3d3d3",
    edgecolor: str = "face",
    bbox: tuple[float, float, float, float] = None,
    coastlines: bool = True,
    ocean: bool = True,
    land: bool = True,
):
    """
    Plot an xarray DataArray on a Cartopy map using the specified projection and plot type.

    Parameters
    ----------
    data : xr.DataArray
        The 2D data to plot. Should contain spatial dimensions (e.g., lat/lon or x/y).

    map_type : {"pcolormesh", "contourf", "contour", "imshow"}, optional
        The type of plot to generate. If None, defaults to `DataArray.plot()` behavior.

    projection : {"PlateCarree", "Mercator", "Robinson", "Mollweide", "Orthographic",
                  "LambertConformal", "AlbersEqualArea", "Stereographic",
                  "NorthPolarStereo", "SouthPolarStereo"}, default: "PlateCarree"
        The Cartopy CRS projection to use for the map.

    central_longitude : float, default: 0.0
        Central longitude of the projection.

    global_extent : bool, default: False
        If True, sets extent to show the entire globe. Otherwise, uses `bbox` if given.

    figsize : tuple of float, optional
        Figure size in inches, as (width, height).

    cmap : str or matplotlib.colors.Colormap, optional
        Colormap to use for the plot.

    vmin : float, optional
        Minimum data value for color scaling.

    vmax : float, optional
        Maximum data value for color scaling.

    levels : int or list of float, optional
        Number of contour levels (int) or specific contour values (list).
        Used for contour and contourf plots.

    robust : bool, default: False
        If True and `vmin`/`vmax` are not specified, uses the 2nd and 98th percentiles
        for color limits to reduce the impact of outliers.

    gridlines : bool, default: False
        If True, overlays latitude/longitude gridlines.

    orientation : {"vertical", "horizontal"}, default: "vertical"
        Orientation of the colorbar.

    draw_cbar_edges : bool, default: True
        If True, draws edges on the colorbar for better visibility.

    cbar_label : str, optional
        Label for the colorbar.

    states : bool, default: True
        If True, overlays U.S. state boundaries (only visible in appropriate extents).

    borders : bool, default: True
        If True, overlays international borders.

    facecolor : str, default: "grey"
        Fill color for landmasses or continents.

    edgecolor : str, default: "face"
        Edge color for coastlines and borders.

    bbox : tuple of float, optional
        Geographic extent to display in (min_lon, min_lat, max_lon, max_lat) format.
        Ignored if `global_extent=True`.

    coastlines : bool, default: True
        If True, adds coastlines to the map.

    ocean : bool, default: True
        If True, displays ocean features.

    land : bool, default: True
        If True, displays land features.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The Matplotlib Figure object.

    ax : cartopy.mpl.geoaxes.GeoAxesSubplot
        The Cartopy-aware Axes with the plotted map.

    p : matplotlib.collections.QuadMesh or matplotlib.contour.ContourSet
        The plotted data object.

    Notes
    -----
    - This function requires `cartopy` and `matplotlib`.
    - It is intended for 2D spatial `xarray.DataArray` objects with geographic dimensions.
    - Some features (e.g., borders and state lines) require Natural Earth shapefiles
      which `cartopy` can download automatically.
    """

    allargs = locals()
    map_keys = (
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
    )

    map_kwargs = {k: v for k, v in allargs.items() if k in map_keys}
    plot_kwargs = {k: v for k, v in allargs.items() if k not in map_keys}

    plot_type = plot_kwargs.pop("plot_type", "default")
    orientation = plot_kwargs.pop("orientation", "vertical")
    cbar_label = plot_kwargs.pop("cbar_label", None)
    gridlines = plot_kwargs.pop("gridlines", False)
    drawedges = plot_kwargs.pop("drawedges", True)
    ocean = plot_kwargs.pop("ocean", True)
    land = plot_kwargs.pop("land", True)
    coastlines = plot_kwargs.pop("coastlines", True)
    add_colorbar = plot_kwargs.pop("add_colorbar", True)

    fig, ax = create_map_figure(**map_kwargs)

    plot_funcs = {
        "default": data.plot,
        "pcolormesh": data.plot.pcolormesh,
        "contourf": data.plot.contourf,
        "contour": data.plot.contour,
        "imshow": data.plot.imshow,
    }

    if plot_type not in plot_funcs:
        raise ValueError(
            f"Invalid plot_type '{plot_type}'. Choose from {list(plot_funcs)}."
        )

    plot_func = plot_funcs[plot_type]

    # Plot the data
    p = plot_func(
        ax=ax, add_colorbar=False, transform=ccrs.PlateCarree(), **plot_kwargs
    )

    if ocean and not land:
        ax.add_feature(cfeature.LAND, facecolor="white", zorder=1)
    elif land and not ocean:
        ax.add_feature(cfeature.OCEAN, facecolor="white", zorder=1)

    if gridlines:
        gl = ax.gridlines(
            draw_labels=True, linewidth=0.5, color="gray", alpha=0.5, linestyle="--"
        )

        gl.top_labels = False
        gl.right_labels = False
        gl.bottom_labels = True
        gl.left_labels = True

    if add_colorbar:
        cax = get_cbar_axes(fig=fig, axes=ax, orientation=orientation)

        cb = plt.colorbar(
            p,
            cax=cax,
            ax=ax,
            orientation=orientation,
            drawedges=drawedges,
        )

        if cbar_label:
            cb.set_label(cbar_label)

    return (fig, ax, p)


@xr.register_dataarray_accessor("cartopy")
class CartPlotAccessor:
    def __init__(self, xarray_obj):
        self._obj = xarray_obj

    @functools.wraps(cartplot)
    def plot(self, *args, **kwargs):
        return cartplot(self._obj, *args, **kwargs)
