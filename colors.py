import difflib
import functools
import os
import sys
from typing import Literal, Union

import cmocean
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from cmocean import cm as cmo
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, to_hex

from .config import SCRIPT_DIR, eval_pkg_latex

LATEX_INSTALLED = eval_pkg_latex()


def set_plot_theme(
    *,
    font_scale: float = 1.5,
    line_width: float = 1.5,
    column_width: Literal["single", "double"] = None,
    latex: bool = False,
    context: Literal["paper", "notebook", "talk", "poster"] = "paper",
):
    """
    Configure the global matplotlib and seaborn theme for publication-quality plots.

    This function sets figure dimensions, font scaling, line widths, and optionally enables LaTeX rendering,
    ensuring that visualizations are formatted for either single- or double-column layouts typically required
    by scientific journals.

    Parameters
    ----------

    font_scale : float, optional
        Scaling factor for fonts. This is passed to `seaborn.set_theme()`. Default is 1.5.

    column_width : "single" or "double". Default is "single".
        Target layout width:
        - "single" corresponds to 9 cm (≈ 3.54 inches),
        - "double" corresponds to 18 cm (≈ 7.09 inches).
        Used to determine appropriate font and layout scaling. Default is "single". Overidden when `figsize` is set.

    line_width : float, optional
        Default line width for plot elements. Applied via `matplotlib.rcParams`. Default is 1.5.

    latex : bool, optional
        If True, enables LaTeX text rendering for all plot text via `matplotlib.rcParams["text.usetex"]`.
        Requires a working LaTeX installation. Default is False.
    context : str, optional
        Sets the context for the plot. Options are "paper", "notebook", "talk", or "poster". Default is "paper".
        This affects font sizes and other parameters to suit different presentation formats.

    """

    if column_width == "single":
        font_scale = 1
        line_width = 1
        fig_size = (3.5, 2.19)  # Single-column (9 cm × 5.56 cm)
    elif column_width == "double":
        fig_size = (7.09, 4.38)  # Double-column (18 cm × 11.12 cm)
    else:
        fig_size = None

    if "ipykernel" in sys.modules:
        import matplotlib_inline as plt_inline

        plt_inline.backend_inline.set_matplotlib_formats("retina")

    if latex and not LATEX_INSTALLED:
        latex = False

    rc = {
        "lines.linewidth": line_width,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.bottom": True,
        "ytick.left": True,
        "ytick.minor.visible": True,
        "xtick.minor.visible": True,
        "savefig.dpi": 1200,
        "text.usetex": latex,
    }

    if fig_size:
        rc["figure.figsize"] = fig_size
    if latex:
        rc["font.size"] = 12

    sns.set_theme(
        style="ticks",
        font="sans-serif",
        context=context,
        font_scale=font_scale,
        palette="colorblind",
        rc=rc,
    )


def spine_off(ax=None):

    ax = plt.gca() if ax is None else ax
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


class IPCCColorMaps:

    def __new__(
        cls,
    ):
        instance = super(IPCCColorMaps, cls).__new__(cls)
        instance.N = 256

        instance.colormaps = {}
        instance._load_colormaps()

        return instance

    SRC_DIR = SCRIPT_DIR / "src" / "data"

    def _load_colormaps(self):
        files = os.listdir(f"{self.SRC_DIR}")
        ipcc_cmap_list = [f.replace(".txt", "") for f in files]
        plt_cmap_list = plt.colormaps()
        cmocean_cmap_list = list(cmocean.cm.cmapnames)

        for cmap_name in ipcc_cmap_list:
            color_file = self.SRC_DIR / f"{cmap_name}.txt"
            colormap_data = np.loadtxt(color_file)
            cmap = LinearSegmentedColormap.from_list(cmap_name, colormap_data, N=self.N)
            self.colormaps[cmap_name] = cmap
            self.colormaps[f"{cmap_name}_r"] = cmap.reversed()

        for cmap_name in cmocean_cmap_list:
            cmap = getattr(cmo, cmap_name)
            self.colormaps[cmap_name] = cmap
            self.colormaps[f"{cmap_name}_r"] = cmap.reversed()

        for cmap_name in plt_cmap_list:
            if "cmo" in cmap_name:
                continue
            cmap = plt.colormaps[cmap_name]
            self.colormaps[cmap_name] = cmap

    @property
    def list_cmaps(self):
        for cmap_name in self.colormaps:

            fig, ax = plt.subplots(figsize=(5, 0.3))

            plt.imshow(
                np.linspace(0, 1, self.N).reshape(1, self.N),
                aspect="auto",
                cmap=self.colormaps[cmap_name],
            )

            fig.patch.set_visible(False)
            ax.axis("off")
            plt.title(cmap_name, loc="left", color="white", fontsize=8)

            plt.show()

    def _hint(self, name, hint=None):
        suggestions = difflib.get_close_matches(name, self.colormaps.keys(), n=5)
        if suggestions != []:
            hint = f"Did you mean one of {suggestions}?"
        else:
            hint = f"No close matches found for '{name}'."
        return hint

    def _validate_getitem(self, key):
        if isinstance(key, (list, tuple)):
            if all(
                isinstance(c, str) and (c.startswith("#") or c in mcolors.CSS4_COLORS)
                for c in key
            ):
                return BlendedColormap(key, self)
            else:
                raise ValueError(
                    "Invalid colors specified. Please provide a list of valid color names or hex values."
                )
        elif isinstance(key, str):
            if key in self.colormaps:
                return IPCCColorMapsManager(key, self.colormaps[key], self)
            else:
                raise KeyError(f"Colormap '{key}' is not available {self._hint(key)}")

        else:
            raise TypeError(
                "Invalid key type, please provide a valid colormap name or a list of colors in hex format."
            )

    def add_colors(
        self,
        obj,
        cmap,
        *,
        N=None,
        where: Literal["left", "middle", "right"] = "left",
        discrete=True,
        reverse: bool = False,
    ):

        if reverse:
            cmap = cmap.reversed()

        N = self.N if N is None else N
        odd_N = lambda x: x if x % 2 == 1 else x + 1
        N = odd_N(N) if where == 0 else N

        # Ensure `objs` is a list of strings
        if isinstance(obj, str):
            objs = [obj]
        elif isinstance(obj, (list, tuple)):
            objs = list(obj)
        else:
            raise TypeError(
                "Invalid colors specified. Please provide a list of valid color names or hex values."
            )

        colors_to_add = []
        for color in objs:
            if isinstance(color, str):
                if color.startswith("#"):
                    colors_to_add.append(color)
                elif mcolors.CSS4_COLORS.get(color) is not None:
                    colors_to_add.append(to_hex(mcolors.CSS4_COLORS[color]))
                else:
                    raise ValueError(
                        f"Invalid color '{color}'. Must be a hex code or a named CSS4 color."
                    )
            else:
                raise TypeError("Color must be a string (hex or named CSS4 color).")

        colors = [to_hex(tuple(c), keep_alpha=True) for c in cmap(np.linspace(0, 1, N))]

        if where == "left":
            new_colors = colors_to_add + colors

        elif where == "middle":
            half = len(colors) // 2
            new_colors = colors[:half] + colors_to_add + colors[half:]
        elif where == "right":
            new_colors = colors + colors_to_add
            discrete = False

        else:
            raise ValueError("`pos` must be 'left', 'middle', or 'right'.")

        new_colors = np.array(new_colors)

        N += len(colors_to_add)

        if discrete:
            cmap = ListedColormap(new_colors, N=N, name=cmap.name)
        else:
            cmap = LinearSegmentedColormap.from_list(cmap.name, new_colors, N=N)

        return cmap

    def adjust(
        self,
        cmap=None,
        N: int = 25,
        *,
        split: tuple[float, float] = (0, 1),
        add_colors: Union[str, list[str]] = None,
        where: Literal["left", "middle", "right"] = "left",
        reverse: bool = False,
        discrete: bool = True,
    ):
        """
        Access a colormap by name or object and apply optional adjustments.

        Parameters:
            cmap (str or matplotlib.colors.Colormap, optional):
                A string name of the colormap to retrieve from `self.colormaps`, or a matplotlib-compatible colormap object.
                If a string is provided, it will be looked up from `self.colormaps`.

            N (int, default=256):
                Number of colors in the colormap if `discrete=True`.

            split (tuple[float, float], default=(0.0, 1.0)):
                Fraction of the colormap to retain. Must be a pair of floats in [0.0, 1.0] representing the start and end positions.

            add_colors (str or list of str, optional):
                A color (e.g., "#FF0000", "red") or list of colors to add to the colormap. Only used if specified.

            where (str, optional):
                Location to insert `add_colors` into the colormap. Must be one of {"left", "middle", "right"}.
                Required if `add_colors` is specified.

            discrete (bool, default=True):
                Whether to convert the colormap into a discrete one with `N` bins.

            reverse (bool, default=False):
                Whether to reverse the colormap.

        Returns:
            matplotlib.colors.Colormap:
                The adjusted colormap object.
        """

        if isinstance(cmap, str):
            cmap = self.colormaps.get(cmap)
            if cmap is None:
                raise KeyError(f"Colormap '{cmap}' is not available {self._hint(cmap)}")

        if reverse:
            res = cmap.reversed()

        if not isinstance(split, tuple) or len(split) != 2:
            raise ValueError("`split` must be a tuple of two floats (start, end).")

        range_values = np.linspace(split[0], split[1], N)
        colors = [cmap(value) for value in range_values]

        if discrete:
            res = ListedColormap(colors, N=N, name=cmap.name)
        else:
            res = LinearSegmentedColormap.from_list(cmap.name, colors, N=N)

        if add_colors:
            if where == "right":
                discrete = False
            res = self.add_colors(
                add_colors,
                res,
                N=N,
                where=where,
                discrete=discrete,
            )

        return res

    def __iter__(self):
        return iter(self.colormaps)

    def __len__(self):
        return len(self.colormaps)

    def __dir__(self):
        return sorted(set(super().__dir__()) | set(self.colormaps.keys()))

    def __contains__(self, key):
        return key in self.colormaps

    def __getattr__(self, key):
        return self._validate_getitem(key)

    def __getitem__(self, key):
        return self._validate_getitem(key)


class IPCCColorMapsManager:
    def __init__(self, name: str, cmap, parent: "IPCCColorMaps"):
        self.name = name
        self.cmap = cmap
        self.parent = parent

    @functools.wraps(IPCCColorMaps.adjust)
    def adjust(
        self,
        N: int = 25,
        *,
        split: tuple[float, float] = (0, 1),
        add_colors: Union[str, list[str]] = None,
        where: Literal["left", "middle", "right"] = "left",
        reverse: bool = False,
        discrete: bool = True,
    ):

        self.cmap = self.parent.adjust(
            self.name,
            N=N,
            split=split,
            add_colors=add_colors,
            where=where,
            reverse=reverse,
            discrete=discrete,
        )

        return self.cmap

    def get_colors(
        self, *, reverse: bool = False, split: tuple[float, float] = (0, 1), N=None
    ):
        """
        Returns a list of colors in the colormap.
        Parameters
        ----------
            reverse (bool): If True, reverses the colormap.
            split (tuple): A tuple of two floats (start, end) to split the colormap.
            N (int): The number of colors in the colormap.

        Returns:
            list: A list of colors in the colormap.


        """

        if reverse:
            self.cmap = self.cmap.reversed()

        N = self.N if N is None else N

        if not isinstance(split, tuple) or len(split) != 2:
            raise ValueError("`split` must be a tuple of two floats (start, end).")

        return [to_hex(c) for c in self.cmap(np.linspace(split[0], split[1], self.N))]

    @functools.wraps(IPCCColorMaps.adjust)
    def __call__(
        self,
        N: int = 25,
        *,
        split: tuple[float, float] = (0, 1),
        add_colors: Union[str, list[str]] = None,
        where: Literal["left", "middle", "right"] = "left",
        reverse: bool = False,
        discrete: bool = True,
    ):

        return self.adjust(
            N=N,
            split=split,
            add_colors=add_colors,
            where=where,
            reverse=reverse,
            discrete=discrete,
        )

    def __getattr__(self, attr):
        return getattr(self.cmap, attr)

    def __repr__(self):
        return self.cmap.name

    def __str__(self):
        return self.cmap.name


class BlendedColormap:
    def __init__(self, colors, parent: "IPCCColorMaps"):
        self.colors = colors
        self._parent = parent

    def blend(self, N: int = 25, *, discrete: bool = True):
        """
        Generates a colormap by blending a list of colors.

        Parameters:

            N : int, default=256
            discrete : bool, default=True

        Returns:

            cmap: matplotlib.colors.Colormap


        Examples

            >>> cmap = ipcc_cmaps["#000000", "#ff0000", "#ffffff"].blend(N=256, discrete=True)

        """

        range_values = np.linspace(0, 1, N)

        cmap = sns.blend_palette(self.colors, as_cmap=True, input="hex", n_colors=N)
        color_list = [cmap(value) for value in range_values]
        if discrete:
            res = ListedColormap(color_list, N=N, name=f"Blend_{len(self.colors)}")
        else:
            res = LinearSegmentedColormap.from_list(
                f"Blend_{len(self.colors)}", color_list, N=N
            )

        return res


ipcc_cmap = IPCCColorMaps()
