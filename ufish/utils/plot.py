import typing as T

import numpy as np
import pandas as pd
import scipy.spatial.distance
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from .metrics import f1_at_cutoff


class Plot2d(object):
    def __init__(self):
        self.default_figsize = (10, 10)
        self.default_marker_size = 10
        self.default_marker_color = "red"
        self.default_marker_style = "x"
        self.default_imshow_cmap = "gray"

    def new_fig(self, **kwargs):
        """ Create a new figure and axes.

        Args:
            **kwargs: keyword arguments passed to plt.subplots
        """
        self.fig: Figure
        self.ax: Axes
        kwargs.setdefault("figsize", self.default_figsize)
        self.fig, self.ax = plt.subplots(**kwargs)

    def image(self, img, ax: T.Optional[Axes] = None, **kwargs):
        """ Plot an image.

        Args:
            img: image to plot, 2d array
            ax: axes to plot on
            **kwargs: keyword arguments passed to ax.imshow
        """
        if ax is None:
            ax = self.ax
        kwargs.setdefault("cmap", self.default_imshow_cmap)
        ax.imshow(img, **kwargs)

    def spots(
            self, coords: np.ndarray,
            ax: T.Optional[Axes] = None,
            **kwargs):
        """ Plot spots as scatter points.

        Args:
            coords: coordinates of spots, 2d array (n, 2)
            ax: axes to plot on
            **kwargs: keyword arguments passed to ax.scatter
        """
        if ax is None:
            ax = self.ax
        kwargs.setdefault("s", self.default_marker_size)
        kwargs.setdefault("c", self.default_marker_color)
        kwargs.setdefault("marker", self.default_marker_style)
        ax.scatter(coords[:, 1], coords[:, 0], **kwargs)

    def compare_result(
            self,
            pred: np.ndarray,
            true: np.ndarray,
            cutoff: float = 3.0,
            ax: T.Optional[Axes] = None,
            tp_color: str = "white",
            fp_color: str = "red",
            fn_color: str = "red",
            tp_marker: T.Optional[str] = None,
            fp_marker: T.Optional[str] = None,
            fn_marker: T.Optional[str] = None,
            title_f1: bool = True,
            **kwargs
            ):
        """ Plot true positives, false positives, and false negatives.

        Args:
            pred: predicted coordinates, 2d array (n, 2)
            true: true coordinates, 2d array (n, 2)
            cutoff: cutoff distance for matching
            ax: axes to plot on
            tp_color: color for true positives
            fp_color: color for false positives
            fn_color: color for false negatives
            tp_marker: marker style for true positives
            fp_marker: marker style for false positives
            fn_marker: marker style for false negatives
            title_f1: whether to show f1 score in title
            **kwargs: keyword arguments passed to ax.scatter
        """
        if ax is None:
            ax = self.ax
        tp_marker = tp_marker or self.default_marker_style
        fp_marker = fp_marker or self.default_marker_style
        fn_marker = fn_marker or self.default_marker_style
        # get f1 score and matching indices at given cutoff
        matrix = scipy.spatial.distance.cdist(pred, true, metric="euclidean")
        f1_val, true_pred_r, true_pred_c = f1_at_cutoff(
            matrix, pred, true, cutoff, return_raw=True)
        ind_pred_match = np.zeros(pred.shape[0], dtype=bool)
        ind_pred_match[true_pred_c] = True
        ind_true_match = np.zeros(true.shape[0], dtype=bool)
        ind_true_match[true_pred_r] = True
        tp_coords = pred[true_pred_c]
        fp_coords = pred[~ind_pred_match]
        fn_coords = true[~ind_true_match]
        # plot
        kwargs.setdefault("s", self.default_marker_size)
        kwargs_tp = kwargs.copy()
        kwargs_tp.setdefault("c", tp_color)
        kwargs_tp.setdefault("marker", tp_marker)
        ax.scatter(tp_coords[:, 1], tp_coords[:, 0], **kwargs_tp)
        kwargs_fp = kwargs.copy()
        kwargs_fp.setdefault("c", fp_color)
        kwargs_fp.setdefault("marker", fp_marker)
        ax.scatter(fp_coords[:, 1], fp_coords[:, 0], **kwargs_fp)
        kwargs_fn = kwargs.copy()
        kwargs_fn.setdefault("c", fn_color)
        kwargs_fn.setdefault("marker", fn_marker)
        ax.scatter(fn_coords[:, 1], fn_coords[:, 0], **kwargs_fn)
        # title
        if title_f1:
            ax.set_title(f"f1={f1_val:.3f} at {cutoff:.1f}")


def plot_result(
        img: np.ndarray,
        pred: pd.DataFrame,
        fig_size: T.Tuple[int, int] = (10, 10),
        image_cmap: str = 'gray',
        marker_size: int = 20,
        marker_color: str = 'red',
        marker_style: str = 'x',
        ) -> "Figure":
    plt2d = Plot2d()
    plt2d.default_figsize = fig_size
    plt2d.default_marker_size = marker_size
    plt2d.default_marker_color = marker_color
    plt2d.default_marker_style = marker_style
    plt2d.default_imshow_cmap = image_cmap
    plt2d.new_fig()
    plt2d.image(img)
    plt2d.spots(pred.values)
    return plt2d.fig


def plot_evaluate(
        img: np.ndarray,
        pred: pd.DataFrame,
        true: pd.DataFrame,
        cutoff: float = 3.0,
        fig_size: T.Tuple[int, int] = (10, 10),
        image_cmap: str = 'gray',
        marker_size: int = 20,
        tp_color: str = 'green',
        fp_color: str = 'red',
        fn_color: str = 'yellow',
        tp_marker: str = 'x',
        fp_marker: str = 'x',
        fn_marker: str = 'x',
        ) -> "Figure":
    plt2d = Plot2d()
    plt2d.default_figsize = fig_size
    plt2d.default_marker_size = marker_size
    plt2d.default_imshow_cmap = image_cmap
    plt2d.new_fig()
    plt2d.image(img)
    plt2d.compare_result(
        pred.values, true.values,
        cutoff=cutoff,
        tp_color=tp_color,
        fp_color=fp_color,
        fn_color=fn_color,
        tp_marker=tp_marker,
        fp_marker=fp_marker,
        fn_marker=fn_marker,
    )
    return plt2d.fig
