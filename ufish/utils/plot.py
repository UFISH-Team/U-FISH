import typing as T

import numpy as np
import scipy.spatial.distance
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure


class Plot2d(object):
    def __init__(self):
        self.default_figsize = (10, 10)
        self.default_marker_size = 20
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

    def evaluate_result(
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
            legend: bool = True,
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
            legend: whether to show legend
            **kwargs: keyword arguments passed to ax.scatter
        """
        from .metrics_deepblink import f1_at_cutoff
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
        ax.scatter(
            tp_coords[:, 1], tp_coords[:, 0],
            label="true positive",
            **kwargs_tp)
        kwargs_fp = kwargs.copy()
        kwargs_fp.setdefault("c", fp_color)
        kwargs_fp.setdefault("marker", fp_marker)
        ax.scatter(
            fp_coords[:, 1], fp_coords[:, 0],
            label="false positive",
            **kwargs_fp)
        kwargs_fn = kwargs.copy()
        kwargs_fn.setdefault("c", fn_color)
        kwargs_fn.setdefault("marker", fn_marker)
        ax.scatter(
            fn_coords[:, 1], fn_coords[:, 0],
            label="false negative",
            **kwargs_fn)
        # title
        if title_f1:
            ax.set_title(f"f1={f1_val:.3f} at cutoff {cutoff:.1f}")
        if legend:
            plt.legend()

    @classmethod
    def enhance_compare(
            cls,
            raw_img: np.ndarray,
            enhanced_img: np.ndarray,
            fig_size: T.Tuple[int, int] = (20, 10),
            ) -> "Figure":
        plt2d = cls()
        fig, axes = plt.subplots(1, 2, figsize=fig_size)
        plt2d.image(raw_img, ax=axes[0])
        plt2d.image(enhanced_img, ax=axes[1])
        return fig
