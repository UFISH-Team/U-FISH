import typing as T
import random

import zarr
import numpy as np
from ome_zarr.io import parse_url
from ome_zarr.writer import write_image
from ome_zarr.reader import Reader
from dask.array import Array


def create_ngff(
        data: T.Union[np.ndarray, Array], out_path: str,
        axes: str, chunks: tuple,
        omero_info: T.Optional[dict] = None,
        ) -> zarr.Group:
    assert len(axes) == len(chunks), \
        "axes and chunks must have the same length"
    store = parse_url(out_path, mode="w").store
    root = zarr.group(store=store)
    write_image(
        image=data, group=root, axes="cyx",
        storage_options=dict(chunks=chunks),
    )
    if omero_info is not None:
        root.attrs["omero"] = omero_info
    return root


def random_hex_color():
    """Generate a random hex color code."""
    r = random.randint(0, 255)
    g = random.randint(0, 255)
    b = random.randint(0, 255)
    return '#%02X%02X%02X' % (r, g, b)


def generate_omero_info(
        data: T.Union[np.ndarray, Array],
        axes: str,
        channel_names: T.Optional[T.List[str]],
        channel_colors: T.Optional[T.List[str]],
        ) -> dict:
    channel_num = data.shape[axes.index("c")]
    if channel_names is None:
        channel_names = [
            f"channel_{i}" for i in range(channel_num)
        ]
    if channel_colors is None:
        channel_colors = [
            random_hex_color()[1:]
            for _ in range(channel_num)
        ]
    info = {
        "channels": [
            {
                "name": name,
                "color": color,
                "active": True,
            }
            for name, color in zip(channel_names, channel_colors)
        ]
    }
    return info


def read_ngff(
        path: str,
        resolution_level: int = 0,
        ) -> Array:
    reader = Reader(parse_url(path, mode="r"))
    first_node = next(reader())
    arrs = sorted(
        first_node.data, key=lambda x: x.size,
        reverse=True
    )
    arr = arrs[resolution_level]
    return arr
