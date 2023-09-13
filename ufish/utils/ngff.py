import zarr
import numpy as np
from ome_zarr.io import parse_url
from ome_zarr.writer import write_image


def create_ngff(
        data: np.ndarray, out_path: str,
        axes: str, chunks: tuple,
        ) -> zarr.Group:
    assert len(axes) == len(chunks), \
        "axes and chunks must have the same length"
    store = parse_url(out_path, mode="w").store
    root = zarr.group(store=store)
    write_image(
        image=data, group=root, axes="cyx",
        storage_options=dict(chunks=chunks),
    )
    return root
