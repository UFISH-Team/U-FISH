
"""3D Grid search script for trackmate's FIJI API."""

import csv
import glob
import os

from ij import IJ
from fiji.plugin.trackmate import Model
from fiji.plugin.trackmate import Settings
from fiji.plugin.trackmate import TrackMate
from fiji.plugin.trackmate import Logger
from fiji.plugin.trackmate.detection import LogDetectorFactory
from fiji.plugin.trackmate.detection import DogDetectorFactory


def run_trackmate(
    imp, path_out="./", detector="dog", radius=2.5, threshold=0.0, median_filter=True
):
    """Log Trackmate detection run with given parameters.
    Saves spots in a csv file in the given path_out with encoded parameters.

    Args:
        imp: ImagePlus to be processed
        path_out: Output directory to save files.
        detector: Type of detection method. Options are 'log', 'dog'.
        radius: Radius of spots in pixels.
        threshold: Threshold value to filter spots.
        median_filter: True if median_filtering should be used.
    """

    ''' if imp.dimensions[2] != 1:
        raise ValueError(
            "Imp's dimensions must be [n, n, 1] but are " + imp.dimensions[2]
        )
    '''
   
    # Create the model object now
    model = Model()
    model.setLogger(Logger.VOID_LOGGER)

    # Prepare settings object
    settings = Settings(imp)

    # Configure detector
    settings.detectorFactory = (
        DogDetectorFactory() if detector == "dog" else LogDetectorFactory()
    )
    settings.detectorSettings = {
        "DO_SUBPIXEL_LOCALIZATION": True,
        "RADIUS": radius,
        "TARGET_CHANNEL": 1,
        "THRESHOLD": threshold,
        "DO_MEDIAN_FILTERING": median_filter,
    }
    trackmate = TrackMate(model, settings)


    # Process
    # output = trackmate.process()
    output = trackmate.execDetection()
    if not output:
        print("error process")
        return None

    # Get output from a single image
    fname = str(imp.title)
    spots = [["axis-0", "axis-1", "axis-2"]]
    for spot in model.spots.iterator(0):
        x = spot.getFeature("POSITION_X")
        y = spot.getFeature("POSITION_Y")
        z = spot.getFeature("POSITION_Z")
        #q = spot.getFeature("QUALITY")
        spots.append([z, y, x])

    # Save output
    outname = "TM_" + str(os.path.basename(fname)) + "_r" + str(radius) + "_thr" + str(threshold) + ".csv"
    with open(os.path.join(path_out, outname), "w") as f:
        wr = csv.writer(f)
        for row in spots:
            wr.writerow(row)


def process_im(outdir, fname):
    """One "grid search" loop in trackmate."""

    imp = IJ.openImage(fname)

    opts_radius = [0.01, 0.1, 0.2, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0, 7.5, 10.0]
    thr_range = []
    thr = 0.1
    while thr <= 15:
        thr_range.append(thr)
        thr *= 1.05

    for r in opts_radius:
        for thr in thr_range:
            run_trackmate(imp, path_out=outdir, radius=r, threshold=thr)

    imp.close()

indir = "ufish/data_3d/train/image-uint8/"
outdir = "ufish/benchmarks/3d/TM/results/"

files = sorted(glob.glob(os.path.join(indir, "*.tif")))

for file in files:
    print("Running: " + file)
    process_im(outdir, file)


