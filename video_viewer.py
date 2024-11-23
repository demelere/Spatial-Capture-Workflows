#!/usr/bin/env python3
"""
Example using a local .MOV video file

The .MOV video should contain both RGB and depth information in some way.
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Final

import cv2
import numpy as np
import numpy.typing as npt
import rerun as rr  # pip install rerun-sdk
import rerun.blueprint as rrb
import open3d as o3d
from tqdm import tqdm
import time

DESCRIPTION = """
# RGBD
Visualizes a .MOV video containing RGB and Depth channels.
"""

DEPTH_IMAGE_SCALING: Final = 1e4

VIDEO_FILE: Final = Path(os.path.dirname(__file__)) / "IMG_6628.MOV"  # make sure this file exists in the same directory as the script
PLY_FILE: Final = Path(os.path.dirname(__file__)) / "img-6628-scene-mesh.ply"  # Point cloud mesh file in same directory

def read_image_bgr(frame: np.ndarray) -> npt.NDArray[np.uint8]:
    """Process a BGR frame from the video as RGB data."""
    return frame

def read_depth_image(frame: np.ndarray) -> npt.NDArray[Any]:
    """Process a depth frame. This is a placeholder."""
    return frame  # assumes depth data is in a format compatible with this

def load_point_cloud(ply_path: Path) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.uint8]]:
    """Load point cloud from PLY file and return points and colors."""
    pcd = o3d.io.read_point_cloud(str(ply_path))
    points = np.asarray(pcd.points, dtype=np.float32)
    colors = np.asarray(pcd.colors, dtype=np.float32) * 255  # Convert from [0,1] to [0,255]
    return points, colors.astype(np.uint8)

def log_video_data(video_path: Path, ply_path: Path, frames: int) -> None:
    rr.init("rgbd_viewer", spawn=True)  # explicitly spawn a new viewer
    
    points, colors = load_point_cloud(ply_path) # load the point cloud  
    
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN, static=True)

    # load and log point cloud first as a static entity
    try:
        points, colors = load_point_cloud(ply_path)
        rr.log("world/point_cloud",
               rr.Points3D(
                   points,
                   colors=colors,
                   radii=0.01
               ),
               timeless=True) # make it persist across all frames
    except Exception as e:
        print(f"Error loading point cloud: {e}")
        return
    
    video_capture = cv2.VideoCapture(str(video_path)) # open the video file 
    if not video_capture.isOpened():
        print("Error: Couldn't open video file.")
        return
    
    try: 
        total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"Total frames in video: {total_frames}")

        if frames > total_frames:
            frames = total_frames

        for frame_idx in tqdm(range(frames), desc="Processing frames"):
            ret, frame = video_capture.read()
            if not ret:
                print(f"Error reading frame {frame_idx}.")
                break

            timestamp = datetime.utcnow()  # using UTC timestamp for frame
            rr.set_time_seconds("time", timestamp.timestamp())

            img_bgr = read_image_bgr(frame) # read the RGB frame
            rr.log("world/camera/image/rgb", rr.Image(img_bgr, color_model="BGR").compress(jpeg_quality=95)) # log the RGB frame

            img_depth = read_depth_image(frame) # read the depth frame
            rr.log(
                "world/camera/image",
                rr.Pinhole(
                    resolution=[img_depth.shape[1], img_depth.shape[0]],
                    focal_length=0.7 * img_depth.shape[1],
                    principal_point=[0.45 * img_depth.shape[1], 0.55 * img_depth.shape[0]],
                ),
            )

            # rr.log("world/camera/image/depth", rr.DepthImage(img_depth, meter=DEPTH_IMAGE_SCALING)) # TO DO: extract depth data from the video file  
            time.sleep(0.01) # add a small sleep to prevent overwhelming the viewer 

    finally:
        video_capture.release()

def main() -> None:
    parser = argparse.ArgumentParser(description="Example using a local .MOV video file.")
    parser.add_argument(
        "--frames", type=int, default=sys.maxsize, help="If specified, limits the number of frames logged"
    )
    rr.script_add_args(parser)
    args = parser.parse_args()

    rr.script_setup(
        args,
        "rerun_example_rgbd",
        default_blueprint=rrb.Horizontal(
            rrb.Vertical(
                rrb.Spatial3DView(name="3D", origin="world"),
                rrb.TextDocumentView(name="Description", origin="/description"),
                row_shares=[7, 3],
            ),
            rrb.Vertical(
                rrb.Spatial2DView(name="RGB & Depth", origin="world/camera/image", overrides={"world/camera/image/rgb": [rr.components.Opacity(0.5)]}),
                rrb.Tabs(
                    rrb.Spatial2DView(name="RGB", origin="world/camera/image", contents="world/camera/image/rgb"),
                    rrb.Spatial2DView(name="Depth", origin="world/camera/image", contents="world/camera/image/depth"),
                ),
                name="2D",
                row_shares=[3, 3, 2],
            ),
            column_shares=[2, 1],
        ),
    )

    rr.log("description", rr.TextDocument(DESCRIPTION, media_type=rr.MediaType.MARKDOWN), timeless=True)

    log_video_data(video_path=VIDEO_FILE, ply_path=PLY_FILE, frames=args.frames)

    rr.script_teardown(args)

if __name__ == "__main__":
    main()
