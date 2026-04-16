# Copyright (c) OpenMMLab. All rights reserved.
import os
import os.path as osp
from collections import OrderedDict

import cv2
from cv2 import (CAP_PROP_FOURCC, CAP_PROP_FPS, CAP_PROP_FRAME_COUNT,
                 CAP_PROP_FRAME_HEIGHT, CAP_PROP_FRAME_WIDTH,
                 CAP_PROP_POS_FRAMES, VideoWriter_fourcc)

from mmcv.utils import (check_file_exist, mkdir_or_exist, scandir,
                        track_progress)
from unhcv.common.utils import walk_all_files_with_suffix

def frames2video(frame_dir: str,
                 video_file: str,
                 fps: float = 30,
                 fourcc: str = 'XVID',
                 filename_tmpl: str = '{:06d}.jpg',
                 start: int = 0,
                 end: int = None,
                 show_progress: bool = True) -> None:
    """Read the frame images from a directory and join them as a video.

    Args:
        frame_dir (str): The directory containing video frames.
        video_file (str): Output filename.
        fps (float): FPS of the output video.
        fourcc (str): Fourcc of the output video, this should be compatible
            with the output file type.
        filename_tmpl (str): Filename template with the index as the variable.
        start (int): Starting frame index.
        end (int): Ending frame index.
        show_progress (bool): Whether to show a progress bar.
    """
    img_names = walk_all_files_with_suffix(frame_dir)
    first_file = img_names[0]
    img = cv2.imread(first_file)
    height, width = img.shape[:2]
    resolution = (width, height)
    os.makedirs(osp.dirname(video_file), exist_ok=True)
    vwriter = cv2.VideoWriter(video_file, VideoWriter_fourcc(*fourcc), fps,
                              resolution)

    def write_frame(file_idx):
        filename = img_names[file_idx]
        img = cv2.imread(filename)
        vwriter.write(img)
    if end is None:
        end = len(img_names)
    if show_progress:
        track_progress(write_frame, range(start, end))
    else:
        for i in range(start, end):
            write_frame(i)
    vwriter.release()
