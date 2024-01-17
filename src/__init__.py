import glob
import os
import sys
from warnings import warn

import cv2
import imageio
import numpy
from cv2 import dnn_superres


def is_jpg(filename: str) -> bool:
    return filename.lower().endswith((".jpg", ".jpeg"))


def upscale_image_x4(input_filename: str, height_check: int = 3000) -> str:
    """
    Upscales an image by 4. Returns the output filename (saved in same directory as input)

    Parameters:
        input_filename (str): The input file to be upscaled
        height_check (int, optional): Throws an error if input file's height is greater than this. Default is 3000

    Returns:
        str: Output filename
    """

    # throw warning if file is not jpg
    if not is_jpg(input_filename):
        warn(
            f"upscale_image_x4 - {input_filename} is not jpg, upscaler may give poor results or fail",
            stacklevel=3,
        )

    # load model
    model_path = os.path.join(os.path.dirname(__file__), "models", "FSRCNN_x4.pb")
    sr = dnn_superres.DnnSuperResImpl_create()
    sr.readModel(model_path)
    sr.setModel("fsrcnn", 4)
    _, file_extension = os.path.splitext(input_filename)
    output_filename = f"{input_filename[:-4]}_u{file_extension}"

    # read image
    opencv_image = cv2.imread(input_filename)

    # height check
    height = opencv_image.shape[0]
    if height > height_check:
        raise Exception(
            f"{input_filename} is too large to upscale. (Height check set to {height_check})"
        )

    # upscale image and return output filename
    upscaled_image = sr.upsample(opencv_image)
    cv2.imwrite(output_filename, upscaled_image)
    print(f"Upscaled image - {output_filename}")
    return output_filename
