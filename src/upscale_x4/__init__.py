import os
from warnings import warn

import cv2
import imageio
from cv2 import dnn_superres


def is_jpg(filename: str) -> bool:
    return filename.lower().endswith((".jpg", ".jpeg"))


def is_gif(filename: str) -> bool:
    return filename.lower().endswith((".gif"))


def upscale_gif_x4(input_filename: str, side_check: int = 2000) -> str:
    """
    Upscales a gif by 4. Output file is saved in same directory as input.

    Parameters:
        input_filename (str): The input file to be upscaled
        side_check (int, optional): Throws an error if input file's height or width is greater than this. Default is 2000

    Returns:
        str: Output filename
    """

    # throw error if file is not gif
    if not is_gif(input_filename):
        raise Exception("Input file is not a gif")

    # load model
    model_path = os.path.join(os.path.dirname(__file__), "models", "FSRCNN_x4.pb")
    sr = dnn_superres.DnnSuperResImpl_create()
    sr.readModel(model_path)
    sr.setModel("fsrcnn", 4)
    output_filename = input_filename[:-4] + "_u.gif"

    # load gif
    cap = cv2.VideoCapture(input_filename)
    image_lst = []
    ret, frame = cap.read()

    # side check
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    if height > side_check or width > side_check:
        cap.release()
        raise Exception(
            f"{input_filename} is too large to upscale. (Side check set to {side_check})"
        )

    # upscale each frame
    while ret:
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        upscaled_image = sr.upsample(frame_rgb)
        image_lst.append(upscaled_image)

    # release gif and merge frames
    cap.release()
    imageio.mimwrite(output_filename, image_lst, fps=8, loop=0)
    print(f"Upscaled gif - {output_filename}")
    return output_filename


def upscale_image_x4(input_filename: str, side_check: int = 3000) -> str:
    """
    Upscales an image by 4. Output file is saved in same directory as input

    Parameters:
        input_filename (str): The input file to be upscaled
        side_check (int, optional): Throws an error if input file's height/width is greater than this. Default is 3000

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

    # side check
    if opencv_image.shape[0] > side_check or opencv_image.shape[1] > side_check:
        raise Exception(
            f"{input_filename} is too large to upscale. (Side check set to {side_check})"
        )

    # upscale image and return output filename
    upscaled_image = sr.upsample(opencv_image)
    cv2.imwrite(output_filename, upscaled_image)
    print(f"Upscaled image - {output_filename}")
    return output_filename
