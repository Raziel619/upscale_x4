import os
from warnings import warn

import cv2
import imageio
from cv2 import dnn_superres

X2_SIDE_CHECK = 5500
X4_SIDE_CHECK = 3000
X8_SIDE_CHECK = 1500


def is_jpg(filename: str) -> bool:
    return filename.lower().endswith((".jpg", ".jpeg"))


def is_gif(filename: str) -> bool:
    return filename.lower().endswith((".gif"))


# region GIF Methods


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


# endregion

# region Image Methods


def upscale_image_x2(input_filename: str, side_check: int = X2_SIDE_CHECK) -> str:
    """
    Upscales an image by 2. Output file is saved in same directory as input

    Parameters:
        input_filename (str): The input file to be upscaled
        side_check (int, optional): Throws an error if input file's height/width is greater than this. Default is 5000

    Returns:
        str: Output filename
    """

    # throw warning if file is not jpg
    if not is_jpg(input_filename):
        warn(
            f"upscale_image_x2 - {input_filename} is not jpg, upscaler may give poor results or fail",
            stacklevel=3,
        )

    # read image
    opencv_image = cv2.imread(input_filename)

    # side check
    if opencv_image.shape[0] > side_check or opencv_image.shape[1] > side_check:
        raise Exception(
            f"{input_filename} is too large to upscale. (Side check set to {side_check})"
        )

    # load model
    model_path = os.path.join(os.path.dirname(__file__), "models", "FSRCNN_x2.pb")
    sr = dnn_superres.DnnSuperResImpl_create()
    sr.readModel(model_path)
    sr.setModel("fsrcnn", 2)
    _, file_extension = os.path.splitext(input_filename)
    output_filename = f"{input_filename[:-4]}_u{file_extension}"

    # upscale image and return output filename
    upscaled_image = sr.upsample(opencv_image)
    cv2.imwrite(output_filename, upscaled_image)
    print(f"Upscaled image - {output_filename}")
    return output_filename


def upscale_image_x4(input_filename: str, side_check: int = X4_SIDE_CHECK) -> str:
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

    # read image
    opencv_image = cv2.imread(input_filename)

    # side check
    if opencv_image.shape[0] > side_check or opencv_image.shape[1] > side_check:
        raise Exception(
            f"{input_filename} is too large to upscale. (Side check set to {side_check})"
        )

    # load model
    model_path = os.path.join(os.path.dirname(__file__), "models", "FSRCNN_x4.pb")
    sr = dnn_superres.DnnSuperResImpl_create()
    sr.readModel(model_path)
    sr.setModel("fsrcnn", 4)
    _, file_extension = os.path.splitext(input_filename)
    output_filename = f"{input_filename[:-4]}_u{file_extension}"

    # upscale image and return output filename
    upscaled_image = sr.upsample(opencv_image)
    cv2.imwrite(output_filename, upscaled_image)
    print(f"Upscaled image - {output_filename}")
    return output_filename


def upscale_image_x8(input_filename: str, side_check: int = X8_SIDE_CHECK) -> str:
    """
    Upscales an image by 8 (using 2x followed by 4x). Output file is saved in same directory as input

    Parameters:
        input_filename (str): The input file to be upscaled
        side_check (int, optional): Throws an error if input file's height/width is greater than this. Default is 1500

    Returns:
        str: Output filename
    """

    # throw warning if file is not jpg
    if not is_jpg(input_filename):
        warn(
            f"upscale_image_x8 - {input_filename} is not jpg, upscaler may give poor results or fail",
            stacklevel=3,
        )

    # read image
    opencv_image = cv2.imread(input_filename)

    # side check
    if opencv_image.shape[0] > side_check or opencv_image.shape[1] > side_check:
        raise Exception(
            f"{input_filename} is too large to upscale. (Side check set to {side_check})"
        )

    # Load the x2 model
    model_path_x2 = os.path.join(os.path.dirname(__file__), "models", "FSRCNN_x2.pb")
    sr_x2 = dnn_superres.DnnSuperResImpl_create()
    sr_x2.readModel(model_path_x2)
    sr_x2.setModel("fsrcnn", 2)

    # Load the x4 model
    model_path_x4 = os.path.join(os.path.dirname(__file__), "models", "FSRCNN_x4.pb")
    sr_x4 = dnn_superres.DnnSuperResImpl_create()
    sr_x4.readModel(model_path_x4)
    sr_x4.setModel("fsrcnn", 4)

    # Output filename
    _, file_extension = os.path.splitext(input_filename)
    output_filename = f"{input_filename[:-4]}_u{file_extension}"

    # First upscale by 2x
    upscaled_image_x2 = sr_x2.upsample(opencv_image)

    # Then upscale the result by 4x (2x4 = 8x total)
    upscaled_image_x8 = sr_x4.upsample(upscaled_image_x2)

    # Same image and return filename
    cv2.imwrite(output_filename, upscaled_image_x8)
    print(f"Upscaled image - {output_filename}")
    return output_filename


def upscale_image_to_8k(input_filename: str):
    """
    Upscales an image by 8k by determining what scale factor to use. Output file is saved in same directory as input

    Parameters:
        input_filename (str): The input file to be upscaled

    Returns:
        str: Output filename
    """

    # Determine shorter image side
    image = cv2.imread(input_filename)
    larger_side = image.shape[0] if image.shape[0] > image.shape[1] else image.shape[1]
    del image

    if larger_side < X8_SIDE_CHECK:
        return upscale_image_x8(input_filename)
    elif larger_side < X4_SIDE_CHECK:
        return upscale_gif_x4(input_filename)
    elif larger_side < X2_SIDE_CHECK:
        return upscale_image_x2(input_filename)
    else:
        raise Exception(f"{input_filename} is too large to upscale.")


# endregion
