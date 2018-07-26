
"""
convert panorama image into cube map

dependencies:
 - Pillow
 - numpy
 - imageio
 - FreeImage extension for hdr handing

python version
 - tested on 3.7

author: minu jeong
"""

import os
import shutil
import sys
import math
from io import BytesIO

from PIL import Image
import imageio
import numpy as np


def sample(pixels, uv):
    """
    samples pixel from uv.
    captures 2 x 2 pixels, returns average

    @param pixels: numpy array read from imageio
    @param uv: target uv coordinate (2d)
    """
    size = pixels.shape
    ax, ay = uv[0] * size[0], uv[1] * size[1]
    x1, x2 = int(math.floor(ax)), int(math.ceil(ax))
    y1, y2 = int(math.floor(ay)), int(math.ceil(ay))
    x2 = min(x2, size[0] - 1)
    y2 = min(y2, size[1] - 1)
    a = pixels[x1, y1]
    b = pixels[x1, y2]
    c = pixels[x2, y1]
    d = pixels[x2, y2]
    return (
        (a[0] + b[0] + c[0] + d[0]) / 4,
        (a[1] + b[1] + c[1] + d[1]) / 4,
        (a[2] + b[2] + c[2] + d[2]) / 4
    )


def normalize(v):
    """
    turn vector into unit vector

    @param v: source vector
    """
    x, y, z = v
    l = math.sqrt(x * x + y * y + z * z)
    return (x / l, y / l, z / l)


def normal_to_uv(normal)->tuple:
    """
    right-handed system

    @param normal: iterable of normal vector
    """
    phi = math.acos(-normal[1])
    theta = math.atan2(-normal[0], normal[2])
    u = math.fmod(theta / (math.pi * 2.0), 1.0)
    v = math.fmod(phi / math.pi, 1.0)
    if u < 0:
        u += 1.0
    if v < 0:
        v += 1.0
    return (u, v)


def build_images(envmap, resolution):
    """
    build cubemap faces

    @param envmap: numpy array of pixel data read from imageio
    @param resolution: sampling resolution
    """
    top_pixels = np.zeros(shape=(resolution, resolution, 3), dtype=np.float32)
    top_x, top_y, top_z = -0.5, 0.5, -0.5

    left_pixels = np.zeros(shape=(resolution, resolution, 3), dtype=np.float32)
    left_x, left_y, left_z = -0.5, -0.5, -0.5

    right_pixels = np.zeros(shape=(resolution, resolution, 3), dtype=np.float32)
    right_x, right_y, right_z = 0.5, -0.5, -0.5

    bottom_pixels = np.zeros(shape=(resolution, resolution, 3), dtype=np.float32)
    bottom_x, bottom_y, bottom_z = -0.5, -0.5, -0.5

    front_pixels = np.zeros(shape=(resolution, resolution, 3), dtype=np.float32)
    front_x, front_y, front_z = -0.5, -0.5, -0.5

    back_pixels = np.zeros(shape=(resolution, resolution, 3), dtype=np.float32)
    back_x, back_y, back_z = -0.5, -0.5, 0.5

    # only used for debugging
    dw, dh = (envmap.shape[1] * 0.25, envmap.shape[0] * 0.25)
    dw, dh = int(dw), int(dh)
    debug_img = Image.new("RGB", (dw, dh))
    px_debug = debug_img.load()

    for px in range(resolution):
        u = px / resolution
        for py in range(resolution):
            v = py / resolution
            top_uv = normal_to_uv(normalize((top_x + u, top_y, top_z + v)))
            top_pixels[px, py] = sample(envmap, top_uv)
            left_uv = normal_to_uv(normalize((left_x, left_y + v, left_z + u)))
            left_pixels[px, py] = sample(envmap, left_uv)
            right_uv = normal_to_uv(normalize((right_x, right_y + v, right_z + u)))
            right_pixels[px, py] = sample(envmap, right_uv)
            front_uv = normal_to_uv(normalize((front_x + u, front_y + v, front_z)))
            front_pixels[px, py] = sample(envmap, front_uv)
            bottom_uv = normal_to_uv(normalize((bottom_x + u, bottom_y, bottom_z + v)))
            bottom_pixels[px, py] = sample(envmap, bottom_uv)
            back_uv = normal_to_uv(normalize((back_x + u, back_y + v, back_z)))
            back_pixels[px, py] = sample(envmap, back_uv)

            # draw debugging
            px_debug[int(top_uv[0] * dw), int(top_uv[1] * dh)] = (32, int(255 * u), int(255 * v))
            px_debug[int(left_uv[0] * dw), int(left_uv[1] * dh)] = (int(255 * u), int(255 * v), 128)
            px_debug[int(right_uv[0] * dw), int(right_uv[1] * dh)] = (int(255 * u), 255, int(255 * v))
            px_debug[int(front_uv[0] * dw), int(front_uv[1] * dh)] = (int(255 * u), 128, int(255 * v))
            px_debug[int(bottom_uv[0] * dw), int(bottom_uv[1] * dh)] = (int(255 * v), int(255 * u), 56)
            px_debug[int(back_uv[0] * dw), int(back_uv[1] * dh)] = (int(225 * v), int(111 * u), int(24 * u))

    debug_img.save("debug_image.png")

    yield "top", top_pixels
    yield "left", left_pixels
    yield "right", right_pixels
    yield "bottom", bottom_pixels
    yield "front", front_pixels
    yield "back", back_pixels


def main(target_path, resolution):
    """
    create output directory, trigger builder, write result files

    @param target_path: path to HDRI filename
    @param resolution: sample resolution that will pass to builder
    """

    outdir = "out"
    if not os.path.isdir(outdir):
        os.makedirs(outdir)

    envmap = imageio.imread(target_path)
    for dname, gen_img in build_images(envmap, resolution):
        imageio.imwrite(f"{outdir}/{dname}.hdr", gen_img)

        # save preview images
        imageio.imwrite(f"{outdir}/{dname}.png", np.multiply(gen_img, 1024).astype(np.uint8))


if __name__ == "__main__":
    target_path = None
    resolution = 64
    if len(sys.argv) > 2:
        target_path = sys.argv[1]
    else:
        target_path = "test.hdr"

    if len(sys.argv) > 3:
        resolution = sys.argv[2]

    main(target_path, resolution)
