
"""
convert panorama image into cube map

dependencies:
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

import imageio
import numpy as np


def sample(pixels, uv):
    size = pixels.shape
    ax, ay = uv[0] * size[0], uv[1] * size[1]
    x1, x2 = int(math.floor(ax)), int(math.ceil(ax))
    y1, y2 = int(math.floor(ay)), int(math.ceil(ay))
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
    x, y, z = v
    l = math.sqrt(x * x + y * y + z * z)
    return (x / l, y / l, z / l)


def normal_to_uv(normal)->tuple:
    """ right-handed system """
    phi = math.acos(-normal[1])
    theta = math.atan2(normal[0], normal[2])
    u = math.fmod(theta / (math.pi * 2.0), 1.0)
    v = math.fmod(phi / math.pi, 1.0)
    if u < 0:
        u += 1.0
    if v < 0:
        v += 1.0
    return (u, v)


def build_images(envmap, resolution):
    top_pixels = np.zeros(shape=(*resolution, 3), dtype=np.float32)
    top_x, top_y, top_z = -0.5, 0.5, -0.5

    left_pixels = np.zeros(shape=(*resolution, 3), dtype=np.float32)
    left_x, left_y, left_z = -0.5, -0.5, -0.5

    right_pixels = np.zeros(shape=(*resolution, 3), dtype=np.float32)
    right_x, right_y, right_z = 0.5, -0.5, -0.5

    bottom_pixels = np.zeros(shape=(*resolution, 3), dtype=np.float32)
    bottom_x, bottom_y, bottom_z = -0.5, -0.5, -0.5

    front_pixels = np.zeros(shape=(*resolution, 3), dtype=np.float32)
    front_x, front_y, front_z = -0.5, -0.5, -0.5

    back_pixels = np.zeros(shape=(*resolution, 3), dtype=np.float32)
    back_x, back_y, back_z = -0.5, -0.5, 0.5
    
    for px in range(resolution[0]):
        u = px / resolution[0]
        for py in range(resolution[1]):
            v = py / resolution[1]
            top_uv = normal_to_uv(normalize((top_x + u, top_y, top_z + v)))
            left_uv = normal_to_uv(normalize((left_x, left_y + v, left_z + u)))
            right_uv = normal_to_uv(normalize((right_x, right_y + v, right_z - u)))
            bottom_uv = normal_to_uv(normalize((bottom_x + u, bottom_y, bottom_z + v)))
            front_uv = normal_to_uv(normalize((front_x + u, front_y + v, front_z)))
            back_uv = normal_to_uv(normalize((back_x + u, back_y + v, back_z)))

            top_pixels[px, py] = sample(envmap, top_uv)
            left_pixels[px, py] = sample(envmap, left_uv)
            right_pixels[px, py] = sample(envmap, right_uv)
            bottom_pixels[px, py] = sample(envmap, bottom_uv)
            front_pixels[px, py] = sample(envmap, front_uv)
            back_pixels[px, py] = sample(envmap, back_uv)

    yield "top", top_pixels
    yield "left", left_pixels
    yield "right", right_pixels
    yield "bottom", bottom_pixels
    yield "front", front_pixels
    yield "back", back_pixels


def main(target_path):
    """ target path: path to HDRI filename """

    outdir = "out"
    if os.path.isdir(outdir):
        shutil.rmtree(outdir)
    os.makedirs(outdir)

    envmap = imageio.imread(target_path)
    for dname, gen_img in build_images(envmap, (256, 256)):
        imageio.imwrite(f"{outdir}/{dname}.hdr", gen_img)


if __name__ == "__main__":
    target_path = None
    if len(sys.argv) != 2:
        target_path = "test.hdr"
    else:
        target_path = sys.argv[1]

    main(target_path)
