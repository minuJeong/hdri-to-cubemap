HDRI to Cubemap
===============
Generate 6 cubemap textures from panorama image

##### Dependencies
 - Pillow
 - numpy
 - imageio
 - FreeImage extension for hdr handing

##### Python version
 - tested on 3.7, 3.6

### How it works
Use UV from cube map, convert it into spherical vector to sample spherical panorama.

![From Imgur](https://i.imgur.com/wgzfddR.png)
>How is it being sampled (sampling resolution: 64, 64 / hdri resolution: 3072, 1536)

### How to use it
```
    python main.py <image_path(filepath/url)> <resolution>
```
Mind the resolution and python system bits. (Recommend to use 64bit Python for >1024)
