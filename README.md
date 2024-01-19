# upscale_x4

This repository provides out of the box methods for upscaling **images** and **gifs** by 4. A pretrained model is already bundled in. See the simple example below.

```python
from upscale_x4 import upscale_gif_x4, upscale_image_x4

upscaled_gif = upscale_gif_x4("path \ to \ gif")
upscaled_img = upscale_image_x4("path \ to \ image")
```

Currently supports:

1. JPGs
2. PNGs (mixed results and loses transparency)
3. GIFs
