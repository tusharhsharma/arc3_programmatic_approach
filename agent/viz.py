"""Rendering helpers: frames -> PNG images / contact sheets."""
import numpy as np
from PIL import Image

# Official ARC-AGI-3 palette (arc_agi.rendering.COLOR_MAP)
PALETTE = [
    (255, 255, 255), (204, 204, 204), (153, 153, 153), (102, 102, 102),
    (51, 51, 51), (0, 0, 0), (229, 58, 163), (255, 123, 204),
    (249, 60, 49), (30, 147, 255), (136, 216, 241), (255, 220, 0),
    (255, 133, 27), (146, 18, 49), (79, 204, 48), (163, 86, 214),
]


def grid_to_image(grid, scale=6):
    g = np.asarray(grid)
    rgb = np.zeros((*g.shape, 3), dtype=np.uint8)
    for v in range(16):
        rgb[g == v] = PALETTE[v]
    img = Image.fromarray(rgb, "RGB")
    return img.resize((g.shape[1] * scale, g.shape[0] * scale), Image.NEAREST)


def contact_sheet(grids, labels, path, cols=4, scale=4, pad=8):
    """Save a labeled grid-of-grids PNG."""
    from PIL import ImageDraw
    imgs = [grid_to_image(g, scale) for g in grids]
    w = max(i.width for i in imgs)
    h = max(i.height for i in imgs)
    label_h = 14
    rows = (len(imgs) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * (w + pad) + pad,
                              rows * (h + label_h + pad) + pad), (24, 24, 24))
    draw = ImageDraw.Draw(sheet)
    for i, (img, lab) in enumerate(zip(imgs, labels)):
        r, c = divmod(i, cols)
        x = pad + c * (w + pad)
        y = pad + r * (h + label_h + pad)
        draw.text((x, y), lab[:60], fill=(230, 230, 230))
        sheet.paste(img, (x, y + label_h))
    sheet.save(path)
    return path
