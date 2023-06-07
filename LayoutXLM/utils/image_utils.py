from PIL import Image
import numpy as np

def load_image(image_path, size=None):
    image = Image.open(image_path).convert("RGB")
    w, h = image.size
    if size is not None:
        # resize image
        image = image.resize((size, size))
        image = np.asarray(image)
        image = image[:, :, ::-1]  # flip color channels from RGB to BGR
        image = image.transpose(2, 0, 1)  # move channels to first dimension
    return image, (w, h)


def simplify_bbox(bbox):
    return [
        min(bbox[0::2]),
        min(bbox[1::2]),
        max(bbox[2::2]),
        max(bbox[3::2]),
    ]


def merge_bbox(bbox_list):
    x0, y0, x1, y1 = list(zip(*bbox_list))
    return [min(x0), min(y0), max(x1), max(y1)]

def normalize_bbox(bbox, size):
    return [
        int(1000 * bbox[0] / size[0]),
        int(1000 * bbox[1] / size[1]),
        int(1000 * bbox[2] / size[0]),
        int(1000 * bbox[3] / size[1]),
    ]