import os
from typing import IO
import io
from typing import Union, Tuple
from pathlib import Path
from PIL import Image, PngImagePlugin
import numpy as np
import cv2

def read_image(path: Union[str, os.PathLike, IO]) -> np.ndarray:
    """
    Read a image, return uint8 RGB array of shape (H, W, 3).
    """
    if isinstance(path, (str, os.PathLike)):
        data = Path(path).read_bytes()  
    else:
        data = path.read()
    image = cv2.cvtColor(cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    return image

def write_image(path: Union[str, os.PathLike, IO], image: np.ndarray, quality: int = 95):
    """
    Write a image, input uint8 RGB array of shape (H, W, 3).
    """
    data = cv2.imencode('.jpg', cv2.cvtColor(image, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, quality])[1].tobytes()
    if isinstance(path, (str, os.PathLike)):
        Path(path).write_bytes(data)
    else:
        path.write(data)

def read_depth(path: Union[str, os.PathLike, IO]) -> Tuple[np.ndarray, float]:
    """
    Read a depth image, return float32 depth array of shape (H, W).
    """
    if isinstance(path, (str, os.PathLike)):
        data = Path(path).read_bytes()
    else:
        data = path.read()
    pil_image = Image.open(io.BytesIO(data))
    near = float(pil_image.info.get('near'))
    far = float(pil_image.info.get('far'))
    unit = float(pil_image.info.get('unit')) if 'unit' in pil_image.info else None
    depth = np.array(pil_image)
    mask_nan, mask_inf = depth == 0, depth == 65535
    depth = (depth.astype(np.float32) - 1) / 65533
    depth = near ** (1 - depth) * far ** depth
    depth[mask_nan] = np.nan
    depth[mask_inf] = np.inf
    return depth, unit


def write_depth(
    path: Union[str, os.PathLike, IO], 
    depth: np.ndarray, 
    unit: float = None,
    max_range: float = 1e5,
    compression_level: int = 7,
):
    """
    Encode and write a depth image as 16-bit PNG format.
    ### Parameters:
    - `path: Union[str, os.PathLike, IO]`
        The file path or file object to write to.
    - `depth: np.ndarray`
        The depth array, float32 array of shape (H, W). 
        May contain `NaN` for invalid values and `Inf` for infinite values.
    - `unit: float = None`
        The unit of the depth values.
    
    Depth values are encoded as follows:
    - 0: unknown
    - 1 ~ 65534: depth values in logarithmic
    - 65535: infinity
    
    metadata is stored in the PNG file as text fields:
    - `near`: the minimum depth value
    - `far`: the maximum depth value
    - `unit`: the unit of the depth values (optional)
    """
    # 和lidar depth一样约定一个边界
    depth[depth < 1] = np.inf
    depth[depth > 150] = np.inf

    mask_values, mask_nan, mask_inf = np.isfinite(depth), np.isnan(depth), np.isinf(depth)

    depth = depth.astype(np.float32)
    mask_finite = depth
    if mask_values.any():
        near = max(depth[mask_values].min(), 1e-5)
        far = max(near * 1.1, min(depth[mask_values].max(), near * max_range))
    else:   # 修正，支持存储全是INF或者NAN的depth
        near = 1e-5
        far = max_range
    depth = 1 + np.round((np.log(np.nan_to_num(depth, nan=0).clip(near, far) / near) / np.log(far / near)).clip(0, 1) * 65533).astype(np.uint16) # 1~65534
    depth[mask_nan] = 0
    depth[mask_inf] = 65535

    pil_image = Image.fromarray(depth)
    pnginfo = PngImagePlugin.PngInfo()
    pnginfo.add_text('near', str(near))
    pnginfo.add_text('far', str(far))
    if unit is not None:
        pnginfo.add_text('unit', str(unit))
    pil_image.save(path, pnginfo=pnginfo, compress_level=compression_level)