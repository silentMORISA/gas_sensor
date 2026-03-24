import argparse
import cv2
import numpy as np
from typing import List, Tuple, Any, Optional
import matplotlib.pyplot as plt
import gc
import os

Point = Tuple[float, float]
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUPPORTED_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")

def str2bool(v):
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        v_lower = v.lower()
        if v_lower in ("yes", "true", "t", "y", "1"):
            return True
        if v_lower in ("no", "false", "f", "n", "0"):
            return False
    raise argparse.ArgumentTypeError("Boolean value expected. Use true/false, 1/0, yes/no or omit the value for flags.")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sam_type",
        type=str,
        default="vit_h",
        help="vit_h, vit_l or vit_b",
    )
    parser.add_argument(
        "--sam_weights",
        type=str,
        default="/data/basemodel/sam_vit_h_4b8939.pth",
        help="path to the SAM model weights",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="cuda or cpu, default is cuda if available",
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        default="/data/gas/images",
        help="directory of input images",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="output",
        help="directory to save masks and features; relative paths are created under the project root",
    )
    parser.add_argument(
        "--label_file",
        type=str,
        default=None,
        help="path to the label file",
    )
    parser.add_argument(
        "--refine",
        type=str2bool,
        nargs='?',
        const=True,
        default=False,
        help="whether to refine masks to smaller circles (use --refine or --refine true/false)",
    )
    parser.add_argument(
        "--output_name",
        type=str,
        default='',
        help="output csv file name will be 'RGB_val_before_{output_name}.csv' and 'RGB_val_after_{output_name}.csv'",
    )
    parser.add_argument(
        "--generate_masks",
        type=str2bool,
        nargs='?',
        const=True,
        default=True,
        help="Whether to generate masks. If you have sorted masks already, set False to skip mask generation (use --generate_masks false)",
    )
    parser.add_argument(
        "--sample",
        type=str2bool,
        nargs='?',
        const=True,
        default=False,
        help="Whether to sample points within each mask circle. If False, use all pixels in each mask (use --sample false)",
    )
    parser.add_argument(
        "--visualize_masks",
        type=str2bool,
        nargs='?',
        const=True,
        default=False,
        help="Whether to save mask overlays on images to visualizations*/ (use --visualize_masks true)",
    )

    args = parser.parse_args()
    return args


def resolve_save_dir(save_dir: str) -> str:
    """Resolve save_dir and create it automatically if needed."""
    if os.path.isabs(save_dir):
        resolved = save_dir
    else:
        resolved = os.path.join(PROJECT_ROOT, save_dir)
    os.makedirs(resolved, exist_ok=True)
    return resolved


def is_supported_image_file(path: str) -> bool:
    return path.lower().endswith(SUPPORTED_IMAGE_EXTENSIONS)


def get_mask_filename(image_path: str) -> str:
    return os.path.splitext(os.path.basename(image_path))[0] + ".npy"


def sort_3x3_row_major(points: List[Point], attach: Optional[List[Any]] = None):
    """
    将 3x3 点阵按 “从左到右, 从上到下” 排序。
    points: [(x,y), ...]，长度为8或9
    attach: 可选，与 points 一一对应的额外数组（标签/数值），会按同一顺序返回
    返回: (sorted_points, order, sorted_attach[可选])
    """
    # 1) 全局按 y 升序
    idx_by_y = sorted(range(len(points)), key=lambda i: points[i][1])

    # 2) 切成 3 行，每行 3 个；行内按 x 升序. 最终顺序（上→下，行内左→右）
    order = []
    if len(points) == 9:
        for r in range(3):
            row_idx = idx_by_y[r*3:(r+1)*3]
            row_idx_sorted = sorted(row_idx, key=lambda i: points[i][0])
            order.extend(row_idx_sorted)
    elif len(points) == 8:
        for r in range(3):
            if r == 0:
                row_idx = idx_by_y[:2]
            else:
                row_idx = idx_by_y[2+(r-1)*3:2+(r)*3]
            row_idx_sorted = sorted(row_idx, key=lambda i: points[i][0])
            order.extend(row_idx_sorted)
    else:
        raise ValueError("points 长度必须为8或9")

    sorted_points = [points[i] for i in order]

    # 3) 处理有9个色块的情况，去掉第一个点
    if len(points) == 9:
        order.pop(0)
        sorted_points.pop(0)
    if attach is None:
        return sorted_points, order
    else:
        assert len(attach) == len(points), "attach 长度需与 points 相同"
        sorted_attach = [attach[i] for i in order]
        return sorted_points, order, sorted_attach


def load_image(image_path: str) -> np.ndarray:
    """
    加载图像为 RGB 格式的numpy数组
    Be careful that if use PIL.Image.open, the image might have orientation issues due to EXIF data
    """
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        raise ValueError(f"Failed to read image: {image_path}")
    image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    return image


def filter_masks(masks: dict, refine=False) -> list:
    """过滤符合面积和宽高比要求的mask"""
    masks = [i for i in masks if i['area']> 3700 and i['area'] < 4800 and 1.2 > i['bbox'][2]/i['bbox'][3] > 0.80]
    if refine:
        for i in range(len(masks)):
            masks[i]['segmentation'] = refine_circle(masks[i]['segmentation'])
    return masks


def refine_circle(mask, keep_ratio=0.7):
    """将二值 mask 收缩为圆心相同但半径缩小的圆形 mask"""
    # 找到前景点坐标
    ys, xs = np.where(mask > 0)

    # 计算圆心位置（用质心计算）
    cx = xs.mean()
    cy = ys.mean()

    # 计算每个点到圆心的最大距离（≈原圆半径）
    dists = np.sqrt((xs - cx)**2 + (ys - cy)**2)
    r_old = dists.max()

    # 新半径
    r_new = r_old * keep_ratio

    # 构建网格坐标
    H, W = mask.shape
    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')

    # 计算每个像素的距离
    dist_img = np.sqrt((xx - cx)**2 + (yy - cy)**2)

    # 新圆（二值 mask）
    new_mask = (dist_img <= r_new).astype(np.uint8)

    return new_mask


def show_masks_on_image(raw_image, masks, random_color=True, save_path=None, title=None):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(np.array(raw_image))
    ax.set_autoscale_on(False)
    for mask in masks:
        show_mask(mask, ax=ax, random_color=random_color)
    if title:
        ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight", pad_inches=0)
        plt.close(fig)
    else:
        plt.show()
        plt.close(fig)

    gc.collect()

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    del mask
    gc.collect()


class CircleSampler:
    def __init__(self, n_samples=100, repeat=1):
        self.points = []
        self.n = n_samples
        self.repeat = repeat
        self.generate_xy()

    def generate_xy(self):
        u = np.random.rand(self.repeat, self.n)
        theta = np.random.rand(self.repeat, self.n) * 2 * np.pi

        self.r = np.sqrt(u)
        self.x = np.cos(theta)
        self.y = np.sin(theta)
    
    def points_in_circle(self, radius, center=(0, 0)):
        """return: [repeat, 2, n_samples]"""
        r = self.r * radius
        x = self.x * r + center[0]
        y = self.y * r + center[1]
        # self.points = np.column_stack((x, y))
        self.points = np.stack([x, y]).transpose(1, 0, 2).astype(int)
        return self.points

    def get_points(self):
        if self.points is None:
            raise ValueError("Points have not been generated yet.")
        return self.points
