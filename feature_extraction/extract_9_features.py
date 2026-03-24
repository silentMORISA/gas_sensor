import numpy as np
import matplotlib.pyplot as plt   
import torch
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import cv2
from tqdm import tqdm
import os
from typing import List, Tuple, Any, Optional
from PIL import Image
import pandas as pd
import argparse

Point = Tuple[float, float]

def sort_3x3_row_major(points: List[Point], attach: Optional[List[Any]] = None):
    """
    将 3x3 点阵按 “从左到右, 从上到下” 排序。
    points: [(x,y), ...] 恰好 9 个点
    attach: 可选，与 points 一一对应的额外数组（标签/数值），会按同一顺序返回
    返回: (sorted_points, order, sorted_attach[可选])
    """
    assert len(points) == 9, "必须是 3x3 共 9 个点"
    # 1) 全局按 y 升序
    idx_by_y = sorted(range(9), key=lambda i: points[i][1])

    # 2) 切成 3 行，每行 3 个；行内按 x 升序
    rows = []
    for r in range(3):
        row_idx = idx_by_y[r*3:(r+1)*3]
        row_idx_sorted = sorted(row_idx, key=lambda i: points[i][0])
        rows.append(row_idx_sorted)

    # 3) 展平得到最终顺序（上→下，行内左→右）
    order = [i for row in rows for i in row]

    sorted_points = [points[i] for i in order]
    if attach is None:
        return sorted_points, order
    else:
        assert len(attach) == 9, "attach 长度需与 points 相同"
        sorted_attach = [attach[i] for i in order]
        return sorted_points, order, sorted_attach


def load_image(image_path: str) -> np.ndarray:
    """加载图像为 RGB 格式的numpy数组"""
    image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    return image


def filter_masks(masks: dict) -> list:
    """过滤符合面积和宽高比要求的mask"""
    return [i for i in masks if i['area']>15000 and i['area']<18000 and 1.15 > i['bbox'][2]/i['bbox'][3] > 0.85]

class SAM():
    def __init__(self, model_type: str, checkpoint_path: str, device: torch.device):
        self.device = device
        self.sam = sam_model_registry[model_type](checkpoint=checkpoint_path).to(device)

    def generate_masks(self, image: np.ndarray, points_per_side: int = 16,
                       pred_iou_thresh: float = 0.95,
                       stability_score_thresh: float = 0.95,
                       crop_n_layers: int = 1,
                       crop_n_points_downscale_factor: int = 2,
                       min_mask_region_area: int = 1000) -> List[dict]:
        mask_generator = SamAutomaticMaskGenerator(
            model=self.sam,
            points_per_side=points_per_side,
            pred_iou_thresh=pred_iou_thresh,
            stability_score_thresh=stability_score_thresh,
            crop_n_layers=crop_n_layers,
            crop_n_points_downscale_factor=crop_n_points_downscale_factor,
            min_mask_region_area=min_mask_region_area,
        )
        masks = mask_generator.generate(image)
        return masks
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_type",
        type=str,
        default="vit_h",
        help="vit_h, vit_l or vit_b",
    )
    parser.add_argument(
        "--weights",
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
        default="/data/gas_test",
        help="directory to save masks and features",
    )
    args = parser.parse_args()

    # ---加载模型---
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sam = SAM(model_type=args.model_type, checkpoint_path=args.weights, device=device)

    # ---读取所有图像路径---
    raw_paths = []
    for dirpath, dirnames, filenames in os.walk(args.image_dir):
        if len(filenames) > 0:
            raw_paths.extend([os.path.join(dirpath, f) for f in filenames])

    if not os.path.exists(args.save_dir):
        raise ValueError(f"Output directory {args.save_dir} does not exist.")

    # ---生成所有mask并保存---
    for raw_path in tqdm(raw_paths):
        raw_image = load_image(raw_path)
        masks = sam.generate_masks(raw_image)
        masks_ = filter_masks(masks)
        seg = np.array([i['segmentation'] for i in masks_]).astype(np.uint8)
        dirname, basename = os.path.dirname(raw_path), os.path.basename(raw_path)
        # save_path = dirname.replace('images', 'masks')
        mask_dir = os.path.join(args.save_dir, 'masks', os.path.relpath(dirname, args.image_dir))
        if not os.path.exists(mask_dir):
            os.makedirs(mask_dir)
        np.save(os.path.join(mask_dir, basename.replace('.jpg', '.npy')), seg)
        if len(masks_) != 9:
            print(f"Warning!! For {raw_path}, there are {len(masks)} mask before processing, and {len(masks_)} mask after processing.")
    print('All masks saved!')

    mask_paths = []
    # mask_dir = '/data/gas/masks'
    mask_dir_all = os.path.join(args.save_dir, 'masks')
    for dirpath, dirnames, filenames in os.walk(mask_dir_all):
        if len(filenames) > 0:
            mask_paths.extend([os.path.join(dirpath, f) for f in filenames])


    # ---对每张图的mask进行排序并保存---
    # 每张图的所有mask
    for p in mask_paths:
        seg = np.load(p)
        x_avg = []
        y_avg = []
        # 每个mask
        for i in range(len(seg)):
            coords = np.where(seg[i] == 1)
            x_avg.append(np.mean(coords[1]))
            y_avg.append(np.mean(coords[0]))
        points = list(zip(x_avg, y_avg))
        sorted_points, order = sort_3x3_row_major(points)
        seg = seg[order]
        mask_sorted_path = p.replace('masks', 'masks_sorted')
        if not os.path.exists(os.path.dirname(mask_sorted_path)):
            os.makedirs(os.path.dirname(mask_sorted_path))
        np.save(mask_sorted_path, seg)
    print('All masks sorted!')


    # ---提取每个mask的RGB均值特征并保存---
    normal_before_paths = []
    normal_after_paths = []
    patient_before_paths = []
    patient_after_paths = []
    # root = '/data/gas/images'
    for dirpath, dirnames, filenames in os.walk(args.image_dir):
        if len(filenames) == 0:
            continue
        if 'normal' in dirpath and 'before' in dirpath:
            normal_before_paths.extend([os.path.join(dirpath, f) for f in filenames])
        if 'normal' in dirpath and 'after' in dirpath:
            normal_after_paths.extend([os.path.join(dirpath, f) for f in filenames])
        if 'patient' in dirpath and 'before' in dirpath:
            patient_before_paths.extend([os.path.join(dirpath, f) for f in filenames])
        if 'patient' in dirpath and 'after' in dirpath:
            patient_after_paths.extend([os.path.join(dirpath, f) for f in filenames])

    names = ['normal_before', 'normal_after', 'patient_before', 'patient_after']
    paths_list = [normal_before_paths, normal_after_paths, patient_before_paths, patient_after_paths]
    for paths, name in zip(paths_list, names):
        features = []
        for p in tqdm(paths):
            img = Image.open(p)
            img = np.array(img)
            # seg = np.load(p.replace('images', 'masks_sorted').replace('.jpg', '.npy')).astype(bool)
            dirname, basename = os.path.dirname(p), os.path.basename(p)
            seg = np.load(os.path.join(args.save_dir, 'masks_sorted', os.path.relpath(dirname, args.image_dir), basename.replace('.jpg', '.npy'))).astype(bool)
            feat = []
            for j in range(9):
                mask = seg[j]
                feat.extend(img[mask].mean(axis=0).tolist())
            features.append(feat)
        print(f"Processed {len(paths)} images from {args.image_dir}")

        df = pd.DataFrame(features, columns=[f'{j}_{i}' for i in range(9) for j in ['R', 'G', 'B']]).round(6)
        ids = [os.path.basename(p).replace('.jpg', '') for p in paths]
        df.insert(0, 'id', ids)
        df['id_int'] = [int(i) for i in ids]
        df_sorted = df.sort_values(by='id_int', ascending=True)
        del df_sorted['id_int']
        value_dir = os.path.join(args.save_dir, 'values')
        if not os.path.exists(value_dir):
            os.makedirs(value_dir)
        df_sorted.to_csv(os.path.join(value_dir, f"RGB_val_{name}.csv"), index=False)
        print(f"Features saved to RGB_val_{name}.csv")
    print('All features extracted and saved!')

    # ---计算前后差值并保存---
    # root = '/data/gas/values'
    normal_before = pd.read_csv(os.path.join(value_dir, 'RGB_val_normal_before.csv'))
    normal_after = pd.read_csv(os.path.join(value_dir, 'RGB_val_normal_after.csv'))
    patient_before = pd.read_csv(os.path.join(value_dir, 'RGB_val_patient_before.csv')) 
    patient_after = pd.read_csv(os.path.join(value_dir, 'RGB_val_patient_after.csv'))

    res_normal = normal_after.drop('id', axis=1, inplace=False) - normal_before.drop('id', axis=1, inplace=False)
    res_normal = res_normal.round(6)
    res_normal.insert(0, 'id', normal_after['id'])
    res_normal.to_csv(os.path.join(value_dir, 'RGB_diff_normal.csv'), index=False)

    res_patient = patient_after.drop('id', axis=1, inplace=False) - patient_before.drop('id', axis=1, inplace=False)
    res_patient = res_patient.round(6)
    res_patient.insert(0, 'id', patient_after['id'])
    res_patient.to_csv(os.path.join(value_dir, 'RGB_diff_patient.csv'), index=False)