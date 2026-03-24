import numpy as np
import torch
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from tqdm import tqdm
import os
import re
from typing import List
import pandas as pd
from utils import *

class SAM():
    def __init__(self, model_type: str, checkpoint_path: str, device: torch.device):
        self.device = device
        self.sam = sam_model_registry[model_type](checkpoint=checkpoint_path).to(device)

    def infer(self, image: np.ndarray, points_per_side: int = 16,
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


def collect_image_paths(image_dir: str, keyword: str = None) -> List[str]:
    raw_paths = []
    for dirpath, _, filenames in os.walk(image_dir):
        if keyword is not None and keyword not in dirpath:
            continue
        for filename in filenames:
            if not is_supported_image_file(filename):
                continue
            raw_paths.append(os.path.join(dirpath, filename))
    return sorted(raw_paths)


def generate_masks(args):
    # ---加载模型---
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sam = SAM(model_type=args.sam_type, checkpoint_path=args.sam_weights, device=device)
    print(f"Successfully load SAM model loaded on {device}.")

    # ---生成所有mask并保存---
    raw_paths = collect_image_paths(args.image_dir)
    if not raw_paths:
        raise ValueError(f"No supported images were found under {args.image_dir}.")

    print('Generating masks via SAM...')
    mask_dir_all = os.path.join(args.save_dir, f'masks{args.output_name}')
    for raw_path in tqdm(raw_paths):
        raw_image = load_image(raw_path)
        masks = sam.infer(raw_image)
        masks_ = filter_masks(masks, refine=args.refine)
        seg = np.array([i['segmentation'] for i in masks_]).astype(np.uint8)
        dirname, basename = os.path.dirname(raw_path), os.path.basename(raw_path)
        mask_dir = os.path.join(args.save_dir, f'masks{args.output_name}', os.path.relpath(dirname, args.image_dir))
        os.makedirs(mask_dir, exist_ok=True)
        np.save(os.path.join(mask_dir, get_mask_filename(basename)), seg)
        if len(masks_) != 9 and len(masks_) != 8:
            print(f"Warning!! For {raw_path}, there are {len(masks)} mask before processing, and {len(masks_)} mask after processing.")

    print(f'All masks saved to {mask_dir_all}!')

    # ---对每张图的mask进行排序并保存---
    mask_paths = []
    for dirpath, dirnames, filenames in os.walk(mask_dir_all):
        if len(filenames) > 0:
            mask_paths.extend([os.path.join(dirpath, f) for f in filenames if f.endswith('.npy')])
    mask_paths.sort()

    # 每张图的所有masks
    print('Sorting masks...')
    for p in tqdm(mask_paths):
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
        mask_sorted_path = p.replace(f'masks{args.output_name}', f'masks_sorted{args.output_name}')
        os.makedirs(os.path.dirname(mask_sorted_path), exist_ok=True)
        np.save(mask_sorted_path, seg)
    print(f'All masks sorted and saved!')


def save_mask_visualizations(args):
    raw_paths = collect_image_paths(args.image_dir)
    if not raw_paths:
        return

    print('Saving mask visualizations...')
    for raw_path in tqdm(raw_paths):
        dirname, basename = os.path.dirname(raw_path), os.path.basename(raw_path)
        mask_path = os.path.join(
            args.save_dir,
            f'masks_sorted{args.output_name}',
            os.path.relpath(dirname, args.image_dir),
            get_mask_filename(basename),
        )
        if not os.path.exists(mask_path):
            print(f"Warning!! Sorted mask not found for visualization: {mask_path}")
            continue

        raw_image = load_image(raw_path)
        seg = np.load(mask_path).astype(bool)
        viz_dir = os.path.join(
            args.save_dir,
            f'visualizations{args.output_name}',
            os.path.relpath(dirname, args.image_dir),
        )
        save_path = os.path.join(viz_dir, f"{os.path.splitext(basename)[0]}_masks.png")
        show_masks_on_image(raw_image, seg, random_color=False, save_path=save_path)
    print('All mask visualizations saved!')


def get_RGB_values(args):
    # ---提取特征并保存---
    before_paths = collect_image_paths(args.image_dir, keyword='before')
    after_paths = collect_image_paths(args.image_dir, keyword='after')
    n_repeats = 10
    sampler = CircleSampler(n_samples=100, repeat=n_repeats)  # 生成采样模式
    output_name = args.output_name
    if args.sample:
        args.output_name = '_sample'+args.output_name

    for paths, name in zip([before_paths, after_paths], ['before', 'after']):
        features = []  # all features of before/after images

        # ---提取每个mask的RGB均值特征并保存---
        print(f'Extracting RGB values from {name} images...')
        for p in tqdm(paths):
            img = load_image(p)
            img = np.array(img)
            dirname, basename = os.path.dirname(p), os.path.basename(p)
            seg = np.load(
                os.path.join(
                    args.save_dir,
                    f'masks_sorted{output_name}',
                    os.path.relpath(dirname, args.image_dir),
                    get_mask_filename(basename),
                )
            ).astype(bool)

            if args.sample:
                feat = []  # all features of one image
                for i in range(len(seg)):
                    coords = np.where(seg[i] == 1)
                    x_avg = np.mean(coords[1])
                    y_avg = np.mean(coords[0])
                    r = np.max(np.sqrt((coords[1] - x_avg)**2 + (coords[0] - y_avg)**2))
                    points = sampler.points_in_circle(radius=r, center=(x_avg, y_avg))  # [n_repeats, 2, n_samples]
                    rgb = []  # r, g, b features for one circle with n repeats
                    for xy in points:
                        rgb.append(img[xy[0], xy[1]].mean(axis=0).tolist())
                    feat.extend(np.stack(rgb).mean(axis=0).tolist())
                features.append(feat)
                
            else:
                feat = []
                for j in range(len(seg)):
                    mask = seg[j]
                    feat.extend(img[mask].mean(axis=0).tolist())
                features.append(feat)

        print(f"Processed {len(paths)} images from {args.image_dir}")
        features = np.vstack(features)
        df = pd.DataFrame(features, columns=[f'{j}_{i}' for i in range(1, 9) for j in ['R', 'G', 'B']]).round(6)  # 特征编号从1到8

        ids = [os.path.splitext(os.path.basename(p))[0] for p in paths]
        id_col = []
        for iid in ids:
            ints = re.findall(r"\d+", iid)
            if len(ints) !=1:
                print(f"Unexpected filename format: e.g., {iid}. Use natural numbers as IDs automatically.")
                break
            id_col.append(int(ints[0]))
        
        if len(id_col) != len(ids):
            id_col = list(range(len(ids)))

        df.insert(0, 'id', id_col)

        df_sorted = df.sort_values(by='id', ascending=True, kind='stable')

        value_dir = os.path.join(args.save_dir, f'values{args.output_name}')
        os.makedirs(value_dir, exist_ok=True)
        df_sorted.to_csv(os.path.join(value_dir, f"RGB_val_{name}.csv"), index=False)
        print(f"RGB values saved to RGB_val_{name}.csv")
    print('All RGB values extracted and saved!')


if __name__ == "__main__":
    # ---加载参数---
    args = get_args()
    args.save_dir = resolve_save_dir(args.save_dir)
    if args.output_name:
        args.output_name = '_' + args.output_name.lstrip('_')
    
    if args.refine:
        args.output_name = '_refine'+args.output_name

    if args.generate_masks:
        generate_masks(args)

    if args.visualize_masks:
        save_mask_visualizations(args)

    get_RGB_values(args)


    # ---计算前后差值并保存---
    #### 0: normal, 1: patient
    value_dir = os.path.join(args.save_dir, f'values{args.output_name}')
    before_value = pd.read_csv(os.path.join(value_dir, f'RGB_val_before.csv'), index_col='id')
    after_value = pd.read_csv(os.path.join(value_dir, f'RGB_val_after.csv'), index_col='id')

    res = after_value - before_value
    res = res.round(6)

    # if with labels
    if args.label_file is not None:
        label_df = pd.read_csv(args.label_file, index_col='id')
        if len(label_df) != len(res):
            print(f"Warning!!! label file length {len(label_df)} does not match data length {len(res)}.")

        res['label'] = label_df['label']

    res.to_csv(os.path.join(value_dir, f'RGB_diff.csv'), index=True)
