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


def generate_masks(args):
    # ---加载模型---
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sam = SAM(model_type=args.sam_type, checkpoint_path=args.sam_weights, device=device)
    print(f"Successfully load SAM model loaded on {device}.")

    # ---生成所有mask并保存---
    raw_paths = []
    for dirpath, dirnames, filenames in os.walk(args.image_dir):
        if len(filenames) > 0:
            raw_paths.extend([os.path.join(dirpath, f) for f in filenames])

    if not os.path.exists(args.save_dir):
        raise ValueError(f"Output directory {args.save_dir} does not exist.")

    print('Generating masks via SAM...')
    for raw_path in tqdm(raw_paths):
        if not raw_path.endswith('.jpg'):
            continue
        raw_image = load_image(raw_path)
        masks = sam.infer(raw_image)
        masks_ = filter_masks(masks, refine=args.refine)
        seg = np.array([i['segmentation'] for i in masks_]).astype(np.uint8)
        dirname, basename = os.path.dirname(raw_path), os.path.basename(raw_path)
        mask_dir = os.path.join(args.save_dir, f'masks{args.output_name}', os.path.relpath(dirname, args.image_dir))
        if not os.path.exists(mask_dir):
            os.makedirs(mask_dir)
        np.save(os.path.join(mask_dir, basename.replace('.jpg', '.npy')), seg)
        if len(masks_) != 9 and len(masks_) != 8:
            print(f"Warning!! For {raw_path}, there are {len(masks)} mask before processing, and {len(masks_)} mask after processing.")

    print(f'All masks saved to {mask_dir}!')

    # ---对每张图的mask进行排序并保存---
    mask_paths = []
    mask_dir_all = os.path.join(args.save_dir, f'masks{args.output_name}')
    for dirpath, dirnames, filenames in os.walk(mask_dir_all):
        if len(filenames) > 0:
            mask_paths.extend([os.path.join(dirpath, f) for f in filenames])

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
        if not os.path.exists(os.path.dirname(mask_sorted_path)):
            os.makedirs(os.path.dirname(mask_sorted_path))
        np.save(mask_sorted_path, seg)
    print(f'All masks sorted and saved!')


def get_RGB_values(args):
    # ---提取特征并保存---
    before_paths = []
    after_paths = []
    n_repeats = 10
    sampler = CircleSampler(n_samples=100, repeat=n_repeats)  # 生成采样模式
    output_name = args.output_name
    if args.sample:
        args.output_name = '_sample'+args.output_name

    for dirpath, dirnames, filenames in os.walk(args.image_dir):
        if len(filenames) == 0:
            continue
        if 'before' in dirpath:
            before_paths.extend([os.path.join(dirpath, f) for f in filenames])
        if 'after' in dirpath:
            after_paths.extend([os.path.join(dirpath, f) for f in filenames])

    for paths, name in zip([before_paths, after_paths], ['before', 'after']):
        features = []  # all features of before/after images

        # ---提取每个mask的RGB均值特征并保存---
        print(f'Extracting RGB values from {name} images...')
        for p in tqdm(paths):
            img = load_image(p)
            img = np.array(img)
            dirname, basename = os.path.dirname(p), os.path.basename(p)
            seg = np.load(os.path.join(args.save_dir, f'masks_sorted{output_name}', os.path.relpath(dirname, args.image_dir), basename.replace('.jpg', '.npy'))).astype(bool)

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
                    feat.append(np.stack(rgb))
                features.append(np.column_stack(feat))
                
            else:
                feat = []
                for j in range(len(seg)):
                    mask = seg[j]
                    feat.extend(img[mask].mean(axis=0).tolist())
                features.append(feat)

        print(f"Processed {len(paths)} images from {args.image_dir}")
        features = np.vstack(features)
        df = pd.DataFrame(features, columns=[f'{j}_{i}' for i in range(1, 9) for j in ['R', 'G', 'B']]).round(6)  # 特征编号从1到8

        ids = [os.path.basename(p).replace('.jpg', '') for p in paths]
        id_col = []
        for iid in ids:
            ints = re.findall(r"\d+", iid)
            if len(ints) !=1:
                print(f"Unexpected filename format: e.g., {iid}. Use natural numbers as IDs automatically.")
                break
            id_col.append(int(ints[0]))
        
        if len(id_col) != len(ids):
            id_col = list(range(len(ids)))

        if args.sample:
            df.insert(0, 'id', np.repeat(id_col, n_repeats))
        else:
            df.insert(0, 'id', id_col)

        df_sorted = df.sort_values(by='id', ascending=True, kind='stable')

        value_dir = os.path.join(args.save_dir, f'values{args.output_name}')
        if not os.path.exists(value_dir):
            os.makedirs(value_dir)
        df_sorted.to_csv(os.path.join(value_dir, f"RGB_val_{name}.csv"), index=False)
        print(f"RGB values saved to RGB_val_{name}.csv")
    print('All RGB values extracted and saved!')


if __name__ == "__main__":
    # ---加载参数---
    args = get_args()
    if args.output_name:
        args.output_name = '_' + args.output_name.lstrip('_')
    
    if args.refine:
        args.output_name = '_refine'+args.output_name

    if args.generate_masks:
        generate_masks(args)

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
        if args.sample:
            label_expanded = np.repeat(label_df['label'].values, 10)
            label_df = pd.DataFrame(label_expanded, index=res.index, columns=['label'])

        if len(label_df) != len(res):
            print(f"Warning!!! label file length {len(label_df)} does not match data length {len(res)}.")

        res['label'] = label_df['label']

    res.to_csv(os.path.join(value_dir, f'RGB_diff.csv'), index=True)
