import os
import torch
import tqdm
import numpy as np
import imageio.v2 as imageio
import multiprocessing as mp
from torch.utils.data import Dataset
from utils import gen_grid_np


class SimpleDepthDataset(Dataset):
    def __init__(self, args, max_interval=None):
        self.args = args
        self.seq_dir = args.data_dir
        self.seq_name = os.path.basename(self.seq_dir.rstrip('/'))
        self.img_dir = os.path.join(self.seq_dir, 'color')
        # self.depth_dir = os.path.join(self.seq_dir, 'raw_depth', "depth")
        # self.depthmask_dir = os.path.join(self.seq_dir, 'raw_depth', "mask")
        self.flow_dir = os.path.join(self.seq_dir, 'raft_exhaustive')
        img_names = sorted(os.listdir(self.img_dir))
        self.num_imgs = min(self.args.num_imgs, len(img_names))
        self.img_names = img_names[:self.num_imgs]
        self.img_range = mp.Value('i', args.inc_step) if args.inc_step > 0 else None
        h, w, _ = imageio.imread(os.path.join(self.img_dir, img_names[0])).shape
        self.h, self.w = h, w
        self.max_interval = self.num_imgs - 1 if not max_interval else max_interval
        self.num_pts = self.args.num_pts
        self.grid = gen_grid_np(self.h, self.w).reshape(-1, 2)

        # 根据两帧的间隔，计算采样权重
        normal = torch.distributions.Normal(0, self.args.norm_neighbor if self.args.norm_neighbor > 0 else 10)
        weights = normal.log_prob(torch.linspace(0, self.num_imgs - 1, self.num_imgs)).exp()
        weights[0] = 0
        weights[1:5] = weights[4]
        self.dist_weights = weights / weights.sum()
        
        # flow & mask: ids1_ids1-16 --> ids1_ids1+16 (1 frame to 32 candidate frames)
        self.flow = np.zeros((self.num_imgs, 32, self.h, self.w, 2), dtype=np.float32)
        self.masks = np.zeros((self.num_imgs, 32, self.h, self.w), dtype=np.float32)
        self.candidate_pair_range = np.zeros((self.num_imgs, 2), dtype=np.int32)
        self.candidate_pair_range[:, 0] = np.iinfo(np.int32).max  # 2147483647
        self.candidate_pair_range[:, 1] = -np.iinfo(np.int32).max   # -2147483647
        
        # window size: 32
        for i in tqdm.trange(self.num_imgs, desc="loading flow and mask"):
            for j in range(-16, 17):
                if j == 0 or not (0 <= i + j < self.num_imgs):
                    continue
                
                mask_file = flow_file.replace('raft_exhaustive', 'full_mask').replace('.npy', '.png')
                masks = imageio.imread(mask_file) / 255.0
                if masks.sum() < 4096:
                    continue
                
                flow_file = os.path.join(self.flow_dir, f"{self.img_names[i]}_{self.img_names[i+j]}.npy")
                flow = np.load(flow_file)

                self.candidate_pair_range[i, 0] = min(self.candidate_pair_range[i, 0], j)
                self.candidate_pair_range[i, 1] = max(self.candidate_pair_range[i, 1], j)

                offset = 15 if j > 0 else 16
                self.flow[i, j + offset] = flow
                self.masks[i, j + offset] = masks
                
        countmaps = np.zeros((self.num_imgs, self.h, self.w), dtype=np.float32)
        for i in tqdm.trange(self.num_imgs):
            countmap = imageio.imread(os.path.join(self.seq_dir, 'count_maps', self.img_names[i].replace('.jpg', '.png')))
            countmaps[i] = countmap
        
        self.masks = np.round(self.masks) * countmaps[:, None]
        positive_index = self.masks > 0
        self.masks[positive_index] = 1. / np.sqrt(self.masks[positive_index] + 1)
        row_col_sums = self.masks.sum(axis=(-1, -2), keepdims=True)
        positive_index = row_col_sums > 0
        self.masks[positive_index] /= row_col_sums[positive_index]

    def __len__(self):
        return self.num_imgs**2*1000

    def increase_range(self):
        current_range = self.img_range.value
        self.img_range.value = min(self.inc_step + current_range, self.num_imgs)
        print("increasing range to ", self.img_range)

    def set_max_interval(self, max_interval):
        self.max_interval.value = min(max_interval, self.num_imgs - 1)

    def __getitem__(self, idx):
        # 从序列中间往左右img_range帧中随机选取一帧
        if self.inc_step > 0:
            id1 = idx % self.img_range.value - self.img_range.value // 2 + self.num_imgs // 2
        else:   
            id1 = idx % self.num_imgs
      
        # 选定第一帧后，候选帧的范围
        max_interval = min(self.max_interval, self.num_imgs - 1)
        start = max(id1 + self.candidate_pair_range[id1, 0], 0, id1 - max_interval)
        end = min(id1 + self.candidate_pair_range[id1, 1], self.num_imgs - 1, id1 + max_interval)
        id2s = np.arange(start, end + 1)
        
        # 采样候选帧
        normal_weights = self.dist_weights[abs(id2s - id1)]
        if self.args.norm_neighbor >= 1:
            sample_weights = normal_weights.numpy()
        else:
            sample_weights = np.ones_like(id2s) / len(id2s)
        sample_weights /= np.sum(sample_weights)
        id2 = np.random.choice(id2s, p=sample_weights)

        # 加载对应pair的flow和mask
        offset = 15 if id2 - id1 > 0 else 16
        flow = self.flow[id1, id2 - id1 + offset].reshape(-1, 2)
        mask = self.masks[id1, id2 - id1 + offset].reshape(-1)
        invalid = mask.sum() == 0
        mask = np.ones_like(mask) / mask.sum() if invalid else mask

        # 采样num_pts个匹配点
        is_replace=((mask>0).sum() < self.num_pts) # 如果可用点小于num_pts，则有放回采样
        select_mask = np.random.choice(mask.shape[0], self.num_pts, replace=is_replace, p=mask)
        frame_interval = abs(id1 - id2)
           
        # 采样pair的权重
        pair_weight = np.cos((frame_interval - 1.) / max_interval * np.pi / 3)  

        # 生成匹配点坐标
        coord1 = self.grid[select_mask]
        coord2 = coord1 + flow[select_mask]
        pts1 = torch.from_numpy(coord1).float()
        pts2 = torch.from_numpy(coord2).float()
        weights = torch.zeros_like(pts1[:, :1]) if invalid else torch.ones_like(pts1[:, :1]) * pair_weight

        if np.random.rand() > 0.5:
            id1, id2, pts1, pts2= id2, id1, pts2, pts1

        data = {'ids1': id1,
                'ids2': id2,
                'pts1': pts1,  # [n_pts, 2]
                'pts2': pts2,  # [n_pts, 2]
                'weights': weights,  # [n_pts, 1]
                }
        return data
    
def test():
    id2s = np.arange(1, 11)
    print(id2s)
    
if __name__ == '__main__':
    test()



