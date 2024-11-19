import os
import glob
import torch
import tqdm
import numpy as np
from torch.utils.data import Dataset


class LongtermDataset(Dataset):
    def __init__(self, args, max_interval=8, minbatch=4):
        self.args = args
        self.seq_dir = args.data_dir
        self.seq_name = os.path.basename(self.seq_dir.rstrip('/'))
        self.img_dir = os.path.join(self.seq_dir, 'color')
        self.match_dir = os.path.join(self.seq_dir, 'match')
        if not os.path.exists(self.match_dir):
            raise FileNotFoundError(f"Directory '{self.match_dir}' does not exist.")
        
        all_matches = sorted(glob.glob(os.path.join(self.match_dir, '*.npz')))
        all_images = sorted(glob.glob(os.path.join(self.img_dir, '*.jpg')))
        self.num_imgs = min(self.args.num_imgs, len(all_images))
        # n*7: [id1, x1, y1, id2, x2, y2, weight]
        self.matches = np.concatenate([np.load(matchfile)['match'] for matchfile in all_matches[:self.num_imgs]], axis=0)
        
        self.matchid = []
        self.candidates = []
        self.batchsize = 1e10
        valid_pairs = 0
        total_pairs = 0
        for i in range(self.num_imgs):
            matches_i = self.matches[self.matches[..., 0] == i]
            if matches_i.size == 0:
                continue

            candidate_ids, counts = np.unique(matches_i[:, 3], return_counts=True)
            total_pairs += len(candidate_ids)

            valid_mask = counts >= minbatch
            valid_candidates = candidate_ids[valid_mask]

            if valid_candidates.size == 0:
                continue

            self.batchsize = min(self.batchsize, counts[valid_mask].min())
            self.matchid.append(i)
            self.candidates.append(valid_candidates)
            valid_pairs += valid_candidates.size
        print("longterm dataset batchsize: ", self.batchsize)
        
        if valid_pairs < self.num_imgs*2:
            print("longterm dataset no enough pairs")
            self.longterm = False
        else:
            self.longterm = True

    def __len__(self):
        if not self.longterm:
            return 100
        return len(self.matchid)**2*self.batchsize**2

    def __getitem__(self, idx):
        if not self.longterm:
            return {
                'ids1': -1,
                'ids2': -1,
                'pts1': torch.zeros(0, 2),
                'pts2': torch.zeros(0, 2),
                'weights': torch.zeros(0, 1),
            }

        # 获取有效索引并随机选取第二张图片
        idx %= len(self.matchid)
        id1 = self.matchid[idx]
        id2 = np.random.choice(self.candidates[idx])
        
        # 筛选匹配点
        candidates = self.matches[(self.matches[..., 0] == id1) & (self.matches[..., 3] == id2)]

        # 随机选取 batchsize 个匹配点
        select_id = np.random.choice(len(candidates), self.batchsize, replace=False)
        select_pts = torch.from_numpy(candidates[select_id])
        pts1 = select_pts[..., 1:3]
        pts2 = select_pts[..., 4:6]
        weights = select_pts[..., 6:7]

        # 随机交换先后顺序
        if torch.rand(1) > 0.5:
            id1, id2 = id2, id1
            pts1, pts2 = pts2, pts1

        return {
            'ids1': id1,
            'ids2': id2,
            'pts1': pts1,
            'pts2': pts2,
            'weights': weights,
        }
    
def test():
    a = [1,4,3,4,5,4,7,3,9]
    b,count = np.unique(a, return_counts=True)
    mask = count >= 2
    print(b[mask], type(b[mask]))

if __name__ == "__main__":
    test()