import glob
import os
import pdb
import cv2
import tqdm
import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import utils
from loss import masked_l1_loss
from kornia import morphology as morph
from wis3d import Wis3D
import pdb
from pathlib import Path
from model.nvp_nonlin import NVPnonlin

def de_parallel(model):
    return model.module if hasattr(model, 'module') else model

class DepthMem(nn.Module):
    def __init__(self, args, depthmaps, device='cuda'):
        super(DepthMem, self).__init__()
        self.args = args
        self.device = device
        if args.opt_depth:
            self.depthmaps = nn.parameter.Parameter(depthmaps.clone())
        else:
            self.depthmaps = depthmaps.clone().to(device)


class ComboTrainer():
    def __init__(self, args, device='cuda'):
        self.args = args
        self.device = device
        self.out_dir = args.out_dir

        self.read_data()
        self.depthmem = DepthMem(args, self.depthmaps, device=device)
        self.wis3d = Wis3D(self.out_dir+"/viserr", f"vis_err", "xyz")

        self.deform_nvp = NVPnonlin(n_layers=6,
                                    n_frames=self.images.shape[0],
                                    feature_dim=args.feat_dim,
                                    t_dim=16,
                                    bound=self.bound,
                                    spatial_base_res=args.spatial_base_res,
                                    net_layer=args.net_layer,
                                    device=device).to(device)

        self.optimizer = torch.optim.Adam([
            {'params': self.deform_nvp.parameters(), 'lr': args.lr_deform},
            {'params': self.depthmem.parameters(), 'lr': args.lr_depth},
        ])

        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,
                                                         step_size=args.lrate_decay_steps,
                                                         gamma=args.lrate_decay_factor)

        self.step = self.load_from_ckpt(self.out_dir if self.args.load_dir == '' else self.args.load_dir,
                                        load_opt=self.args.load_opt,
                                        load_scheduler=self.args.load_scheduler)

        self.time_steps = torch.linspace(
            1, self.num_imgs, self.num_imgs, device=self.device)[:, None] / self.num_imgs

    def read_data(self):
        self.read_done = False
        self.seq_dir = self.args.data_dir
        self.seq_name = os.path.basename(self.seq_dir.rstrip('/'))
        self.img_dir = os.path.join(self.seq_dir, 'color')

        # load images
        img_files = sorted(glob.glob(os.path.join(self.img_dir, '*')))
        self.num_imgs = min(self.args.num_imgs, len(img_files))
        self.img_files = img_files[:self.num_imgs]
        images = []

        for img_file in tqdm(self.img_files, desc="Trainer: Loading images"):
            images.append(imageio.imread(img_file) / 255.)

        images = np.array(images)
        self.images = torch.from_numpy(images).float()  # [n_imgs, h, w, 3]
        self.h, self.w = self.images.shape[1:3]

        # Load masks
        mask_files = [
            img_file.replace('color', 'mask').replace('.jpg', '.png')
            for img_file in self.img_files
        ]
        if mask_files and os.path.exists(mask_files[0]):
            masks = []
            for mask_file in tqdm(mask_files, desc="Trainer: Loading masks"):
                mask = imageio.imread(mask_file)
                if mask.ndim == 3:
                    mask = mask[..., :3].sum(axis=-1)
                masks.append(mask / 255.)
            masks = np.array(masks)
            self.masks = torch.from_numpy(masks).to(
                self.device) > 0.  # [n_imgs, h, w]
            self.with_mask = True
        else:
            self.masks = torch.ones(self.images.shape[:-1],
                                    device=self.device) > 0.
            self.with_mask = False

        self.grid = utils.gen_grid(
            self.h, self.w, device=self.device, normalize=False, homogeneous=True).float()  # [h, w, 3]

        # load annotation for evaluation
        annotation_dir = os.path.join(self.seq_dir, 'annotation')
        annotation_file = "{}/{}.pkl".format(annotation_dir, self.seq_name)
        if not os.path.exists(annotation_file):
            print("\n Annotation file not found")
            self.eval = False
        else:
            self.eval = True
            annotation = np.load(annotation_file, allow_pickle=True).item()
            # psot process...

        # init depthmaps
        self.depthmaps = torch.zeros((self.num_imgs, int(
            self.h*self.args.depth_res), int(self.w*self.args.depth_res)), device=self.device)

        # load depthmaps
        depth_dir = os.path.join(self.seq_dir, self.args.depth_dir, 'depth')
        depth_files = sorted(glob.glob(os.path.join(depth_dir, '*.npz')))
        depth_files = depth_files[:self.num_imgs]
        assert len(
            depth_files) == self.num_imgs, "Number of depth files does not match number of images"
        for i, depth_file in enumerate(tqdm(depth_files, desc="Trainer: Loading depth maps")):
            depth = np.load(depth_file)['depth'].astype(np.float32)
            depth = cv2.resize(
                depth,
                (int(self.w * self.args.depth_res),
                 int(self.h * self.args.depth_res)),
                interpolation=cv2.INTER_NEAREST
            )
            self.depthmaps[i] = torch.from_numpy(depth).to(self.device)

        # to constrain the optimization of depthmaps
        self.depth_grad_maps = torch.zeros(
            (*self.depthmaps.shape, 2), device=self.device)
        coarse_grid = utils.gen_grid(
            self.depthmaps.shape[-2], self.depthmaps.shape[-1], device=self.device, normalize=False, homogeneous=False)

        # caculate the gradient of depthmaps
        for fid in range(self.num_imgs):
            self.depth_grad_maps[fid] = self.get_pixel_depth_gradient(coarse_grid.reshape(1, -1, 2), torch.tensor([fid]).to(
                self.device), original=True, scale=False).reshape(self.depthmaps.shape[-2], self.depthmaps.shape[-1], 2)  # [1, 1, w*h, 2] -> [h, w, 2]

        fov = torch.tensor(40).to(self.device)
        self.f = self.w / (2 * torch.tan((fov) / 2 / 180 * torch.pi))

        # get all pts
        all_pts = []
        for i in tqdm.trange(self.depthmaps.shape[0], desc="Calculating all points"):
            depthi = self.get_init_depth_maps([i])  # [1, h, w]
            all_pts.append(self.unproject(
                self.grid[..., 0:2].reshape(-1, 2), depthi.reshape(-1, 1)).reshape(-1, 3))
        if len(all_pts) > 0:
            all_pts = torch.cat(all_pts, dim=0)
        else:
            all_pts = torch.tensor([[0, 0, 0], [0.5, 0.5, 0.5], [
                                   1, 1, 1.0]], device=self.device)

        # caculate the bound of all points
        self.bound = torch.zeros(2, 3)
        x_mid = torch.median(all_pts[:, 0])
        y_mid = torch.median(all_pts[:, 1])
        z_mid = torch.median(all_pts[:, 2])

        x_range = min(x_mid - all_pts[:, 0].min(),
                      all_pts[:, 0].max() - x_mid) * 0.8
        y_range = min(y_mid - all_pts[:, 1].min(),
                      all_pts[:, 1].max() - y_mid) * 0.8
        z_range = min(z_mid - all_pts[:, 2].min(),
                      all_pts[:, 2].max() - z_mid) * 0.8

        assert x_range > 0 and y_range > 0 and z_range > 0

        self.bound[0, 0] = x_mid - x_range
        self.bound[1, 0] = x_mid + x_range
        self.bound[0, 1] = y_mid - y_range
        self.bound[1, 1] = y_mid + y_range
        self.bound[0, 2] = z_mid - z_range
        self.bound[1, 2] = z_mid + z_range
        self.bound = self.bound.to(self.device)
        self.read_done = True

    def get_init_depth_maps(self, ids):
        grid = self.grid[..., :2].clone()
        normed_grid = utils.normalize_coords(grid, self.h, self.w)
        init_maps = self.depthmaps[ids][:, None]
        sampled_maps = F.grid_sample(
            init_maps, normed_grid[None], align_corners=True, mode='nearest')
        sampled_maps = sampled_maps.squeeze(1)
        return sampled_maps

    def get_pixel_depth_gradient(self, pixels, fids, original=False, scale=True):
        if original:
            if self.read_done:
                sample_frames = self.depth_grad_maps[fids].permute(
                    0, 3, 1, 2)
                normed_px = utils.normalize_coords(
                    pixels, self.h, self.w)[:, None]
                return F.grid_sample(sample_frames, normed_px, align_corners=True, mode='nearest').permute(0, 2, 3, 1)
            sample_frames = self.depthmaps[fids][..., None].permute(
                0, 3, 1, 2)
        else:
            sample_frames = self.depthmem.depthmaps[fids][..., None].permute(
                0, 3, 1, 2)

        if scale:
            scaled_pixels = pixels * torch.tensor([self.depthmaps.shape[-1]/float(
                self.w), self.depthmaps.shape[-2]/float(self.h)], device=self.device)
        else:
            scaled_pixels = pixels

        pix_l = scaled_pixels.clone()
        pix_l[..., 0] = torch.clamp(
            pix_l[..., 0]-1, 0, self.depthmaps.shape[-1]-1)

        pix_u = scaled_pixels.clone()
        pix_u[..., 1] = torch.clamp(
            pix_u[..., 1]-1, 0, self.depthmaps.shape[-2]-1)

        pix_c = utils.normalize_coords(
            scaled_pixels, self.depthmaps.shape[-2], self.depthmaps.shape[-1])[:, None]
        pix_l = utils.normalize_coords(
            pix_l, self.depthmaps.shape[-2], self.depthmaps.shape[-1])[:, None]
        pix_u = utils.normalize_coords(
            pix_u, self.depthmaps.shape[-2], self.depthmaps.shape[-1])[:, None]

        depth_c = F.grid_sample(
            sample_frames, pix_c, align_corners=True, mode='nearest').permute(0, 2, 3, 1)
        depth_l = F.grid_sample(
            sample_frames, pix_l, align_corners=True, mode='nearest').permute(0, 2, 3, 1)
        depth_u = F.grid_sample(
            sample_frames, pix_u, align_corners=True, mode='nearest').permute(0, 2, 3, 1)

        x_grad = (depth_l - depth_c)
        y_grad = (depth_u - depth_c)
        return torch.cat([x_grad, y_grad], dim=-1)  # [1, 1, w*h, 2]

    def project(self, x, return_depth=False):
        '''
        perspective projection
        :param x: [..., 3]
        :param return_depth: if returning depth
        :return: pixel_coords in image space [..., 2], depth [..., 1]
        '''
        depth = x[..., -1:]
        x = x[..., :2] / depth
        x = x * self.f
        x[..., 0] += self.w / 2.0
        x[..., 1] += self.h / 2.0
        if return_depth:
            return x, depth
        else:
            return x

    def unproject(self, pixels, depths):
        '''
        perspective unprojection
        :param pixels: [..., 2] pixel coordinates (unnormalized), in -w, w, -h, h / 2
        :param depths: [..., 1]
        :return: 3d locations in normalized space [..., 3]
        '''
        assert pixels.shape[-1] in [2, 3]
        assert pixels.ndim == depths.ndim
        px = pixels.clone()
        px[..., 0] -= self.w / 2.0
        px[..., 1] -= self.h / 2.0
        px = px / self.f
        px = px * depths
        return torch.cat([px, depths], dim=-1)

    def get_in_range_mask(self, x, max_padding=0):
        mask = (x[..., 0] >= -max_padding) * \
               (x[..., 0] <= self.w - 1 + max_padding) * \
               (x[..., 1] >= -max_padding) * \
               (x[..., 1] <= self.h - 1 + max_padding)
        return mask

    def sample_3d_pts_for_pixels(self, pixels, fids, return_depth=False, max_batch=16, original=False):
        '''
        stratified sampling
        sample points on ray for each pixel
        :param pixels: [n_imgs, n_pts, 2]
        :param fid: [n_imgs]
        :param return_depth: True or False
        :return: sampled 3d locations [n_imgs, n_pts, n_samples, 3]
        '''
        if original:
            sample_frames = self.depthmaps[fids][..., None].permute(
                0, 3, 1, 2).clone()
        else:
            sample_frames = self.depthmem.depthmaps[fids][..., None].permute(
                0, 3, 1, 2).clone()

        sample_grid = utils.normalize_coords(pixels, self.h, self.w)
        if sample_grid.ndim == 3:
            sample_grid = sample_grid[:, :, None]

        sample_grid = torch.clamp(sample_grid, -1, 1)
        depths = F.grid_sample(sample_frames, sample_grid,
                               align_corners=True, mode='nearest').permute(0, 2, 3, 1)

        pixels_expand = pixels[:, :, None, :].expand(-1, -1, 1, -1)

        # [n_imgs, n_pts, n_samples, 3] default n_samples = 1
        x = self.unproject(pixels_expand, depths)
        if return_depth:
            return x, depths
        else:
            return x

    def get_prediction_one_way(self, x, id, inverse=False):
        '''
        mapping 3d points from local to canonical or from canonical to local (inverse=True)
        :param x: [n_imgs, n_pts, n_samples, 3]
        :param id: [n_imgs, ]
        :param inverse: True or False
        :return: [n_imgs, n_pts, n_samples, 3]
        '''
        t = self.time_steps[id]  # [n_imgs, 1]

        if inverse:
            out = self.deform_nvp.inverse(t, None, x)
        else:
            out = self.deform_nvp.forward(t, None, x)

        return out  # [n_imgs, n_pts, n_samples, 3]

    def get_predictions(self, x1, id1, id2, return_canonical=False):
        '''
        mapping 3d points from one frame to another frame
        :param x1: [n_imgs, n_pts, n_samples, 3]
        :param id1: [n_imgs,]
        :param id2: [n_imgs,]
        :return: [n_imgs, n_pts, n_samples, 3]
        '''
        x1_canonical = self.get_prediction_one_way(x1, id1)
        x2_pred = self.get_prediction_one_way(x1_canonical, id2, inverse=True)
        if return_canonical:
            return x2_pred, x1_canonical
        else:
            return x2_pred  # [n_imgs, n_pts, n_samples, 3]

    def get_pred_depths_for_pixels(self, ids, pixels):
        '''
        :param ids: list [n_imgs,]
        :param pixels: [n_imgs, n_pts, 2]
        :return: pred_depths: [n_imgs, n_pts, 1]
        '''
        xs_samples, pxs_depths_samples = self.sample_3d_pts_for_pixels(
            pixels, return_depth=True, fids=ids)
        # xs_canonical_samples = self.get_prediction_one_way(xs_samples, ids)
        # out = self.get_blending_weights(xs_canonical_samples)
        # pred_depths = torch.sum(out['weights'].unsqueeze(-1) * pxs_depths_samples, dim=-2)
        # x2_pred = self.get_prediction_one_way(x1_canonical, id2, inverse=True)
        pred_depths = xs_samples[..., -1]
        return pred_depths  # [n_imgs, n_pts, 1]

    def get_correspondences_for_pixels(self, ids1, px1s, ids2,
                                       return_depth=False,
                                       use_max_loc=False):
        '''
        get correspondences for pixels in one frame to another frame
        :param ids1: [num_imgs]
        :param px1s: [num_imgs, num_pts, 2]
        :param ids2: [num_imgs]
        :param return_depth: if returning the depth of the mapped point in the target frame
        :param use_max_loc: if using only the sample with the maximum blending weight to
                            compute the corresponding location rather than doing over composition.
                            set to True leads to better results on occlusion boundaries,
                            by default it is set to True for inference.

        :return: px2s_pred: [num_imgs, num_pts, 2], and optionally depth: [num_imgs, num_pts, 1]
        '''
        # [n_pair, n_pts, n_samples, 3]
        x1s_samples = self.sample_3d_pts_for_pixels(px1s, fids=ids1)
        x2s_proj_samples, xs_canonical_samples = self.get_predictions(
            x1s_samples, ids1, ids2, return_canonical=True)
        # out = self.get_blending_weights(xs_canonical_samples)  # [n_imgs, n_pts, n_samples]
        # if use_max_loc:
        #     blending_weights = out['weights']
        #     indices = torch.max(blending_weights, dim=-1, keepdim=True)[1]
        #     x2s_pred = torch.gather(x2s_proj_samples, 2, indices[..., None].repeat(1, 1, 1, 3)).squeeze(-2)
        #     return self.project(x2s_pred, return_depth=return_depth)
        # else:
        #     x2s_pred = torch.sum(out['weights'].unsqueeze(-1) * x2s_proj_samples, dim=-2)
        #     return self.project(x2s_pred, return_depth=return_depth)
        x2s_pred = x2s_proj_samples.squeeze(-2)
        return self.project(x2s_pred, return_depth=return_depth)

    def get_correspondences_and_occlusion_masks_for_pixels(self, ids1, px1s, ids2,
                                                           return_depth=False,
                                                           use_max_loc=False,
                                                           depth_err=0.02):
        px2s_pred, depth_proj = self.get_correspondences_for_pixels(ids1, px1s, ids2,
                                                                    return_depth=True,
                                                                    use_max_loc=use_max_loc)

        px2s_pred_samples, px2s_pred_depths_samples = self.sample_3d_pts_for_pixels(
            px2s_pred, return_depth=True, fids=ids2)
        # xs_canonical_samples = self.get_prediction_one_way(px2s_pred_samples, ids2)
        # out = self.get_blending_weights(xs_canonical_samples)
        # weights = out['weights']
        # eps = 1.1 * (self.args.max_depth - self.args.min_depth) / self.args.num_samples_ray
        # mask_zero = px2s_pred_depths_samples.squeeze(-1) >= (depth_proj.expand(-1, -1, self.args.num_samples_ray)) - eps)
        # weights[mask_zero] = 0.
        # occlusion_score = weights.sum(dim=-1, keepdim=True)

        # zero = no occulusion
        occlusion_score = px2s_pred_depths_samples.squeeze(
            -1) <= (depth_proj - depth_err)
        occlusion_score = occlusion_score.float()
        if return_depth:
            return px2s_pred, occlusion_score, depth_proj
        else:
            return px2s_pred, occlusion_score

    def gradient_loss(self, pred, gt, weight=None):
        pred_grad = pred[..., 1:, :] - pred[..., :-1, :]
        gt_grad = gt[..., 1:, :] - gt[..., :-1, :]
        if weight is not None:
            weight_grad = weight[..., 1:, :] * weight[..., :-1, :]
        else:
            weight_grad = None
        loss = masked_l1_loss(pred_grad, gt_grad, weight_grad)
        return loss

    def compute_match_losses(self,
                             batch,
                             step,
                             w_depth=100,
                             w_smooth=10,
                             write_logs=True,
                             return_data=False,
                             log_prefix='loss',
                             ):

        max_padding = self.args.max_padding

        ids1 = batch['ids1'].numpy()
        ids2 = batch['ids2'].numpy()
        px1s = batch['pts1'].to(self.device)
        px2s = batch['pts2'].to(self.device)

        # px1s: [n_pairs, npts, 2] -> x1s_samples: [n_imgs, npts, n_samples, 3]
        x1s_samples = self.sample_3d_pts_for_pixels(px1s, return_depth=False, fids=ids1)
        _, depth2 = self.sample_3d_pts_for_pixels(px2s, return_depth=True, fids=ids2)
        _, origin_depth2 = self.sample_3d_pts_for_pixels(px2s, return_depth=True, fids=ids2, original=True)

        # get depth gradients and original depth gradients
        local_grad1 = self.get_pixel_depth_gradient(px1s, ids1, original=True, scale=True)
        opt_grad1 = self.get_pixel_depth_gradient(px1s, ids1, original=False, scale=True)
        local_grad2 = self.get_pixel_depth_gradient(px2s, ids2, original=True, scale=True)
        opt_grad2 = self.get_pixel_depth_gradient(px2s, ids2, original=False, scale=True)

        depth2 = depth2.squeeze()
        origin_depth2 = origin_depth2.squeeze()

        try:
            x2s_proj_samples, x1s_canonical_samples = self.get_predictions(
                x1s_samples, ids1, ids2, return_canonical=True)
        except Exception as e:
            print(e)
            pdb.set_trace()
            print("mindepth= ", self.depthmem.depthmaps.min().item())

        # get the projected 2d points of frame2
        x2s_pred = x2s_proj_samples.squeeze(-2)
        px2s_proj, px2s_proj_depths = self.project(x2s_pred, return_depth=True)

        # get the mask of the projected points in the range of frame2
        mask = self.get_in_range_mask(px2s_proj, max_padding)

        # compute the flow loss
        if mask.sum() > 0:
            flow_diff = abs(px2s_proj - px2s)
            if step > 2000:
                inliers = abs(flow_diff) < 2*torch.median(abs(flow_diff))
                inliers = inliers.all(-1)
                flow_diff = flow_diff[inliers]

            optical_flow_loss = torch.mean(flow_diff)
        else:
            optical_flow_loss = torch.tensor(0.).to(self.device)

        # compute the depth loss of between frame2 and its opt depth
        depth_diff = (px2s_proj_depths.squeeze() - depth2).abs()
        if step > 2000:
            depth_diff = depth_diff[inliers]
        depth_pred_loss = torch.mean(depth_diff)

        # compute the depth gradient loss
        grad_diff_1 = (local_grad1 - opt_grad1)   # frame1
        grad_diff_2 = (local_grad2 - opt_grad2)   # frame2
        depth_grad_loss = torch.mean(torch.norm(grad_diff_1, dim=-1)) + torch.mean(torch.norm(grad_diff_2, dim=-1))

        # compute the depth consistency loss between frame2 and its original depth
        depth_bias = torch.mean(depth2 - origin_depth2)
        depth_consist_loss = torch.mean(abs(px2s_proj_depths.squeeze() - origin_depth2))

        loss = optical_flow_loss + \
            w_depth * depth_pred_loss + \
            w_smooth*0.1 * depth_consist_loss + \
            w_smooth * depth_grad_loss

        if torch.isnan(loss) or abs(px2s_proj_depths[mask]).min() < 1e-3:
            pdb.set_trace()

        if write_logs:
            self.scalars_to_log['{}/Loss_match'.format(log_prefix)] = loss.item()
            self.scalars_to_log['{}/loss_flow_match'.format(log_prefix)] = optical_flow_loss.item()
            self.scalars_to_log['{}/loss_depth_match'.format(log_prefix)] = depth_pred_loss.item()
            self.scalars_to_log['{}/loss_depth_consist_match'.format(log_prefix)] = depth_consist_loss.item()
            self.scalars_to_log['{}/loss_depth_grad_match'.format(log_prefix)] = depth_grad_loss.item()
            self.scalars_to_log['{}/min_depth_match'.format(log_prefix)] = abs(px2s_proj_depths).min().item()
            self.scalars_to_log['{}/loss_depth_bias_match'.format(log_prefix)] = depth_bias.item()

        data = {'ids1': ids1,
                'ids2': ids2,
                'x1s': x1s_samples,
                'x2s_pred': x2s_pred,
                'xs_canonical': x1s_canonical_samples,
                'mask': mask,
                'px2s_proj': px2s_proj,
                'px2s_proj_depths': px2s_proj_depths,
                }
        if return_data:
            return loss, data
        else:
            return loss

    def compute_flow_losses(self,
                            batch,
                            step,
                            w_smooth=10,
                            w_depth=100,
                            w_flow_grad=0.01,
                            write_logs=True,
                            return_data=False,
                            log_prefix='loss',
                            ):

        max_padding = self.args.max_padding  # default 0

        ids1 = batch['ids1'].numpy()
        ids2 = batch['ids2'].numpy()
        px1s = batch['pts1'].to(self.device)
        px2s = batch['pts2'].to(self.device)
        weights = batch['weights'].to(self.device)

        # px1s: [n_pairs, npts, 2] -> x1s_samples: [n_imgs, npts, n_samples, 3]
        x1s_samples = self.sample_3d_pts_for_pixels(px1s, return_depth=False, fids=ids1)
        _, depth2 = self.sample_3d_pts_for_pixels(px2s, return_depth=True, fids=ids2)
        _, origin_depth2 = self.sample_3d_pts_for_pixels(px2s, return_depth=True, fids=ids2, original=True)

        # get depth gradients and original depth gradients
        local_grad1 = self.get_pixel_depth_gradient(px1s, ids1, original=True, scale=True).squeeze()
        opt_grad1 = self.get_pixel_depth_gradient(px1s, ids1, original=False, scale=True).squeeze()
        local_grad2 = self.get_pixel_depth_gradient(px2s, ids2, original=True, scale=True).squeeze()
        opt_grad2 = self.get_pixel_depth_gradient(px2s, ids2, original=False, scale=True).squeeze()

        depth2 = depth2.squeeze()
        origin_depth2 = origin_depth2.squeeze()

        # get the 3d points of frame2
        x2s_proj_samples, x1s_canonical_samples = self.get_predictions(x1s_samples, ids1, ids2, return_canonical=True)

        # get the projected 2d points of frame2
        x2s_pred = x2s_proj_samples.squeeze(-2)
        px2s_proj, px2s_proj_depths = self.project(x2s_pred, return_depth=True)

        # get the mask of the projected points in the range of frame2
        mask = self.get_in_range_mask(px2s_proj, max_padding)
        zero_depth = (px2s_proj_depths.abs() < 1e-2).squeeze()
        mask = mask * ~zero_depth

        # compaute the flow loss and flow gradient loss
        if mask.sum() > 0:
            optical_flow_loss = masked_l1_loss(px2s_proj[mask], px2s[mask], weights[mask], normalize=False)
            optical_flow_grad_loss = self.gradient_loss(px2s_proj[mask], px2s[mask], weights[mask])
        else:
            optical_flow_loss = torch.tensor(0.).to(self.device)
            optical_flow_grad_loss = torch.tensor(0.).to(self.device)

        # compute the depth loss of between frame2 and its opt depth
        depth_diff = (px2s_proj_depths.squeeze() - depth2).abs()
        depth_diff = depth_diff[~zero_depth]
        depth_pred_loss = torch.mean(depth_diff.abs())

        # compute the depth gradient loss
        grad_diff_1 = (local_grad1 - opt_grad1)   # frame1
        grad_diff_2 = (local_grad2 - opt_grad2)   # frame2
        grad_diff_1[~mask] *= 0.1
        grad_diff_2[~mask] *= 0.1
        depth_grad_loss = torch.mean(torch.norm(grad_diff_1, dim=-1)) + torch.mean(torch.norm(grad_diff_2, dim=-1))

        # compute the depth consistency loss between frame2 and its original depth
        depth_consist_loss = torch.mean(abs(px2s_proj_depths.squeeze() - origin_depth2))

        # final loss
        loss = optical_flow_loss + \
            w_flow_grad * optical_flow_grad_loss + \
            w_smooth * depth_grad_loss + \
            w_smooth * depth_consist_loss + \
            w_depth * depth_pred_loss

        outlier_ratio = (~mask).sum().item()/(mask.numel())

        if torch.isnan(loss) or (loss > 100 and step > 2000) or (outlier_ratio > 0.1 and step > 2000):
            print("out")
            outdir = Path("outlier")
            outdir.mkdir(exist_ok=True)
            for i in range(0, len(ids1), 10):
                id1 = ids1[i]
                id2 = ids2[i]
                image1 = self.images[id1].detach().cpu().numpy().copy()*255
                image2 = self.images[id2].detach().cpu().numpy().copy()*255
                image1 = image1.astype(np.uint8)
                image2 = image2.astype(np.uint8)
                px1s_ = px1s[i].detach().cpu().numpy()
                px2s_ = px2s[i].detach().cpu().numpy()
                pred_px2s_ = px2s_proj[i].detach().cpu().numpy()
                colors = plt.cm.hsv(np.linspace(0, 1, len(px2s_)))[..., :3]*255
                colors = colors.astype(np.uint8)
                for px, color in zip(px1s_, colors):
                    cv2.circle(image1, (int(px[0]), int(px[1])), 3, color.tolist(), -1)
                for px, color in zip(px2s_, colors):
                    cv2.circle(image2, (int(px[0]), int(px[1])), 3, color.tolist(), -1)
                inliers = mask[i].detach().cpu().numpy()
                for px, pred_px, color in zip(px2s_[inliers], pred_px2s_[inliers], colors[inliers]):
                    cv2.line(image2, (int(px[0]), int(px[1])), (int(pred_px[0]), int(pred_px[1])), color.tolist(), 1)
                for px, color in zip(px2s_[~inliers], colors[~inliers]):
                    cv2.circle(image2, (int(px[0]), int(px[1])), 5, (0, 0, 255), 1)

            pdb.set_trace()

        if write_logs:
            self.scalars_to_log['{}/Loss_dense'.format(log_prefix)] = loss.item()
            self.scalars_to_log['{}/loss_flow_dense'.format(log_prefix)] = optical_flow_loss.item()
            self.scalars_to_log['{}/loss_flow_gradient'.format(log_prefix)] = optical_flow_grad_loss.item()
            self.scalars_to_log['{}/loss_depth_dense'.format(log_prefix)] = depth_pred_loss.item()
            self.scalars_to_log['{}/loss_depth_grad_dense'.format(log_prefix)] = depth_grad_loss.item()
            self.scalars_to_log['{}/loss_depth_consist_dense'.format(log_prefix)] = depth_consist_loss.item()
            self.scalars_to_log['{}/min_depth_dense'.format(log_prefix)] = abs(px2s_proj_depths).min().item()
            self.scalars_to_log['{}/outlier_ratio'.format(log_prefix)] = outlier_ratio

        data = {'ids1': ids1,
                'ids2': ids2,
                'x1s': x1s_samples,
                'x2s_pred': x2s_pred,
                'xs_canonical': x1s_canonical_samples,
                'mask': mask,
                'px2s_proj': px2s_proj,
                'px2s_proj_depths': px2s_proj_depths,
                }
        if return_data:
            return loss, data
        else:
            return loss

    def weight_scheduler(self, step, start_step, w, min_weight, max_weight):
        if step <= start_step:
            weight = 0.0
        else:
            weight = w * (step - start_step)
        weight = np.clip(weight, a_min=min_weight, a_max=max_weight)
        return weight

    def train_one_step(self, step, batch):
        self.deform_nvp.train()
        self.step = step
        self.scalars_to_log = {}
        self.optimizer.zero_grad()

        longterm_end = 24000

        if len(batch) != 2:
            longterm_end = 0
            batch = {'gm': batch}
        if "simple" in batch.keys():
            batch["gm"] = batch["simple"]

        w_flow_grad = self.weight_scheduler(step, 0, 1./500000, 0, 0.1)

        loss_flow, data_flow = self.compute_flow_losses(batch['gm'],
                                                        step,
                                                        w_smooth=1.,
                                                        w_depth=100,
                                                        w_flow_grad=w_flow_grad,
                                                        return_data=True)

        if step < longterm_end and ('long' in batch.keys()) and batch['long']['ids1'][0] >= 0:
            loss_match = self.compute_match_losses(batch['long'],
                                                   step,
                                                   w_smooth=self.args.smooth_weight,
                                                   w_depth=100,
                                                   w_flow_grad=w_flow_grad,
                                                   return_data=False)
        else:
            loss_match = 0

        match_overfit_iters = 1000
        if step < match_overfit_iters:
            loss = loss_flow + loss_match
        elif step < match_overfit_iters*2:
            loss = loss_flow + loss_match*0.1
        else:
            loss = loss_flow + loss_match*0.01

        self.scalars_to_log['loss/Loss'] = loss.item()

        if torch.isnan(loss):
            pdb.set_trace()

        loss.backward()

        # gradient clipping
        if self.args.grad_clip > 0 and self.depthmem.depthmaps.grad.norm() > self.args.grad_clip:
            torch.nn.utils.clip_grad_norm_(self.depthmem.parameters(), self.args.grad_clip)

        self.optimizer.step()
        self.scheduler.step()

        self.scalars_to_log['loss/min_depth'] = self.depthmem.depthmaps.min().item()
        self.scalars_to_log['loss/max_depth'] = self.depthmem.depthmaps.max().item()
        self.scalars_to_log['loss/depth<0_ratio'] = (self.depthmem.depthmaps < 0).float().mean().item()

        try:
            self.scalars_to_log['loss/min_depth_pos'] = torch.where(
                self.depthmem.depthmaps == self.depthmem.depthmaps.min())[0][0].item()
        except:
            pdb.set_trace()

        self.scalars_to_log['lr'] = self.optimizer.param_groups[0]['lr']

        self.ids1 = data_flow['ids1']
        self.ids2 = data_flow['ids2']
        
        return loss.item()

    def sample_pts_within_mask(self, mask, num_pts, return_normed=False, seed=None,
                               use_mask=False, reverse_mask=False, regular=False, interval=10):
        rng = np.random.RandomState(seed) if seed is not None else np.random
        if use_mask:
            if reverse_mask:
                mask = ~mask
            kernel = torch.ones(7, 7, device=self.device)
            mask = morph.erosion(
                mask.float()[None, None], kernel).bool().squeeze()  # Erosion
        else:
            mask = torch.ones_like(self.grid[..., 0], dtype=torch.bool)

        if regular:
            coords = self.grid[::interval, ::interval,
                               :2][mask[::interval, ::interval]].clone()
        else:
            coords_valid = self.grid[mask][..., :2].clone()
            rand_inds = rng.choice(
                len(coords_valid), num_pts, replace=(num_pts > len(coords_valid)))
            coords = coords_valid[rand_inds]

        coords_normed = utils.normalize_coords(coords, self.h, self.w)
        if return_normed:
            return coords, coords_normed
        else:
            return coords  # [num_pts, 2]

    def vis_pairwise_correspondences(self, ids=None, num_pts=200, use_mask=False, use_max_loc=True,
                                     reverse_mask=False, regular=True, interval=20):
        if ids is not None:
            id1, id2 = ids
        else:
            id1 = self.ids1[0]
            id2 = self.ids2[0]

        px1s = self.sample_pts_within_mask(self.masks[id1], num_pts, seed=1234,
                                           use_mask=use_mask, reverse_mask=reverse_mask,
                                           regular=regular, interval=interval)
        num_pts = len(px1s)

        with torch.no_grad():
            px2s_pred, occlusion_score = \
                self.get_correspondences_and_occlusion_masks_for_pixels([id1], px1s[None], [id2],
                                                                        use_max_loc=use_max_loc)
            px2s_pred = px2s_pred[0]
            mask = occlusion_score > self.args.occlusion_th

        kp1 = px1s.detach().cpu().numpy()
        kp2 = px2s_pred.detach().cpu().numpy()
        img1 = self.images[id1].cpu().numpy()
        img2 = self.images[id2].cpu().numpy()
        mask = mask[0].squeeze(-1).cpu().numpy()
        out = utils.drawMatches(img1, img2, kp1, kp2,
                               num_vis=num_pts, mask=mask)
        out = cv2.putText(out, str(id2 - id1), org=(30, 50), fontScale=1, color=(255, 255, 255),
                          fontFace=cv2.FONT_HERSHEY_SIMPLEX, thickness=2)
        out = utils.uint82float(out)
        return out

    def plot_correspondences_for_pixels(self, query_kpt, query_id, num_pts=200,
                                        vis_occlusion=False,
                                        occlusion_th=0.95,
                                        use_max_loc=False,
                                        radius=2,
                                        return_kpts=False):
        frames = []
        kpts = []
        with torch.no_grad():
            img_query = self.images[query_id].cpu().numpy()
            for id in range(0, self.num_imgs):
                if vis_occlusion:
                    if id == query_id:
                        kp_i = query_kpt
                        occlusion_score = torch.zeros_like(query_kpt[..., :1])
                    else:
                        kp_i, occlusion_score = \
                            self.get_correspondences_and_occlusion_masks_for_pixels([query_id], query_kpt[None], [id],
                                                                                    use_max_loc=use_max_loc)
                        kp_i = kp_i[0]
                        occlusion_score = occlusion_score[0]

                    mask = occlusion_score > occlusion_th
                    kp_i = torch.cat([kp_i, mask.float()], dim=-1)
                    mask = mask.squeeze(-1).cpu().numpy()
                else:
                    if id == query_id:
                        kp_i = query_kpt
                    else:
                        kp_i = self.get_correspondences_for_pixels([query_id], query_kpt[None], [id],
                                                                   use_max_loc=use_max_loc)[0]
                    mask = None
                img_i = self.images[id].cpu().numpy()
                out = utils.drawMatches(img_query, img_i, query_kpt.cpu().numpy(), kp_i.cpu().numpy(),
                                       num_vis=num_pts, mask=mask, radius=radius)
                frames.append(out)
                kpts.append(kp_i)
        kpts = torch.stack(kpts, dim=0)
        if return_kpts:
            return frames, kpts
        return frames

    def eval_video_correspondences(self, query_id, pts=None, num_pts=200, seed=1234, use_mask=False,
                                   mask=None, reverse_mask=False, vis_occlusion=False, occlusion_th=0.99,
                                   use_max_loc=False, regular=True,
                                   interval=10, radius=2, return_kpts=False):
        with torch.no_grad():
            if mask is not None:
                mask = torch.from_numpy(mask).bool().to(self.device)
            else:
                mask = self.masks[query_id]

            if pts is None:
                x_0 = self.sample_pts_within_mask(mask, num_pts, seed=seed, use_mask=use_mask,
                                                  reverse_mask=reverse_mask, regular=regular, interval=interval)
                num_pts = 1e7 if regular else num_pts
            else:
                x_0 = torch.from_numpy(pts).float().to(self.device)
            return self.plot_correspondences_for_pixels(x_0, query_id, num_pts=num_pts,
                                                        vis_occlusion=vis_occlusion,
                                                        occlusion_th=occlusion_th,
                                                        use_max_loc=use_max_loc,
                                                        radius=radius, return_kpts=return_kpts)

    def get_pred_depth_maps(self, ids, chunk_size=40000):
        grid = self.grid[..., :2].reshape(-1, 2).clone()
        pred_depths = []
        for id in ids:
            depth_map = []
            for coords in torch.split(grid, split_size_or_sections=chunk_size, dim=0):
                depths_chunk = self.get_pred_depths_for_pixels([id], coords[None])[
                    0]
                depths_chunk = torch.nan_to_num(depths_chunk)
                depth_map.append(depths_chunk)
            depth_map = torch.cat(depth_map, dim=0).reshape(self.h, self.w)
            pred_depths.append(depth_map)
        pred_depths = torch.stack(pred_depths, dim=0)
        return pred_depths  # [n, h, w]

    def get_pred_flows(self, ids1, ids2, chunk_size=40000, use_max_loc=False, return_original=False):
        grid = self.grid[..., :2].reshape(-1, 2).clone()
        flows = []
        for id1, id2 in zip(ids1, ids2):
            flow_map = []
            for coords in torch.split(grid, split_size_or_sections=chunk_size, dim=0):
                with torch.no_grad():
                    flows_chunk = self.get_correspondences_for_pixels([id1], coords[None], [id2],
                                                                      use_max_loc=use_max_loc)[0]
                    flow_map.append(flows_chunk)
            flow_map = torch.cat(flow_map, dim=0).reshape(self.h, self.w, 2)
            flow_map = (flow_map - self.grid[..., :2]).cpu().numpy()
            flows.append(flow_map)
        flows = np.stack(flows, axis=0)
        flow_imgs = utils.flow_to_image(flows)
        if return_original:
            return flow_imgs, flows
        else:
            return flow_imgs  # [n, h, w, 3], numpy arra

    def log(self, writer, step):
        if step % self.args.i_log == 0:
            for k in self.scalars_to_log.keys():
                writer.add_scalar(k, self.scalars_to_log[k], step)

        if step % self.args.i_img == 0:
            # flow
            flows = self.get_pred_flows(self.ids1[0:1], self.ids2[0:1], chunk_size=self.args.chunk_size)[0]
            writer.add_image('flow', flows, step, dataformats='HWC')

            # correspondences
            out_trained = self.vis_pairwise_correspondences()
            out_fix_10 = self.vis_pairwise_correspondences(ids=(0, min(self.num_imgs // 10, 10)))
            out_fix_half = self.vis_pairwise_correspondences(ids=(0, self.num_imgs // 2))
            out_fix_full = self.vis_pairwise_correspondences(ids=(0, self.num_imgs - 1))

            writer.add_image('correspondence/trained', out_trained, step, dataformats='HWC')
            writer.add_image('correspondence/fix_10', out_fix_10, step, dataformats='HWC')
            writer.add_image('correspondence/fix_half', out_fix_half, step, dataformats='HWC')
            writer.add_image('correspondence/fix_whole', out_fix_full, step, dataformats='HWC')

        if step % self.args.i_save == 0 and step > 0:
            # save checkpoints
            os.makedirs(self.out_dir, exist_ok=True)
            print(f"Saving checkpoints at {step} to {self.out_dir}...")
            fpath = os.path.join(self.out_dir, 'model_{:06d}.pth'.format(step))
            self.save_model(fpath)

            vis_dir = os.path.join(self.out_dir, 'vis')
            os.makedirs(vis_dir, exist_ok=True)
            print(f"Saving visualizations to {vis_dir}...")
            if self.with_mask:
                video_correspondences = self.eval_video_correspondences(0,
                                                                        use_mask=True,
                                                                        vis_occlusion=self.args.vis_occlusion,
                                                                        use_max_loc=self.args.use_max_loc,
                                                                        occlusion_th=self.args.occlusion_th)
                imageio.mimwrite(os.path.join(vis_dir, '{}_corr_foreground_{:06d}.mp4'.format(self.seq_name, step)),
                                 video_correspondences,
                                 quality=8, fps=10)
                video_correspondences = self.eval_video_correspondences(0,
                                                                        use_mask=True,
                                                                        reverse_mask=True,
                                                                        vis_occlusion=self.args.vis_occlusion,
                                                                        use_max_loc=self.args.use_max_loc,
                                                                        occlusion_th=self.args.occlusion_th)
                imageio.mimwrite(os.path.join(vis_dir, '{}_corr_background_{:06d}.mp4'.format(self.seq_name, step)),
                                 video_correspondences,
                                 quality=8, fps=10)
            else:
                video_correspondences = self.eval_video_correspondences(0,
                                                                        vis_occlusion=self.args.vis_occlusion,
                                                                        use_max_loc=self.args.use_max_loc,
                                                                        occlusion_th=self.args.occlusion_th)
                imageio.mimwrite(os.path.join(vis_dir, '{}_corr_{:06d}.mp4'.format(self.seq_name, step)),
                                 video_correspondences,
                                 quality=8, fps=10)

    def save_model(self, filename):
        to_save = {'optimizer': self.optimizer.state_dict(),
                   'scheduler': self.scheduler.state_dict(),
                   'deform_nvp': de_parallel(self.deform_nvp).state_dict(),

                   'num_imgs': self.num_imgs
                   }
        if self.args.opt_depth:
            to_save['depth_mem'] = de_parallel(self.depthmem).state_dict()

        torch.save(to_save, filename)

    def load_model(self, filename, load_opt=True, load_scheduler=True):
        to_load = torch.load(filename)

        if load_opt:
            self.optimizer.load_state_dict(to_load['optimizer'])
        if load_scheduler:
            self.scheduler.load_state_dict(to_load['scheduler'])
        num_imgs = to_load['num_imgs']
        self.num_imgs = num_imgs
        self.deform_nvp = NVPnonlin(n_layers=6,
                                    n_frames=num_imgs,
                                    feature_dim=self.deform_nvp.feature_dim,
                                    t_dim=16,
                                    multires=self.deform_nvp.multires,
                                    base_res=self.deform_nvp.base_res,
                                    net_layer=self.deform_nvp.net_layer,
                                    bound=self.deform_nvp.bound,
                                    device=self.device).to(self.device)

        self.deform_nvp.load_state_dict(to_load['deform_nvp'])

        if self.args.opt_depth:
            self.depthmaps = torch.zeros(
                (num_imgs, self.depthmaps.shape[1], self.depthmaps.shape[2]), device=self.device)
            self.depthmem = DepthMem(
                self.args, self.depthmaps, self.device).to(self.device)
            self.depthmem.load_state_dict(to_load['depth_mem'])

    def load_from_ckpt(self, out_folder,
                       load_opt=True,
                       load_scheduler=True,
                       force_latest_ckpt=False):
        '''
        load model from existing checkpoints and return the current step
        :param out_folder: the directory that stores ckpts
        :return: the current starting step
        '''

        # all existing ckpts
        ckpts = []
        if os.path.exists(out_folder):
            ckpts = [os.path.join(out_folder, f)
                     for f in sorted(os.listdir(out_folder)) if f.endswith('.pth')]

        if self.args.ckpt_path is not None and not force_latest_ckpt:
            if os.path.isfile(self.args.ckpt_path):  # load the specified ckpt
                ckpts = [self.args.ckpt_path]

        if len(ckpts) > 0 and not self.args.no_reload:
            fpath = ckpts[self.args.ckpt]
            self.load_model(fpath, load_opt, load_scheduler)
            step = int(fpath[-10:-4])
            print('Reloading from {}, starting at step={}'.format(fpath, step))
        else:
            print('No ckpts found, from scratch...')
            step = 0

        return step
