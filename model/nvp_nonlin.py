import torch
import torch.nn.functional as F
from torch import nn
from model.hash_map import MultiResBiplane


class InterpGrad(torch.autograd.Function):

    @staticmethod
    @torch.jit.script
    def forw_compute(x, y, qx):
        grad_qx = torch.zeros_like(qx)
        grad_x = torch.zeros_like(x)
        grad_y = torch.zeros_like(y)
        out = torch.zeros_like(qx)
        all_xr = x[..., 1:]
        all_xl = x[..., :-1]
        all_yr = y[..., 1:]
        all_yl = y[..., :-1]
        qx = torch.clamp(qx, all_xl[..., 0:1]*0.99, all_xr[..., -1:]*0.99)
        in_range = (qx >= all_xl) & (qx < all_xr)
        xr = all_xr[in_range]
        xl = all_xl[in_range]
        yr = all_yr[in_range]
        yl = all_yl[in_range]
        x_range = xr - xl

        dYdqx = (yr - yl) / x_range
        dYdyr = (qx[..., 0] - xl) / x_range
        dYdyl = (xr - qx[..., 0]) / x_range
        dYdxr = -dYdyr * (yr - yl) / x_range
        dYdxl = -dYdyl * (yr - yl) / x_range

        grad_qx[..., 0] += dYdqx
        grad_x[..., :-1][in_range] += dYdxl
        grad_x[..., 1:][in_range] += dYdxr
        grad_y[..., :-1][in_range] += dYdyl
        grad_y[..., 1:][in_range] += dYdyr
        out[..., 0] = dYdqx * (qx[..., 0] - xl) + yl
        return out, grad_x, grad_y, grad_qx

    @staticmethod
    def forward(ctx, x, y, qx):
        '''
        x: [N, n_samples], should be monotonic
        y: [N, n_samples]
        qx: [N, 1]
        '''
        out, grad_x, grad_y, grad_qx = InterpGrad.forw_compute(x, y, qx)

        ctx.save_for_backward(grad_x, grad_y, grad_qx)

        return out

    @staticmethod
    @torch.jit.script
    def back_compute(grad_out, grad_x, grad_y, grad_qx):
        grad_x = grad_x * grad_out
        grad_y = grad_y * grad_out
        grad_qx = grad_qx * grad_out
        return grad_x, grad_y, grad_qx

    @staticmethod
    def backward(ctx, grad_out):
        grad_x, grad_y, grad_qx = ctx.saved_tensors
        grad_x, grad_y, grad_qx = InterpGrad.back_compute(
            grad_out, grad_x, grad_y, grad_qx)
        return grad_x, grad_y, grad_qx


interp_grad = InterpGrad.apply


class CouplingLayer(nn.Module):
    def __init__(self, map, mask):
        super().__init__()
        self.map = map
        self.register_buffer('mask', mask)

    @staticmethod
    @torch.jit.script
    def get_xy_compute(dxl1, dxl2, dxr1, dxr2, dyl1, dyl2, dyr1, dyr2, kl, kr):
        kl = kl * 2.0  # + 1e-8
        kr = kr * 2.0  # + 1e-8

        xL1 = -dxl1  # alpha3
        xL2 = -dxl1 - dxl2  # alpha2
        yL1 = - dyl1  # beta3
        yL2 = - dyl1 - dyl2  # beta2

        xR1 = dxr1  # alpha4
        xR2 = dxr1 + dxr2  # alpha5
        yR1 = dyr1  # beta4
        yR2 = dyr1 + dyr2  # beta5

        xR3 = xR2 + 1e4  # alpha6
        xL3 = xL2 - 1e4  # alpha1
        yR3 = yR2 + kr*1e4  # beta6
        yL3 = yL2 - kl*1e4  # beta1

        # [alpha1, alpha2, alpha3, alpha4, alpha5, alpha6]
        all_x = torch.cat([xL3, xL2, xL1, xR1, xR2, xR3], dim=-1)
        # [beta1, beta2, beta3, beta4, beta5, beta6]
        all_y = torch.cat([yL3, yL2, yL1, yR1, yR2, yR3], dim=-1)
        return all_x, all_y

    def get_all_xy(self, dxdykk):
        '''
        dxdykk: [N, P, 1, 10], dxl1, dxl2, dxr1, dxr2, dy...., kl, kr, all > 0
        '''

        dxl2, dxl1, dxr1, dxr2, dyl2, dyl1, dyr1, dyr2, kl, kr = torch.split(
            dxdykk, split_size_or_sections=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dim=-1)

        return CouplingLayer.get_xy_compute(dxl1, dxl2, dxr1, dxr2, dyl1, dyl2, dyr1, dyr2, kl, kr)

    def forward(self, input, t_feat):
        '''
        t_feat: [n_imgs, t_dim * 3]
        input: [n_imgs, num_pts, num_samples, 3] local coordinate
        output: [n_imgs, num_pts, num_samples, 3] caconical coordinate
        '''
        # 不变的x, y
        output = input * self.mask

        # 获取插值控制点
        dxdykk = self.map(torch.tanh(input[..., self.mask.bool()]), t_feat)
        all_x, all_y = self.get_all_xy(dxdykk)

        # 插值z
        z = input[..., (1 - self.mask).bool()].reshape(-1, 1)
        z_interp = interp_grad(
            all_x.reshape(-1, 6), all_y.reshape(-1, 6), z).reshape(*output.shape[:-1], 1)
        output[..., (1-self.mask).bool()] = z_interp

        return output

    def inverse(self, output, t_feat):
        '''
        t_feat: [n_imgs, t_dim * 3]
        output: [n_imgs, num_pts, num_samples, 3] caconical coordinate
        input: [n_imgs, num_pts, num_samples, 3] local coordinate
        '''
        input = output * self.mask

        dxdykk = self.map(torch.tanh(output[..., self.mask.bool()]), t_feat)
        all_x, all_y = self.get_all_xy(dxdykk)

        z_interp = output[..., (1 - self.mask).bool()].reshape(-1, 1)
        z = interp_grad(all_y.reshape(-1, 6), all_x.reshape(-1, 6),
                        z_interp).reshape(*output.shape[:-1], 1)
        input[..., (1-self.mask).bool()] = z

        return input


class NVPnonlin(nn.Module):
    def __init__(
            self,
            n_layers,  # coupling layers number
            n_frames,  # frames number
            feature_dim=32,  # spatial feature dimension
            t_dim=8,  # time dimension
            bound=torch.tensor([[-1, -1, -1], [1, 1, 1]]),
            base_res=8,
            net_layer=2,  # the layer number of MLP for predicting control points
            device='cuda',
    ):
        super().__init__()
        self.register_buffer('frames', torch.tensor(n_frames))
        self.register_buffer('bound', bound.to(device))
        self.layer_idx = [i for i in range(n_layers)]

        t_baseres = n_frames // 20
        t_res = [t_baseres, t_baseres*5, t_baseres*13]
        self.t_embeddings = nn.ParameterList()
        for res in t_res:
            self.t_embeddings.append(nn.Parameter(
                torch.randn(1, t_dim, 1, res)*0.001))

        self.coupling_layers = nn.ModuleList()
        pattern = torch.arange(3)  # 生成基础序列 [0, 1, 2]
        self.mask_selection = pattern.repeat(
            (n_layers + 2) // 3)[:n_layers]  # 计算重复并截断

        for i in self.layer_idx:
            # get mask
            constant_mask = torch.ones(3, device=device)
            constant_mask[self.mask_selection[i]] = 0

            # get transformation
            map = MultiResBiplane(feat_dim=feature_dim,
                                  res=[base_res, base_res*8],
                                  t_dim=t_dim,
                                  output_dim=10,
                                  net_layer=net_layer,
                                  )

            self.coupling_layers.append(CouplingLayer(map, constant_mask))

    def get_t_feature(self, t):
        '''
        t: [n_imgs, 1]
        t_feat: [n_imgs, t_dim]
        '''
        t = t * 2 - 1.0
        N = t.shape[0]
        t = t.reshape(1, N, 1, 1)
        zeros = torch.zeros_like(t)
        t = torch.cat([t, zeros], dim=-1)

        t_feat = []
        for featlist in self.t_embeddings:
            t_feat.append(F.grid_sample(
                featlist, t, align_corners=True).squeeze(0).squeeze(-1).T)
        t_feat = torch.cat(t_feat, dim=-1)
        return t_feat

    def forward(self, t, x):
        '''
        t: [n_imgs, 1]
        x: [n_imgs, num_pts, num_samples, 3] local coordinate
        y: [n_imgs, num_pts, num_samples, 3] caconical coordinate
        '''
        t_feat = self.get_t_feature(t)
        x = (x - (self.bound[1] + self.bound[0]) / 2) / \
            ((self.bound[1] - self.bound[0])/2)

        y = x

        for i in self.layer_idx:
            l1 = self.layers1[i]
            y, _ = l1(y, t_feat)
        return y

    def inverse(self, t, y):
        '''
        t: [n_imgs, 1]
        y: [n_imgs, num_pts, num_samples, 3] caconical coordinate
        x: [n_imgs, num_pts, num_samples, 3] local coordinate
        '''
        t_feat = self.get_t_feature(t)

        x = y

        for i in reversed(self.layer_idx):
            l1 = self.layers1[i]
            x, _ = l1.inverse(x, t_feat)

        x = x * ((self.bound[1] - self.bound[0])/2) + \
            (self.bound[1] + self.bound[0]) / 2

        return x
