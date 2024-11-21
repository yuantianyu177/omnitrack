import torch
import torch.nn as nn

class MultiResBiplane(nn.Module):
    def __init__(self, output_dim=2,
                 res=[256],
                 feat_dim=16,
                 t_dim=8,
                 net_layer=2,
                 act=nn.Sigmoid()) -> None:
        super().__init__()
        self.xy_embeddings = nn.ParameterList()
        for r in res:
            self.xy_embeddings.append(nn.Parameter(
                torch.randn(1, feat_dim, r, r)*0.0001))

        input_dim = feat_dim * len(res) + t_dim * 3
        if net_layer == 2:
            self.net = nn.Sequential(
                nn.Linear(input_dim, input_dim // 2),
                nn.LeakyReLU(),
                nn.Linear(input_dim//2, output_dim),
                act,
            )
        elif net_layer == 3:
            self.net = nn.Sequential(
                nn.Linear(input_dim, input_dim // 2),
                nn.LeakyReLU(),
                nn.Linear(input_dim // 2, input_dim // 4),
                nn.LeakyReLU(),
                nn.Linear(input_dim//4, output_dim),
                act,
            )

    def forward(self, coordinates, t_feat):
        '''
        t_feat: [n_imgs, t_dim * 3] 
        coordinates: [n_imgs, num_pts, num_samples, 2] [-1, 1]
        out: [n_imgs, num_pts, num_samples, 10] [0, 1]
        '''
        in_shape = coordinates.shape
        coordinates = coordinates.reshape(-1, 2)
        coordinates = coordinates[None, :, None]

        xy_features = []

        for emb in self.xy_embeddings:
            xy_features.append(torch.nn.functional.grid_sample(emb,
                                                               coordinates,
                                                               mode='bilinear', padding_mode='zeros', align_corners=True).squeeze(0).squeeze(-1).T)
        features = torch.cat(xy_features, dim=-1)

        # [n_imgs, t_dim * 3] -> [n_imgs, num_pts, (num_samples), t_dim] -> [n_imgs * num_pts * (num_samples), t_dim * 3]
        if in_shape[2] > 1:
            t_feat = t_feat[:, None, None].expand(
                -1, in_shape[1], in_shape[2], -1).reshape(-1, t_feat.shape[-1])
        else:
            t_feat = t_feat[:, None].expand(-1, in_shape[1], -
                                            1).reshape(-1, t_feat.shape[-1])

        # concatenate t_feat and features
        features = torch.cat([features, t_feat], dim=-1)

        # [n_imgs * num_pts * (num_samples), t_dim * 3] -> [n_imgs, num_pts, num_samples, t_dim]
        out = self.net(features).reshape(*in_shape[:-1], -1)

        return out
