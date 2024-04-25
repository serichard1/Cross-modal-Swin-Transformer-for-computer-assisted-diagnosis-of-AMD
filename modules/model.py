from .swin_transformer_v2 import SwinTransformerV2
from glob import glob
import torch
import torch.nn as nn
import torch.nn.functional as F


class OCTswap(nn.Module):
    def __init__(
        self
    ):
        super().__init__()
        # self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.out = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1))

    def forward(self, batch_oct):
        x = torch.topk(batch_oct, 1, dim=1).values
        # print(x.shape)
        x= torch.flatten(x, 1)
        # print(x.shape)
        x = self.out(x)

        return x

class PoolHead(nn.Module):
    def __init__(
        self,
        in_features
    ):
        super().__init__()

        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.classi_head = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.GELU(),
            nn.Linear(256, 1))

    def forward(self, attn_cfp, attn_oct, idx_to_keep1_2, idx_to_keep2_1):

        batch, _, _ = attn_cfp.size()

        attn_cfp = attn_cfp[torch.arange(batch).unsqueeze(1), idx_to_keep1_2]
        attn_oct = attn_oct[torch.arange(batch).unsqueeze(1), idx_to_keep2_1]
        attn_cfp = self.avgpool(attn_cfp.transpose(1, 2))
        attn_oct = self.avgpool(attn_oct.transpose(1, 2))

        logits_cfp = torch.flatten(attn_cfp, 1)
        logits_oct = torch.flatten(attn_oct, 1)

        x = torch.cat((logits_cfp, logits_oct), dim=1)
        x = self.classi_head(x)

        return x

class BimodalEncoder(nn.Module):
    def __init__(
        self,
        encoder_cfp,
        encoder_oct
    ):
        super().__init__()

        self.encoder_cfp = encoder_cfp
        self.encoder_oct = encoder_oct

    def forward(self, cfp_imgs, oct_imgs):

        attn_cfp = self.encoder_cfp(cfp_imgs)
        attn_oct = self.encoder_oct(oct_imgs)

        return attn_cfp, attn_oct
    
class UnimodalEncoderv2(nn.Module):
    def __init__(
        self,
        encoder
    ):
        super().__init__()

        self.encoder = encoder
        # self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.classi_head = nn.Sequential(
            nn.Linear(3*768, 128),
            nn.ReLU(),
            nn.Linear(128, 1))

    def forward(self, imgs, npy):
        # print(imgs.shape)
        # print(npy.shape)

        x = self.encoder(imgs)
        # print(x.shape)
        # npy = self.avgpool(npy.transpose(1, 2))
        npy_l = torch.flatten(torch.topk(npy, 1, dim=1).values, 1)
        npy_s = torch.flatten(torch.topk(npy, 1, dim=1, largest=False).values, 1)
        # npy = torch.flatten(npy, 1)
        
        # print(npy.shape)
        x = torch.cat((x, npy_l, npy_s), dim=1)
        # print(x.shape)
        x = self.classi_head(x)
        # print(x.shape)
 
        return x
    
class UnimodalEncoder(nn.Module):
    def __init__(
        self,
        encoder
    ):
        super().__init__()

        self.encoder = encoder
        self.classi_head = nn.Linear(768, 1)

    def forward(self, imgs):

        x = self.encoder(imgs)
        x = self.classi_head(x)

        return x
    
class CrossSightv3(nn.Module):
    def __init__(
        self,
        model,
        encoder_cfp,
        encoder_oct
    ):
        super().__init__()

        self.bimodal_encoder = BimodalEncoder(encoder_cfp,  encoder_oct)
        if model == 'cfp':
            self.pool_head = PoolHead(in_features=2048)
        else:
            self.pool_head = PoolHead(in_features=1536)

    def forward_bimodal_encoder(self, cfp_imgs, oct_imgs):
        return self.bimodal_encoder(cfp_imgs, oct_imgs)
    
    def forward_pool_head(self, attn_cfp, attn_oct, idx_to_keep1_2, idx_to_keep2_1):
        return self.pool_head(attn_cfp, attn_oct, idx_to_keep1_2, idx_to_keep2_1)
    
def build_swin_encoder(model='base', modality='cfp', ckpts_dir='ckpts', crossSight=False, drop_path_rate=0.3):

    if modality == 'cfp':
        in_chans = 3
    elif modality == 'oct':
        in_chans = 19

    if model == 'base':
        embed_dim=128
        depths=[ 2, 2, 18, 2 ]
        num_heads=[ 4, 8, 16, 32 ]
        pretrained_window_sizes=[ 12, 12, 12, 6 ]

    elif model == 'tiny':
        embed_dim=96
        depths=[ 2, 2, 6, 2 ]
        num_heads=[ 3, 6, 12, 24 ]
        pretrained_window_sizes=[0, 0, 0, 0]
    
    encoder = SwinTransformerV2(img_size=256,
                                patch_size=4,
                                in_chans=in_chans,
                                num_classes=1000,
                                embed_dim=embed_dim,
                                depths=depths,
                                num_heads=num_heads,
                                window_size=16,
                                mlp_ratio=4.,
                                qkv_bias=True,
                                drop_rate=0.0,
                                drop_path_rate=drop_path_rate,
                                ape=False,
                                patch_norm=True,
                                use_checkpoint=False,
                                pretrained_window_sizes=pretrained_window_sizes,
                                crossSight=crossSight)
    
    checkpoints = glob(f'{ckpts_dir}/ckpt_model_{modality}_*.pth')
    if len(checkpoints) > 1:
        print(f'WARNING: more than one checkpoint available for the {modality} encoder')
    checkpoint = torch.load(checkpoints[0], map_location='cpu')
    # if modality == 'oct':
    #     checkpoint['model']['patch_embed.proj.weight'] = checkpoint['model']['patch_embed.proj.weight'].repeat(1, 19, 1, 1)
    # encoder.load_state_dict(checkpoint['model'], strict=False)
    encoder.load_state_dict(checkpoint['model'], strict=False)
    print(f'INFO: correctly loaded {checkpoints[0]} for the {modality} encoder')

    return encoder