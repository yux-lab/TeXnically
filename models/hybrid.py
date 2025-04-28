import torch
import torch.nn as nn

from timm.models.vision_transformer import VisionTransformer
from timm.models.vision_transformer_hybrid import HybridEmbed
from timm.models.resnetv2 import ResNetV2
from timm.models.layers import StdConv2dSame
from einops import repeat


class CustomVisionTransformer(VisionTransformer):
    # 继承自 timm 库中的 VisionTransformer，用于构建 Transformer 编码器。
    # 主要修改 forward_features 方法，使位置编码支持动态输入尺寸（非固定大小的图像）。
    def __init__(self, img_size=224, patch_size=16, *args, **kwargs):
        super(CustomVisionTransformer, self).__init__(img_size=img_size, patch_size=patch_size, *args, **kwargs)
        self.height, self.width = img_size # 输入图像大小，支持 tuple 格式，如 (192, 672)
        self.patch_size = patch_size

    def forward_features(self, x):
        B, c, h, w = x.shape # B: batch size, c: 通道数, h/w: 原始图像尺寸
        x = self.patch_embed(x) # 将图像分割成 patch，并做线性变换得到 token（包括 CNN 处理后的）

        cls_tokens = self.cls_token.expand(B, -1, -1)  # 每个样本加一个可学习的 [CLS] token # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)

        # 根据输入图像的大小动态计算位置编码索引（适配可变尺寸）
        h, w = h//self.patch_size, w//self.patch_size
        pos_emb_ind = repeat(torch.arange(h)*(self.width//self.patch_size-w), 'h -> (h w)', w=w)+torch.arange(h*w)
        pos_emb_ind = torch.cat((torch.zeros(1), pos_emb_ind+1), dim=0).long()
        x += self.pos_embed[:, pos_emb_ind] # 添加可学习的位置编码
        #x = x + self.pos_embed
        x = self.pos_drop(x) # dropout，防止过拟合

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        return x


def get_encoder(args):
    # 用于提取视觉特征的CNN（这里是 ResNetV2，可看作 CNN Encoder 的主干）

    backbone = ResNetV2(
        layers=args.backbone_layers,
        num_classes=0,
        global_pool='',
        in_chans=args.channels, # 3 for RGB, 1 for grayscale
        preact=False,
        stem_type='same',
        conv_layer=StdConv2dSame)
    # 输出的特征图大小feature map：维度大约是原图的H/16 x W/16
    min_patch_size = 2**(len(args.backbone_layers)+1)
    # 举例：backbone_layers = [2, 3, 7]，则 min_patch_size = 2^(3+1) = 16
    # 即 ResNetV2 输出特征图的下采样倍数为 16

    # HybridEmbed 是一个连接 CNN 和 ViT 的桥梁：一个卷积层+Transformer的组合
    # 它接收 ResNet 的输出 feature map，按 patch 划分后进行 flatten 和线性映射，得到 ViT 的输入 token。
    # patch_size 参数必须是 CNN 下采样倍数的整数倍，确保分 patch 时不会出错。
    # 把 CNN 提取的特征 map 再进行 patch 分割（通常是16x16），作为 Transformer 的输入。比纯 ViT 更有效地保留了局部特征
    def embed_layer(**x):
        ps = x.pop('patch_size', min_patch_size)
        assert ps % min_patch_size == 0 and ps >= min_patch_size, 'patch_size needs to be multiple of %i with current backbone configuration' % min_patch_size
        return HybridEmbed(**x, patch_size=ps//min_patch_size, backbone=backbone)

    encoder = CustomVisionTransformer(img_size=(args.max_height, args.max_width),
                                      patch_size=args.patch_size,
                                      in_chans=args.channels,
                                      num_classes=0,
                                      embed_dim=args.dim,
                                      depth=args.encoder_depth,
                                      num_heads=args.heads,
                                      embed_layer=embed_layer
                                     )
    # encoder 输出一个序列，包含：
    # [CLS] token + flatten 后的 patch embedding。
    # 输出形状为：B x (1 + N) x D
    # N = patch 数 = H' x W'，H' 和 W' 为分 patch 后的尺寸（通常为 H/16, W/16）
    return encoder