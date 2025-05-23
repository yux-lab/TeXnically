import albumentations as alb
from albumentations.pytorch import ToTensorV2

# refer:
# https://github.com/opendatalab/UniMERNet/blob/main/unimernet/processors/formula_processor.py
# https://github.com/hendrycks/robustness
# https://github.com/facebookresearch/nougat/blob/main/nougat/transforms.py

from dataset.processors.formula_processor_helper.weather import Fog, Frost, Snow, Rain, Shadow
from dataset.processors.formula_processor_helper.nougat import Bitmap, Dilation, Erosion

train_transform = alb.Compose([
    alb.Compose([
        Bitmap(p=0.05), # 位图
        alb.OneOf([Fog(), Frost(), Snow(), Rain(), Shadow()], p=0.2), # 天气背景效果
        alb.OneOf([Erosion((2, 3)), Dilation((2, 3))], p=0.2), # 腐蚀 / 膨胀
        alb.ShiftScaleRotate(shift_limit=0, scale_limit=(-.15, 0), rotate_limit=1, border_mode=0,
                             interpolation=3, value=[255, 255, 255], p=1), # 缩放 + 微小旋转
        alb.GridDistortion(distort_limit=0.1, border_mode=0, interpolation=3, value=[255, 255, 255], p=.5) # 模拟角度、形变
    ], p=.15),
    alb.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.3), # RGB通道偏移
    alb.GaussNoise(10, p=.2), # 高斯噪声
    alb.RandomBrightnessContrast(.05, (-.2, 0), True, p=0.2),
    alb.ImageCompression(95, p=.3),
    alb.ToGray(always_apply=True),
    alb.Normalize((0.7931, 0.7931, 0.7931), (0.1738, 0.1738, 0.1738)),
    ToTensorV2()
])


test_transform = alb.Compose(
    [
        alb.ToGray(always_apply=True),
        alb.Normalize((0.7931, 0.7931, 0.7931), (0.1738, 0.1738, 0.1738)),
        # alb.Sharpen()
        ToTensorV2(),
    ]
)