import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import imagesize
import logging
import glob
import os
from os.path import join
from collections import defaultdict
import pickle
import cv2
from transformers import PreTrainedTokenizerFast
from tqdm.auto import tqdm

from utils.utils import in_model_path
from dataset.transforms import train_transform, test_transform



class Im2LatexDataset:
    keep_smaller_batches = False
    shuffle = True
    batchsize = 16
    max_dimensions = (1024, 512)
    min_dimensions = (32, 32)
    max_seq_len = 1024
    pad_token = "[PAD]"
    bos_token = "[BOS]"
    eos_token = "[EOS]"
    pad_token_id = 0
    bos_token_id = 1
    eos_token_id = 2
    transform = train_transform
    data = defaultdict(lambda: [])

    def __init__(self, equations=None, images=None, tokenizer=None, shuffle=True, batchsize=16, max_seq_len=1024,
                 max_dimensions=(1024, 512), min_dimensions=(32, 32), pad=False, keep_smaller_batches=False, test=False):
        """Generates a torch dataset from pairs of `equations` and `images`.

        Args:
            equations (str, optional): Path to equations. Defaults to None.
            images (str, optional): Directory where images are saved. Defaults to None.
            tokenizer (str, optional): Path to saved tokenizer. Defaults to None.
            shuffle (bool, opitonal): Defaults to True.
            batchsize (int, optional): Defaults to 16.
            max_seq_len (int, optional): Defaults to 1024.
            max_dimensions (tuple(int, int), optional): Maximal dimensions the model can handle
            min_dimensions (tuple(int, int), optional): Minimal dimensions the model can handle
            pad (bool): Pad the images to `max_dimensions`. Defaults to False.
            keep_smaller_batches (bool): Whether to also return batches with smaller size than `batchsize`. Defaults to False.
            test (bool): Whether to use the test transformation or not. Defaults to False.
        """

        if images is not None and equations is not None:
            assert tokenizer is not None
            # 处理路径
            self.images = [path.replace('\\', '/') for path in glob.glob(join(images, '*.png'))]
            #获取样本数量
            self.sample_size = len(self.images)
            # 加载公式文件，转成一个列表
            eqs = open(equations, 'r', encoding='UTF-8').read().split('\n')
            # 提取图像文件名编号作为索引
            self.indices = [int(os.path.basename(img).split('.')[0]) for img in self.images]
            # 初始化分词器（Tokenizer） encode:文本转 token ID decode:ID 转回文本
            self.tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer)
            # 各种配置参数
            self.shuffle = shuffle
            self.batchsize = batchsize
            self.max_seq_len = max_seq_len
            self.max_dimensions = max_dimensions
            self.min_dimensions = min_dimensions
            self.pad = pad
            self.keep_smaller_batches = keep_smaller_batches
            self.test = test
            # check the image dimension for every image and group them together
            # 相同尺寸的图像会被放在一起，可以避免填充（padding）造成的资源浪费
            try:
                for i, im in tqdm(enumerate(self.images), total=len(self.images)):
                    width, height = imagesize.get(im)
                    if min_dimensions[0] <= width <= max_dimensions[0] and min_dimensions[1] <= height <= max_dimensions[1]:
                        self.data[(width, height)].append((eqs[self.indices[i]], im))
            except KeyboardInterrupt:
                pass
            self.data = dict(self.data)
            self._get_size()

            # 让一个类的实例变成可迭代对象（iterable）
            # 可以通过 for batch in dataset: 来遍历数据
            iter(self)

    def __len__(self):
        return self.size

    def __iter__(self):
        self.i = 0
        self.transform = test_transform if self.test else train_transform
        self.pairs = []
        for k in self.data:
            info = np.array(self.data[k], dtype=object)
            p = torch.randperm(len(info)) if self.shuffle else torch.arange(len(info))
            for i in range(0, len(info), self.batchsize):
                batch = info[p[i:i+self.batchsize]]
                if len(batch.shape) == 1:
                    batch = batch[None, :]
                if len(batch) < self.batchsize and not self.keep_smaller_batches:
                    continue
                self.pairs.append(batch)
        if self.shuffle:
            self.pairs = np.random.permutation(np.array(self.pairs, dtype=object))
        else:
            self.pairs = np.array(self.pairs, dtype=object)
        self.size = len(self.pairs)

        self.size = len(self.pairs)
        return self

    def __next__(self):
        if self.i >= self.size:
            raise StopIteration
        self.i += 1
        return self.prepare_data(self.pairs[self.i-1])

    def prepare_data(self, batch):
        """loads images into memory

        Args:
            batch (numpy.array[[str, str]]): array of equations and image path pairs

        Returns:
            tuple(torch.tensor, torch.tensor): data in memory
        """

        eqs, ims = batch.T
        # 编码
        tok = self.tokenizer(list(eqs), return_token_type_ids=False)
        # 返回token id 和 attention mask
        # pad with bos and eos token
        for k, p in zip(tok, [[self.bos_token_id, self.eos_token_id], [1, 1]]):
            # 对 token 序列做 padding ，使得一个 batch 中的不同长度公式可以组成一个固定尺寸的 tensor
            tok[k] = pad_sequence([torch.LongTensor([p[0]]+x+[p[1]]) for x in tok[k]], batch_first=True, padding_value=self.pad_token_id)
        # check if sequence length is too long
        if self.max_seq_len < tok['attention_mask'].shape[1]:
            return next(self)
        images = []
        for path in list(ims):
            im = cv2.imread(path)
            if im is None:
                print(path, 'not found!')
                continue
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            if not self.test:
                # sometimes convert to bitmask
                if np.random.random() < .04:
                    im[im != 255] = 0
            images.append(self.transform(image=im)['image'][:1])
            # 调试 0
            debug_image = self.transform(image=im)['image']
            debug_image_np = (debug_image.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)  # CHW -> HWC
            save_path = "debug/augment/after"
            os.makedirs(save_path, exist_ok=True)

            # 保存 transform 后的图像（调试用）
            # 反归一化参数（和 transform 中的 Normalize 一致）
            # mean = np.array([0.7931, 0.7931, 0.7931])
            # std = np.array([0.1738, 0.1738, 0.1738])
            #
            # # 获取 transform 后的图像张量并转换为 numpy
            # debug_image = self.transform(image=im)['image']  # CHW, [0,1]
            # debug_image_np = debug_image.permute(1, 2, 0).cpu().numpy()  # HWC
            #
            # # 反归一化：x = x * std + mean
            # debug_image_np = debug_image_np * std + mean
            #
            # # 缩放到 [0, 255] 并转为 uint8
            # debug_image_np = (debug_image_np * 255).clip(0, 255).astype(np.uint8)
            #
            # # 保存图像
            # save_path = "debug/augment/after"
            # os.makedirs(save_path, exist_ok=True)
            # filename = os.path.basename(path)
            # cv2.imwrite(os.path.join(save_path, f"aug_{filename}"), cv2.cvtColor(debug_image_np, cv2.COLOR_RGB2BGR))
        try:
            images = torch.cat(images).float().unsqueeze(1)
        except RuntimeError:
            logging.critical('Images not working: %s' % (' '.join(list(ims))))
            return None, None
        if self.pad:
            h, w = images.shape[2:]
            images = F.pad(images, (0, self.max_dimensions[0]-w, 0, self.max_dimensions[1]-h), value=1)
        return tok, images

    def _get_size(self):
        self.size = 0
        for k in self.data:
            div, mod = divmod(len(self.data[k]), self.batchsize)
            self.size += div  # + (1 if mod > 0 else 0)

    def load(self, filename, args=[]):
        """returns a pickled version of a dataset

        Args:
            filename (str): Path to dataset
        """
        if not os.path.exists(filename):
            with in_model_path():
                tempf = os.path.join('..', filename)
                if os.path.exists(tempf):
                    filename = os.path.realpath(tempf)
        with open(filename, 'rb') as file:
            x = pickle.load(file)
        return x

    def combine(self, x):
        """Combine Im2LatexDataset with another Im2LatexDataset

        Args:
            x (Im2LatexDataset): Dataset to absorb
        """
        for key in x.data.keys():
            if key in self.data.keys():
                self.data[key].extend(x.data[key])
                self.data[key] = list(set(self.data[key]))
            else:
                self.data[key] = x.data[key]
        self._get_size()
        iter(self)

    # 把当前数据集对象的所有数据和配置保存到一个 .pkl 文件
    def save(self, filename):
        """save a pickled version of a dataset

        Args:
            filename (str): Path to dataset
        """
        with open(filename, 'wb') as file:
            pickle.dump(self, file)

    def update(self, **kwargs):
        """

        Args:
            **kwargs: 可以传入任意多个关键字参数

        Returns:

        """
        # 批量更新基础属性
        for k in ['batchsize', 'shuffle', 'pad', 'keep_smaller_batches', 'test', 'max_seq_len']:
            if k in kwargs:
                setattr(self, k, kwargs[k])
        # 图像尺寸过滤
        if 'max_dimensions' in kwargs or 'min_dimensions' in kwargs:
            if 'max_dimensions' in kwargs:
                self.max_dimensions = kwargs['max_dimensions']
            if 'min_dimensions' in kwargs:
                self.min_dimensions = kwargs['min_dimensions']
            temp = {}
            for k in self.data:
                if self.min_dimensions[0] <= k[0] <= self.max_dimensions[0] and self.min_dimensions[1] <= k[1] <= self.max_dimensions[1]:
                    temp[k] = self.data[k]
            self.data = temp
        if 'tokenizer' in kwargs:
            tokenizer_file = kwargs['tokenizer']
            if not os.path.exists(tokenizer_file):
                with in_model_path():
                    tokenizer_file = os.path.realpath(tokenizer_file)
            self.tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_file)
        self._get_size()
        iter(self)




def generate_tokenizer(equations, output, vocab_size):
    from tokenizers import Tokenizer, pre_tokenizers
    from tokenizers.models import BPE
    from tokenizers.trainers import BpeTrainer
    tokenizer = Tokenizer(BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    trainer = BpeTrainer(special_tokens=["[PAD]", "[BOS]", "[EOS]"], vocab_size=vocab_size, show_progress=True)
    tokenizer.train(equations, trainer)
    tokenizer.save(path=output, pretty=False)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Train model', add_help=False)
    parser.add_argument('-i', '--images', type=str, nargs='+', default=None, help='Image folders')
    parser.add_argument('-e', '--equations', type=str, nargs='+', default=None, help='equations text files')
    parser.add_argument('-t', '--tokenizer', default=None, help='Pretrained tokenizer file')
    parser.add_argument('-o', '--out', type=str, required=True, help='output file')
    parser.add_argument('-s', '--vocab-size', default=8000, type=int, help='vocabulary size when training a tokenizer')
    args = parser.parse_args()
    if args.tokenizer is None:
        with in_model_path():
            args.tokenizer = os.path.realpath(os.path.join('dataset', 'tokenizer.json'))
    if args.images is None and args.equations is not None:
        print('Generate tokenizer')
        generate_tokenizer(args.equations, args.out, args.vocab_size)
    elif args.images is not None and args.equations is not None:
        print('Generate dataset')
        dataset = None
        for images, equations in zip(args.images, args.equations):
            if dataset is None:
                dataset = Im2LatexDataset(equations, images, args.tokenizer)
            else:
                dataset.combine(Im2LatexDataset(equations, images, args.tokenizer))
        dataset.update(batchsize=1, keep_smaller_batches=True) # cover default
        dataset.save(args.out)
    else:
        print('Not defined')
