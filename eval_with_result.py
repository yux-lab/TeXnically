from pix2tex.dataset.dataset import Im2LatexDataset
import argparse
import logging
import yaml
import csv
import os

import numpy as np
import torch
from torchtext.data import metrics
from munch import Munch
from tqdm.auto import tqdm
import wandb
from Levenshtein import distance

from pix2tex.models import get_model, Model
from pix2tex.utils import *


def detokenize(tokens, tokenizer):
    toks = [tokenizer.convert_ids_to_tokens(tok) for tok in tokens]
    for b in range(len(toks)):
        for i in reversed(range(len(toks[b]))):
            if toks[b][i] is None:
                toks[b][i] = ''
            toks[b][i] = toks[b][i].replace('Ġ', ' ').strip()
            if toks[b][i] in (['[BOS]', '[EOS]', '[PAD]']):
                del toks[b][i]
    return toks


@torch.no_grad()
def evaluate(model: Model, dataset: Im2LatexDataset, args: Munch, num_batches: int = None, name: str = 'test', results_path: str = "results.csv"):
    """evaluate，and save prediction to csv with a parameterized results path"""

    assert len(dataset) > 0
    device = args.device
    log = {}
    bleus, edit_dists, token_acc = [], [], []

    # path to csv
    file_exists = os.path.isfile(results_path)

    with open(results_path, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(
                ["Image Path", "Predicted LaTeX", "Ground Truth", "BLEU Score", "Edit Distance", "Token Accuracy"])

        pbar = tqdm(enumerate(iter(dataset)), total=len(dataset))
        for i, (seq, im) in pbar:
            if seq is None or im is None:
                continue

            # predict
            dec = model.generate(im.to(device), temperature=args.get('temperature', .2))
            pred = detokenize(dec, dataset.tokenizer)
            truth = detokenize(seq['input_ids'], dataset.tokenizer)

            # calculate BLEU
            bleu = metrics.bleu_score(pred, [alternatives(x) for x in truth])
            bleus.append(bleu)

            # calculate Edit
            pred_latex = post_process(token2str(dec, dataset.tokenizer)[0])
            truth_latex = post_process(token2str(seq['input_ids'], dataset.tokenizer)[0])
            edit_dist = distance(pred_latex, truth_latex) / max(1, len(truth_latex))
            edit_dists.append(edit_dist)

            # calculate Token Accuracy
            dec = dec.cpu()
            tgt_seq = seq['input_ids'][:, 1:]
            shape_diff = dec.shape[1] - tgt_seq.shape[1]
            if shape_diff < 0:
                dec = torch.nn.functional.pad(dec, (0, -shape_diff), "constant", args.pad_token)
            elif shape_diff > 0:
                tgt_seq = torch.nn.functional.pad(tgt_seq, (0, shape_diff), "constant", args.pad_token)
            mask = torch.logical_or(tgt_seq != args.pad_token, dec != args.pad_token)
            tok_acc = (dec == tgt_seq)[mask].float().mean().item()
            token_acc.append(tok_acc)

            # get image path
            image_path = dataset.pairs[i][0][1]  #

            # save to CSV
            writer.writerow([image_path, pred_latex, truth_latex, bleu, edit_dist, tok_acc])

            pbar.set_description(
                'BLEU: %.3f, ED: %.2e, ACC: %.3f' % (np.mean(bleus), np.mean(edit_dists), np.mean(token_acc)))
            if num_batches is not None and i >= num_batches:
                break

    print(f"\nResults saved to {results_path}")
    return np.mean(bleus), np.mean(edit_dists), np.mean(token_acc)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test model')
    parser.add_argument('--config', default=None, help='path to yaml config file', type=str)
    parser.add_argument('-c', '--checkpoint', default=None, type=str, help='path to model checkpoint')
    parser.add_argument('-d', '--data', default='dataset/data/val.pkl', type=str, help='Path to Dataset pkl file')
    parser.add_argument('--no-cuda', action='store_true', help='Use CPU')
    parser.add_argument('-b', '--batchsize', type=int, default=10, help='Batch size')
    parser.add_argument('--debug', action='store_true', help='DEBUG')
    parser.add_argument('-t', '--temperature', type=float, default=.333, help='sampling temperature')
    parser.add_argument('-n', '--num-batches', type=int, default=None,
                        help='how many batches to evaluate on. Defaults to None (all)')
    parser.add_argument('-o', '--output', type=str, default="results.csv", help="Path to save results CSV")

    parsed_args = parser.parse_args()
    if parsed_args.config is None:
        with in_model_path():
            parsed_args.config = os.path.realpath('settings/config.yaml')

    with open(parsed_args.config, 'r') as f:
        params = yaml.load(f, Loader=yaml.FullLoader)

    args = parse_args(Munch(params))
    args.testbatchsize = parsed_args.batchsize
    args.wandb = False
    args.temperature = parsed_args.temperature
    logging.getLogger().setLevel(logging.DEBUG if parsed_args.debug else logging.WARNING)
    seed_everything(args.seed if 'seed' in args else 42)

    # load model
    model = get_model(args)
    if parsed_args.checkpoint is None:
        with in_model_path():
            parsed_args.checkpoint = os.path.realpath('checkpoints/weights.pth')
    model.load_state_dict(torch.load(parsed_args.checkpoint, args.device))

    # load dataset
    dataset = Im2LatexDataset().load(parsed_args.data)
    valargs = args.copy()
    valargs.update(batchsize=args.testbatchsize, keep_smaller_batches=True, test=True)
    # shuffle = True will lead to paths of images are not same as results
    dataset.update(**valargs, shuffle=False)

    # evaluate and save results
    evaluate(model, dataset, args, num_batches=parsed_args.num_batches, results_path=parsed_args.output)