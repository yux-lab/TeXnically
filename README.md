## generate dataset
```bash
python dataset/dataset.py --equations path_to_textfile --images path_to_images --out dataset.pkl
```

## train
```bash
python train.py --config model/settings/config.yaml
```


## eval and eval_with_result
```bash
python eval.py --checkpoint model/checkpoints/weights.pth --data dataset/data/miniTrain.pkl --config model/settings/config-mini.yaml
python eval_with_result.py --checkpoint model/checkpoints/weights.pth --data dataset/data/miniTrain.pkl --config model/settings/config-mini.yaml --output dataset/data/csv_results
```

## Acknowledgment
[LaTeX-OCR](https://github.com/lukas-blecher/LaTeX-OCR/tree/main)
[rainyl](https://github.com/rainyl)
