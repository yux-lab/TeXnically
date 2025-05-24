## generate dataset
```bash
python -m dataset.dataset --equations path_to_textfile --images path_to_images --out dataset.pkl --tokenizer model/dataset/tokenizer.json
```

## train
```bash
python -m train --config model/settings/config.yaml
```


## eval and eval_with_result
```bash
python eval.py --checkpoint model/checkpoints/weights.pth --data dataset/data/miniTrain.pkl --config model/settings/config.yaml
python eval_with_result.py --checkpoint model/checkpoints/weights.pth --data dataset/data/miniTrain.pkl --config model/settings/config.yaml --output dataset/data/csv_results
```

## api
```bash
pip install uvicorn
python -m uvicorn api.app:app --host 0.0.0.0 --port 8502
streamlit run api/streamlit.py

```

## Acknowledgment

[LaTeX-OCR](https://github.com/lukas-blecher/LaTeX-OCR/tree/main)

[rainyl](https://github.com/rainyl)

[UniMERNet](https://github.com/opendatalab/UniMERNet/tree/main)
