## Speech-Command-Recognition/Key Word Spotting

### Preprocessing
- NeMo [Matchboxnet_3x1x64_v1](https://github.com/NVIDIA/NeMo/tree/main/examples/asr/conf/matchboxnet) configurations was used.
Labels were configured in `config/matchboxnet_3x1x64_v1.yaml` in the following format also, you can add your labels as follows:

```
  labels_full: ['bed', 'bird', 'cat', 'dog', 'down', 'eight', 'five', 'four', 'go', 'happy', 'house', 'left', 'marvin',
           'nine', 'no', 'off', 'on', 'one', 'right', 'seven', 'sheila', 'six', 'stop', 'three', 'tree', 'two', 'up',
           'wow', 'yes', 'zero', 'label1', 'label2', 'label3']
```
### Training
```
training.py
```
### Load and Export onnx model to Inference
```
export_model.ipynb
```
### Deploy and Inference
- You may perform inference a sample of speech after loading the `model/classification_model.onnx` by running:
```
inference.py
```
