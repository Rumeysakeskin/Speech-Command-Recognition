## Speech-Command-Recognition/Key Word Spotting

### Table Of Contents
- [Configuration Preprocessing](#Configuration-Preprocessing)
- [Custom Speech Classification Data Preparing](#Custom-Speech-Classification-Data-Preparing)
- [Speech Data Augmentation](#Speech-Data-Augmentation)
- [Training](#Training)
- [Export to ONNX Model](#Export-to-ONNX-Model)
- [Deploy and Inference](#Deploy-and-Inference)

### Configuration Preprocessing
- NeMo [Matchboxnet_3x1x64_v1](https://github.com/NVIDIA/NeMo/tree/main/examples/asr/conf/matchboxnet) configurations was used.
- Also, you can use other [Nemo ASR configurations](https://github.com/NVIDIA/NeMo/tree/main/examples/asr/conf).
Labels were configured in `config/matchboxnet_3x1x64_v1.yaml` in the following format also, you can add your labels as follows:

```python
  labels_full: ['bed', 'bird', 'cat', 'dog', 'down', 'eight', 'five', 'four', 'go', 'happy', 'house', 'left', 'marvin',
           'nine', 'no', 'off', 'on', 'one', 'right', 'seven', 'sheila', 'six', 'stop', 'three', 'tree', 'two', 'up',
           'wow', 'yes', 'zero', 'label1', 'label2', 'label3']
```

### Custom Speech Classification Data Preparing
- The `nemo_asr` collection expects each dataset to consist of a set of utterances in individual audio files plus a manifest that describes the dataset, with information about one utterance per line `(.json)`.
- Each line of the manifest `(manifest_files/train_manifest.jsonl and manifest_files/val_manifest.jsonl)` should be in the following format:
```python
{"audio_filepath": "/YourWavFilesPath/label1.wav", "duration": 0.8483333333333334, "command": "label1"}
```
- The `audio_filepath` field should provide an absolute path to the `.wav` file corresponding to the utterance. The `command` field should contain the transcript for the label, and the `duration` field should reflect the duration of the utterance in seconds.

### Speech Data Augmentation
- Also, you can use my repository [
speech-data-augmentation](https://github.com/Rumeysakeskin/speech-data-augmentation) to **increase the diversity** of your dataset augmenting the data artificially for ASR models training.

### Training
- For training run the following command. Checkpoints `.ckpt` will be saved `checkpoints/` after every epoch.
```python
python training.py
```
- You may follow training from [Weight&Biases](https://wandb.ai/site).
```python
from pytorch_lightning.loggers import WandbLogger
wandb_logger = WandbLogger(name=run_name, project="SpeechClassification")
trainer = pl.Trainer(gpus=2, max_epochs=200, amp_level='O1', precision=16,
                       max_steps=None, num_nodes=1, accelerator="ddp",
                       accumulate_grad_batches=1, checkpoint_callback=checkpoint_callback, val_check_interval=1.0,
                       logger=wandb_logger, log_every_n_steps=100)
```
### Export to ONNX Model
- You can create `.onnx` model from your best `.ckpt` file. 
```python
export_model.ipynb
```
### Deploy and Inference
- You may perform inference a sample of speech after loading the `model/classification_model.onnx` by running:
```python
python inference.py
```
