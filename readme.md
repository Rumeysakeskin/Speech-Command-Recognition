## Speech Detection/Classification 

### Preprocessing
- [Matchboxnet_3x1x64_v1 model config file](https://catalog.ngc.nvidia.com/orgs/nvidia/models/quartznet_15x5_ls_sp/files) was used that trained only on LibriSpeech.
Labels were configured in `config/matchboxnet_3x1x64_v1.yaml` in the following format and you can add labels as follows:


```
  labels_full: ['bed', 'bird', 'cat', 'dog', 'down', 'eight', 'five', 'four', 'go', 'happy', 'house', 'left', 'marvin',
           'nine', 'no', 'off', 'on', 'one', 'right', 'seven', 'sheila', 'six', 'stop', 'three', 'tree', 'two', 'up',
           'wow', 'yes', 'zero', 'label1', 'label2', 'label3']
```
