from omegaconf import OmegaConf
from ruamel.yaml import YAML
import torch
import soxr
from jetson_voice_utils.trt_model import TRTModel
import importlib
import numpy as np

class Inference(object):
    def __init__(self):

        config_path = "./config/matchboxnet_3x1x64_v1.yaml"
        self.config = OmegaConf.load(config_path)

        self.sample_rate = 16000
        DYNAMIC_SHAPES = {"min": (1, 64, 1), "max": (1, 64, 1024)}

        self.trt_model = TRTModel(self.config, DYNAMIC_SHAPES)

        yaml = YAML(typ='safe')
        with open(self.config) as f:
            params = yaml.load(f)

        preprocessor_name = params['model']['preprocessor']['_target_'].rsplit(".", 1)
        preprocessor_class = getattr(importlib.import_module(preprocessor_name[0]), preprocessor_name[1])
        preprocessor_config = params['model']['preprocessor'].copy()
        preprocessor_config.pop('_target_')
        self.preprocessor = preprocessor_class(**preprocessor_config)


    def to_numpy(self, tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

    def int2float(self, sound):
        abs_max = np.abs(sound).max()
        sound = sound.astype('float32')
        if abs_max > 0:
            sound *= 1 / abs_max
        sound = sound.squeeze()  # depends on the use case
        return sound

    def inference(self):

        audio_int16 = np.fromstring(bytes_, dtype=np.int16)  # convert bytes to numpy array for stt prediction

        audio_float32 = self.int2float(audio_int16)
        if self.rate != 16000:
            audio_float32 = soxr.resample(audio_float32, self.rate, 16000, quality=soxr.VHQ)

        signal = np.expand_dims(audio_float32, 0)
        signal = torch.from_numpy(signal)

        audio_signal_len = torch.from_numpy(np.array([len(signal)]))

        processed_signal, processed_signal_len = self.preprocessor(input_signal=signal, length=audio_signal_len, )
        processed_signal = self.to_numpy(processed_signal)

        logits = self.trt_model.execute(processed_signal)

        del bytes_
        del signal
        del audio_signal_len

        logits = np.array(logits)
        inference = int(np.argmax(logits))

        return inference

keywords = {0: "bed", 1: "bird", 2: "cat", 3: "dog", 4: "down", 5: "eight",
            6: "five", 7: "four", 8: "go", 9: "happy", 10: "house",
            11: "left", 12: "marvin",13: "nine", 14: "no", 15: "off",
            16: "on", 17: "one", 18: "right",19: "seven", 20: "sheila",
            21: "six", 22: "stop", 23: "three",  24: "tree",  25: "two", 26: "up",
            27: "wow", 28: "yes", 29: "zero", 30: "label1", 31: "label2",32: "label3"}

model_inference = Inference()
inference = model_inference.inference()
print(f"Detected Keyword: {keywords[inference]}")