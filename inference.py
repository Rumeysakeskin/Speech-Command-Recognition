import nemo.collections.asr as nemo_asr
from nemo.collections.asr.models.classification_models import EncDecClassificationModel
from omegaconf import OmegaConf
import torch
import librosa
import numpy as np
import onnxruntime

class Inference(object):
    def __init__(self):

        config_path = "./config/matchboxnet_3x1x64_v1.yaml"
        self.config = OmegaConf.load(config_path)
        self.inference_file_location = "./audio_file/test.wav"

        self.sample_rate = 16000

        self.setup_onnx_model()

        self.labels = self.config.model.labels

    def setup_onnx_model(self):

        self.model_path = "./model/classification_model.onnx"

        self.sess = onnxruntime.InferenceSession(self.model_path)
        self.input_name = self.sess.get_inputs()[0].name
        self.label_name = self.sess.get_outputs()[0].name
        self.preprocessor = EncDecClassificationModel.from_config_dict(self.config.model.preprocessor)
        self.crop_or_pad = EncDecClassificationModel.from_config_dict(self.config.model.crop_or_pad_augment)

    def to_numpy(self, tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

    def inference(self):

        with open(self.inference_file_location, "rb") as f:

            signal, sample_rate = librosa.load(f, self.sample_rate)

        signal = np.expand_dims(signal, 0)
        signal = torch.from_numpy(signal)

        audio_signal_len = torch.from_numpy(np.array([len(signal)]))

        processed_signal, processed_signal_len = self.preprocessor(input_signal=signal, length=audio_signal_len, )
        processed_signal, processed_signal_len = self.crop_or_pad(input_signal=processed_signal, length=processed_signal_len)
        processed_signal = self.to_numpy(processed_signal)

        # make prediction
        logits = self.sess.run([self.label_name], {self.input_name: processed_signal})
        logits = logits[0]

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