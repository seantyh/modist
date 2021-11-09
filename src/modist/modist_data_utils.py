from itertools import islice, chain
import numpy as np
import torch
from torch.utils.data import IterableDataset
import pydub

class ModistDataset(IterableDataset):
    def __init__(self, w2v_feat_extractor, mp3_list, 
                sample_rate=16000, batch_size=16, secs_per_seq=5):
        self.feature_extractor = w2v_feat_extractor
        self.sample_rate = sample_rate
        self.mp3_list = mp3_list

    def __iter__(self):
        mp3_list = self.mp3_list
        mp3_iter = map(lambda x: pydub.AudioSegment.from_mp3(x), mp3_list)
        self.sample_rate = 0
        pass

    def segment(self, x, secs=5):
        blen = secs * 1000
        for offset in range(0, len(x), blen):
            yield np.array(x[offset:offset+blen].get_array_of_samples(), dtype=np.double)

    def batch(self, seg_iter, batch_size):  
        batch_iter = iter(lambda: list(islice(seg_iter, batch_size)), [])
        feature_extractor = self.feature_extractor
        sample_rate = self.sample_rate
        for i, samples in enumerate(batch_iter):
            if len(samples) < batch_size: continue
            in_tensor = feature_extractor(
                samples, sampling_rate=sample_rate, padding=True,
                return_tensors="pt").input_values
            yield in_tensor
