import random
from itertools import islice, chain, starmap
import numpy as np
from torch.utils.data import IterableDataset
import pydub
from .mp3_info import get_lang, get_segment_type, get_segments

class ModistDataset(IterableDataset):
    def __init__(self, w2v_feat_extractor, mp3_list,
                sample_rate=16000, batch_size=16, secs_per_seq=5,
                speech_only=False, randomize_seg=False, random_seed=None,
                limit_minutes=None):
        self.feature_extractor = w2v_feat_extractor
        self.sample_rate = sample_rate
        self.mp3_list = mp3_list
        self.batch_size = batch_size
        self.secs_per_seq = secs_per_seq
        self.speech_only = speech_only
        self.randomize_seg = randomize_seg
        self.random_seed = random_seed
        self.limit_minutes = limit_minutes
        random.seed(self.random_seed)

    def __iter__(self):
        mp3_list = self.mp3_list[::1]
        if self.randomize_seg:            
            random.shuffle(mp3_list)
        mp3_iter = map(self.make_mp3_tuple, mp3_list)
        seg_iter = starmap(self.make_sample_tuple, mp3_iter)
        batch_iter = map(lambda seg: self.batch(seg, self.batch_size), seg_iter)
        data_iter = chain.from_iterable(batch_iter)
        return data_iter

    def make_mp3_tuple(self, mp3_name):
        mp3_dub = pydub.AudioSegment.from_mp3(mp3_name)
        mp3_lang = get_lang(mp3_name)
        return (mp3_name, mp3_dub, mp3_lang)

    def make_sample_tuple(self, mp3_name, dub, lang):
        seg_iter = self.segment(mp3_name, dub, self.secs_per_seq)
        return ((lang, category, samples)
                 for category, samples
                 in seg_iter)

    def segment(self, mp3_name, mp3_dub, secs=5):
        blen = secs * 1000
        
        offset_max = len(mp3_dub)
        if self.limit_minutes:
            offset_max = min(self.limit_minutes*60*1000, len(mp3_dub))
        offsets = list(range(0, offset_max, blen))
        if self.randomize_seg:            
            random.shuffle(offsets)

        for offset in offsets:
            # category: (category, matched_secs)
            category = get_segment_type(mp3_name, int(offset/1000), int((offset+blen)/1000))
            if category[1] != secs:
                continue
            # replace category with category name only
            category = category[0]
            if self.speech_only and category != "speech":
                continue

            samples = np.array(mp3_dub[offset:offset+blen].get_array_of_samples(),
                                dtype=np.double)
            yield (category, samples)

    def batch(self, seg_iter, batch_size):
        batch_iter = iter(lambda: list(islice(seg_iter, batch_size)), [])
        feature_extractor = self.feature_extractor
        sample_rate = self.sample_rate
        for i, batched_data in enumerate(batch_iter):
            langs, categories, samples = list(zip(*batched_data))
            if len(samples) < batch_size: continue

            in_tensor = feature_extractor(
                samples, sampling_rate=sample_rate, padding=True,
                return_tensors="pt").input_values
            yield {"tensorX": in_tensor, "lang": langs, "category": categories}
