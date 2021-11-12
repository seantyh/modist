import random
from itertools import islice, chain, starmap
import numpy as np
from torch.utils.data import Dataset
import pydub
from .mp3_info import get_lang, get_segment_type, get_segments


class ModistPcmDataset(Dataset):
    def __init__(self, pcm_list,
                secs_per_seq=5,
                sample_rate=16000, 
                pcm_length_in_secs=1200,
                speech_only=False,
                limit_minutes=None):
        
        self.pcm_mmap = {}
        self.pcm_list = pcm_list
        self.secs_per_seq = secs_per_seq
        self.speech_only = speech_only
        self.sample_rate = sample_rate
        self.pcm_length = pcm_length_in_secs * sample_rate
        self.limit_minutes = limit_minutes
        self.inventory = self.make_inventory()

    def make_inventory(self):
        pcm_tuples = map(self.make_pcm_tuples, self.pcm_list)
        seg_tuples = starmap(self.make_sample_tuples, pcm_tuples)
        data_iter = chain.from_iterable(seg_tuples)
        return list(data_iter)

    def __getitem__(self, idx):
        seg_item = self.inventory[idx]
        seg_data = self.load_data(seg_item)
        return seg_data

    def __len__(self):
        return len(self.inventory)

    def make_pcm_tuples(self, pcm_name):
        pcm_lang = get_lang(pcm_name)
        return (pcm_name, pcm_lang)

    def make_sample_tuples(self, pcm_name, pcm_lang):
        seg_iter = self.segment(pcm_name, self.secs_per_seq)
        return ((pcm_name, pcm_lang, category, ss, ee)
                 for category, ss, ee in seg_iter)

    def segment(self, pcm_name, secs=5):

        offset_max = self.pcm_length // self.sample_rate
        if self.limit_minutes:
            offset_max = min(self.limit_minutes*60, offset_max)
        offsets = list(range(0, offset_max, secs))

        for offset in offsets:
            # category: (category, matched_secs)
            category = get_segment_type(pcm_name, int(offset), int((offset+secs)))
            if category[1] != secs:
                continue
            # replace category with category name only
            category = category[0]
            if self.speech_only and category != "speech":
                continue

            yield (category, offset, offset+secs)

    def load_data(self, seg_item):
        # seg_item: (pcm_name, pcm_lang, category, ss, ee)
        pcm_name = seg_item[0]
        ss, ee = seg_item[3:5]
        if pcm_name not in self.pcm_mmap:
            pcm_arr = np.memmap(pcm_name, np.dtype('int16'), 'r')
            self.pcm_mmap[pcm_name] = pcm_arr
        else:
            pcm_arr = self.pcm_mmap[pcm_name]

        sr = self.sample_rate
        samples = pcm_arr[ss*sr: ee*sr]
        return {
            "pcm_lang": seg_item[1],
            "category": seg_item[2],
            "ss": ss, "ee": ee,
            "samples": samples
        }

def get_modist_pcm_collate_fn(feature_extractor, lang_encoder, sample_rate):
    def modist_pcm_collatefn(batch):
        samples = [x["samples"] for x in batch]
        in_tensor = feature_extractor(
            samples, sampling_rate=sample_rate, padding=True,
            return_tensors="pt").input_values
        categories = [x["category"] for x in batch]
        langs = lang_encoder.transform([x["pcm_lang"] for x in batch])
        return {"tensorX": in_tensor, "lang": langs, "category": categories}
    
    return modist_pcm_collatefn
