import random
from itertools import islice, chain, starmap
import numpy as np
import torch
from torch.utils.data import Dataset
import pydub
from .mp3_info import get_lang, get_segment_type, get_segments


class ModistPcmAnDataset(Dataset):
    def __init__(self, pcm_list,
                secs_per_seq=3,
                sample_rate=16000, 
                pcm_length_in_secs=1200,
                speech_only=False,
                limit_minutes=None,
                limit_data_prop=None):
        """ModistPcmDataset
        limit_data_prop: a number between 0~1, the proportion of data being served
        """
        self.pcm_mmap = {}
        self.pcm_list = pcm_list
        self.secs_per_seq = secs_per_seq
        self.speech_only = speech_only
        self.sample_rate = sample_rate
        self.pcm_length = pcm_length_in_secs * sample_rate
        self.limit_minutes = limit_minutes        
        self.inventory = self.make_inventory(limit_data_prop)

    def make_inventory(self, limit_data_prop):                
        seg_tuples = starmap(self.make_sample_tuples, self.pcm_list)
        data_iter = chain.from_iterable(seg_tuples)
        inventory = list(data_iter)
        if limit_data_prop is not None:
            random.shuffle(inventory)
            inventory = inventory[:int(len(inventory) * limit_data_prop)]
        return inventory

    def __getitem__(self, idx):
        seg_item = self.inventory[idx]
        seg_data = self.load_data(seg_item)
        return seg_data

    def __len__(self):
        return len(self.inventory)

    def make_pcm_tuples(self, pcm_name):
        pcm_lang = get_lang(pcm_name)
        return (pcm_name, pcm_lang)

    def make_sample_tuples(self, pcm_name, pcm_lang, pcm_len):
        seg_iter = self.segment(pcm_name, pcm_len, self.secs_per_seq)
        return ((pcm_name, pcm_lang, category, ss, ee)
                 for category, ss, ee in seg_iter)

    def segment(self, pcm_name, pcm_len, secs=5):

        offset_max = pcm_len // self.sample_rate
        if self.limit_minutes:
            offset_max = min(self.limit_minutes*60, offset_max)
        offsets = list(range(0, offset_max, secs))

        for offset in offsets:
            category = "speech"
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
        samples = [np.array(x["samples"], dtype=np.float32) for x in batch]
        in_tensor = feature_extractor(
            samples, sampling_rate=sample_rate, padding=True,
            return_tensors="pt").input_values
        categories = [x["category"] for x in batch]
        langs = lang_encoder.transform([x["pcm_lang"] for x in batch])
        langs = torch.tensor(langs, dtype=torch.long)
        return {"tensorX": in_tensor, "lang": langs, "category": categories}
    
    return modist_pcm_collatefn
