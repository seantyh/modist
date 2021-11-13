import os
import glob
from itertools import islice
from modist_import import modist
from modist.modist_pcm_dataset import ModistPcmDataset, get_modist_pcm_collate_fn
from modist.mp3_info import get_lang_encoder
from torch.utils.data import DataLoader
import numpy as np

def dummy_feat_extractor(x, *args, **kwargs):
    data = [xx.shape for xx in x]    
    return type('DummyFeats', (object,), {"input_values": data})

def test_modist_pcm_dataset():
    mp3_dir = os.path.join(
                os.path.dirname(__file__), 
                "../data/pcm")
    
    pcm_list = glob.glob(mp3_dir + "/*.pcm")[:2]    
    ds = ModistPcmDataset(pcm_list, speech_only=True)
    
    assert len(ds.inventory) != 0
    lang_encoder = get_lang_encoder()
    collate_fn = get_modist_pcm_collate_fn(dummy_feat_extractor, lang_encoder, ds.sample_rate)
    train_loader = DataLoader(ds, batch_size=16, collate_fn=collate_fn, drop_last=True, shuffle=True)
    batch = next(iter(train_loader))

    assert batch

def test_modist_pcm_dataset_limit():
    mp3_dir = os.path.join(
                os.path.dirname(__file__), 
                "../data/pcm")
    
    pcm_list = glob.glob(mp3_dir + "/*.pcm")  
    ds = ModistPcmDataset(pcm_list, speech_only=True, limit_data_prop=0.1)
            
    assert len(ds.inventory) != 0
    lang_encoder = get_lang_encoder()
    collate_fn = get_modist_pcm_collate_fn(dummy_feat_extractor, lang_encoder, ds.sample_rate)
    train_loader = DataLoader(ds, batch_size=16, collate_fn=collate_fn, drop_last=True, shuffle=True)
    batch = next(iter(train_loader))

    assert batch