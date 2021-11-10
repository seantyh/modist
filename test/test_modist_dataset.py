import os
import glob
from itertools import islice
from modist_import import modist
from modist.modist_data_utils import ModistDataset
from icecream import ic
import numpy as np

def dummy_feat_extractor(x, *args, **kwargs):
    data = [xx.shape for xx in x]    
    return type('DummyFeats', (object,), {"input_values": data})

def test_dummy_feat_extractor():
    assert dummy_feat_extractor([np.random.randn(10, 4)]*5).input_values

def test_modist_randomize():
    mp3_dir = os.path.join(
                os.path.dirname(__file__), 
                "../data/mp3")
    
    mp3_list = glob.glob(mp3_dir + "/*.mp3")[:1]
    ic(mp3_dir, mp3_list)    
    dataset_1 = ModistDataset(dummy_feat_extractor, mp3_list, randomize_seg=True, random_seed=12345)    
    dataset_3 = ModistDataset(dummy_feat_extractor, mp3_list, randomize_seg=True, random_seed=12346)    
    assert not all(x == y for x, y in zip(dataset_1, dataset_3))

def test_modist_dataset():
    mp3_dir = os.path.join(
                os.path.dirname(__file__), 
                "../data/mp3")
    
    mp3_list = glob.glob(mp3_dir + "/*.mp3")[:2]
    ic(mp3_dir, mp3_list)    
    dataset = ModistDataset(dummy_feat_extractor, mp3_list)
    lang_set = set()
    category_set = set()
    samples_set = set()
    for batch_x in dataset:        
        lang_set.update(batch_x["lang"])
        category_set.update(batch_x["category"])
        samples_set.update(batch_x["tensorX"])
    assert lang_set
    assert len(lang_set) == 2
    assert len(category_set) == 2
    assert len(samples_set) == 1
    
    dataset_so = ModistDataset(dummy_feat_extractor, mp3_list, speech_only=True)
    assert len(list(dataset_so)) < len(list(dataset))