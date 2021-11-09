import sys
import os
from modist_import import modist
from modist.mp3_info import get_segments, get_segment_type, get_lang

def test_get_lang():    
    lang_path = os.path.join(
                    os.path.dirname(__file__), 
                    "../data/lang_map.csv")
    lang = get_lang("c5008-2110071100.mp3", lang_path)
    assert lang == "Truku"
    lang = get_lang("c5008-2110111100.mp3", lang_path)
    assert lang == "Atayal"    

def test_get_segments():
    mp3_name = "c5008-2109202000.mp3"
    segs = get_segments(mp3_name)
    assert ('22', '454', "speech") in segs

def test_get_segment_type():
    mp3_name = "c5008-2109202000.mp3"
    seg_type = get_segment_type(mp3_name, 0, 5)
    assert seg_type[0] == "music"
    seg_type = get_segment_type(mp3_name, 370, 375)
    assert seg_type == ("speech", 5)
