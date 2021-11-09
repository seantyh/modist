import os
import glob
from tqdm import tqdm
import numpy as np
from pydub import AudioSegment
import pyAudioAnalysis
from pyAudioAnalysis import audioSegmentation as aS
from pyAudioAnalysis import audioTrainTest as at, MidTermFeatures as mtf

model_path = os.path.join(pyAudioAnalysis.__path__[0], "data/models/")
(classifier, mean, std, class_names, 
 mt_win, mid_step, st_win, st_step, 
 compute_beat) = at.load_model(model_path + "svm_rbf_sm")


# adapted from:
# https://github.com/tyiannak/pyAudioAnalysis/blob/ddd6db7955e9cf6b09b6033488343d3e0746ec8d/pyAudioAnalysis/audioSegmentation.py#L566
def classify_mp3(mp3_file):
    mp3 = AudioSegment.from_mp3(mp3_file)
    # for debugging
    # mp3 = mp3[10*1000:30*1000]
    sr = mp3.frame_rate
    labels = []
    
    samples = (np.array(
                mp3.get_array_of_samples())
                   .reshape(-1, mp3.channels)[:,0])

    # feature extraction
    mf_feats, _, _ = mtf.mid_feature_extraction(
                     samples.astype(np.double), sr, mt_win*sr, 
                     mid_step*sr, round(sr*st_win), round(sr*st_step))
    
    mp3_fname = os.path.basename(mp3_file)
    for col_index in tqdm(range(mf_feats.shape[1]), desc=mp3_fname):
        # normalize feature
        feature_vector = (mf_feats[:, col_index] - mean) / std

        # classify
        label_predicted, _ = \
            at.classifier_wrapper(classifier, 'svm', feature_vector)
        labels.append(label_predicted)

    labels = np.array(labels)
    segs, seg_classes = aS.labels_to_segments(labels, mid_step)
    return (segs, seg_classes)

def postproc_segments(segs, seg_classes):
    merged_segs = []
    last_seg_start = 0
    for (ss, es), cat in zip(segs, seg_classes):
        if es - ss < 5:
            pass
        else:
            merged_segs.append((
                int(last_seg_start), int(es), 
                class_names[int(cat)]))
            last_seg_start = es
    return merged_segs

def write_segments(segs, outpath):
    with open(out_path, "w") as fout:
        for seg_x in segs:
            fout.write(",".join(str(x) for x in seg_x))
            fout.write("\n")

if __name__ == "__main__":
    BASE_DIR = os.getcwd()    
    out_dir = os.path.join(BASE_DIR, "../data/segments")
    os.makedirs(out_dir, exist_ok=True)
    
    mp3_dir = os.path.join(BASE_DIR, "../data/mp3")
    mp3_files = glob.glob(mp3_dir + "/*.mp3")
    for mp3_fpath in mp3_files:
        fname = os.path.basename(mp3_fpath).replace(".mp3", ".seg.csv")
        out_path = os.path.join(out_dir, fname)
        if os.path.exists(out_path):
            continue

        segs, seg_classes = classify_mp3(mp3_fpath)
        segs = postproc_segments(segs, seg_classes)
        write_segments(segs, out_path)

