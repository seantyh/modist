#!/bin/bash

BASE_DIR=$(cd $(dirname $0) && pwd)
echo $BASE_DIR
VIDEO_DIR="/hdd/redhen/"
DATA_DIR="../data/"
MP3_DIR="../data/mp3"
PCM_DIR="../data/pcm"

if [ ! -d ${PCM_DIR} ]; then
    mkdir -p ${PCM_DIR}
fi;

for fname in $(find $MP3_DIR *.mp3); do
    fpath=$(realpath ${fname})
    base_name=$(basename ${fname})
    pcm_path=${PCM_DIR}/${base_name%.*}.pcm
    echo ${pcm_path}
    
    if [ ! -f ${pcm_path} ]; then
        echo "transcoding to pcm: ${base_name}"
        ffmpeg -nostats -loglevel error \
            -i $fpath -f s16le -acodec pcm_s16le \
	    -ss 0 -t 1200 ${pcm_path}
    fi
done
