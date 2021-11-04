#!/bin/bash

BASE_DIR=$(cd $(dirname $0) && pwd)
echo $BASE_DIR
VIDEO_DIR="/hdd/redhen/"
DATA_DIR="../data/"
AUDIO_DIR="../data/mp3"

if [ ! -d ${AUDIO_DIR} ]; then
    mkdir -p ${AUDIO_DIR}
fi;

for fname in $(cat ../data/c5008_list.txt); do
    fpath=$(realpath $VIDEO_DIR/${fname})
    base_name=$(basename ${fname})
    mp3_path=${AUDIO_DIR}/${base_name%.*}.mp3
    if [ ! -f ${mp3_path} ]; then
        echo "transcoding to mp3: ${base_name}"
        ffmpeg -nostats -loglevel error \
            -i $fpath -ac 1 -ar 16000 -vn -f mp3 ${mp3_path}
    fi
done
