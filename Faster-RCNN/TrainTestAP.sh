#!/bin/bash

if [ $# -lt 1 ]; then
    echo "Must provide one argument - number of epochs"
    exit 1
fi

END=$1
FILE="~/TestAcc.txt"
CP=$(awk -F '[^0-9]*' '{print $5}' <<< \
        $(ls models/res101/pascal_voc/faster_rcnn_1_1_*))
TEST="test_net.py --dataset pascal_voc --net res101 --checksession 1 --cuda \
        --checkpoint $CP --checkepoch"

# touch FILE
for epo in $(seq 1 $END)
do
    rm data/cache/*
    rm data/VOCdevkit2007/VOC2007/ImageSets/Main/test.txt_annots.pkl
    python $TEST $epo # >> FILE       # testing accuracy
    cd data/VOCdevkit2007/VOC2007/ImageSets/Main
    # swap test.txt trainval.txt
    mv test.txt tmp.txt && mv trainval.txt test.txt && mv tmp.txt trainval.txt
    cd ../../../../../
    rm data/cache/*
    rm data/VOCdevkit2007/VOC2007/ImageSets/Main/test.txt_annots.pkl
    python $TEST $epo # >> FILE       # training accuracy
    cd data/VOCdevkit2007/VOC2007/ImageSets/Main
    mv test.txt tmp.txt && mv trainval.txt test.txt && mv tmp.txt trainval.txt
    cd ../../../../../
done
