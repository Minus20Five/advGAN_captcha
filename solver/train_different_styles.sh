#! /bin/bash
styles="More_Warp Varying_Contrast_3 Varying_Contrast_5"

for i in $styles; do 
    python3 captcha_train.py -d $i
done