#!/bin/bash
for i in More_Warp Varying_Contrast_3 Varying_Contrast_5
do 
    python3 solver/captcha_train.py -d i
done
