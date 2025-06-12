#!/bin/bash

configs=(
    
    "smooth_attention.yaml" 
    # "act_config_fwo_mask.yaml" 

)

for i in "${!configs[@]}"; do

    config=${configs[$i]}
    python act_training.py --config-name="$config"
    
done



