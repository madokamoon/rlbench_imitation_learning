#!/bin/bash

configs=(
    
    "act_smooth_attention_config_fwo.yaml" 
    # "act_config_fwo_mask.yaml" 

)

for i in "${!configs[@]}"; do

    config=${configs[$i]}

    python act_training.py --config-name="$config"
    python data_sampler.py --config-name="$config"
    

done



