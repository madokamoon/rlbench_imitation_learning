#!/bin/bash

configs=(
    # "act_config_f.yaml"    
    # "act_config_o.yaml"   
    # "act_config_w.yaml"  
    # "act_config_fwo.yaml" 
    "act_weight_config.yaml"
)

for i in "${!configs[@]}"; do

    config=${configs[$i]}
    python act_training.py --config-name="$config"

done


for i in "${!configs[@]}"; do

    config=${configs[$i]}
    # python data_sampler.py --config-name="$config"

done

