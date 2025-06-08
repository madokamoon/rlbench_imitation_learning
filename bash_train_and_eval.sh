#!/bin/bash

configs=(
    
    "act_config_fw_mask.yaml" 

)

for i in "${!configs[@]}"; do

    config=${configs[$i]}

    # python act_training.py --config-name="$config"
    python data_sampler.py --config-name="$config"
    

done



