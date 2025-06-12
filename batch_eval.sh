#!/bin/bash

configs=(
    
    # training/act_policy_pick_and_lift_norot_sweep_kl/100demos_static/kl_1
    # training/act_policy_smooth_attention_pick_and_lift_norot_sweep_kl/100demos_static/kl_3
    # training/act_policy_smooth_attention_pick_and_lift_norot_sweep_kl/100demos_static/kl_5       
    # training/act_policy_smooth_attention_pick_and_lift_norot_sweep_kl/100demos_static/kl_7
    # training/act_policy_smooth_attention_pick_and_lift_norot_sweep_kl/100demos_static/kl_10
    # training/act_policy_smooth_attention_pick_and_lift_norot_sweep_kl/100demos_static/kl_15
    training/act_policy_smooth_attention_pick_and_lift_norot_sweep_kl/100demos_static/kl_20

)

for i in "${!configs[@]}"; do

    config=${configs[$i]}
    python act_eval.py --ckpt "$config" --show_transform_attention
    
done



