#!/bin/bash

# Define the input sets
input_sets=(
    "/data/habitat/serbia/BIOS_visit_march/models/v1/L2/class-R/data/Pannonian_v1_TP_EUNIS2021_2024_L2_R_all_features.csv /data/habitat/serbia/BIOS_visit_march/models/v1/L2/class-R/data" # --> paste your input created in step3b_1 here.
    "/data/habitat/serbia/BIOS_visit_march/models/v1/L2/class-V/data/Pannonian_v1_TP_EUNIS2021_2024_L2_V_all_features.csv /data/habitat/serbia/BIOS_visit_march/models/v1/L2/class-V/data"
    "/data/habitat/serbia/BIOS_visit_march/models/v1/L2/class-Q/data/Pannonian_v1_TP_EUNIS2021_2024_L2_Q_all_features.csv /data/habitat/serbia/BIOS_visit_march/models/v1/L2/class-Q/data"
    "/data/habitat/serbia/BIOS_visit_march/models/v1/L2/class-S/data/Pannonian_v1_TP_EUNIS2021_2024_L2_S_all_features.csv /data/habitat/serbia/BIOS_visit_march/models/v1/L2/class-S/data"

)

# Iterate through each input set and run the feature selection script
for input_set in "${input_sets[@]}"; do
    IFS=' ' read -ra params <<< "$input_set"
    python3 step3b_2_feature_selection_script.py -t "${params[0]}" -o "${params[1]}"
done


