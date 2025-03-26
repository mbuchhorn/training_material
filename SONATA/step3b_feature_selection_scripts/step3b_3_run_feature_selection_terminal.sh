#!/bin/bash

# Define the input sets
input_sets=(
    "" # --> paste your input created in step3b_1 here.
)

# Iterate through each input set and run the feature selection script
for input_set in "${input_sets[@]}"; do
    IFS=' ' read -ra params <<< "$input_set"
    python3 06_feature_selection.py -t "${params[0]}" -o "${params[1]}"
done


