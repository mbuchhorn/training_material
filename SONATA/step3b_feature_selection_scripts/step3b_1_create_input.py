# prepare data for level 2
from pathlib import Path
import os
level = 2
area = 'Pannonian'
version = 1
root = Path(f'/data/habitat/serbia/BIOS_visit_march/models/v{version}/L{level}/')
classes_to_process = set()
for f in root.glob("*class-*"):
    cl = str(f).split('-')[-1]
    classes_to_process.add(cl)
for cl in classes_to_process:
    dest = Path(f'/data/habitat/serbia/BIOS_visit_march/models/v{version}/L{level}/class-{cl}/data')
    csv_files = list(dest.glob(f'*_L{level}_*.csv'))
    #print(f"Processing files in: {dest}")
    if not csv_files:
        print(f"No CSV files found in: {dest}")
    else:
        for csv_file in csv_files:
            csv_file_name = os.path.basename(csv_file)
            print(f"{str(dest / csv_file_name)} {str(dest)}\n")
            #print(f"{str(dest / csv_file_name)} {str(dest)}\n")
