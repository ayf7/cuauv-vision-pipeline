#!/usr/bin/env python3
import argparse
import sys
import os
from pathlib import Path
from ultralytics import YOLO
import re

# adds new yolo SHM variables from .pt file
# deletes old SHM variables
# prints current SHM yolo variables

software_path = os.environ.get("CUAUV_SOFTWARE")
new_file_path = Path(software_path) / 'libshm' / 'vars.conf'

def current_yolo_shm_names():
    word = "yolo"
    yolo_shm_name = []

    with new_file_path.open('r') as file:
        lines = file.readlines()

    for line in lines:
        if line.startswith("\t") or line.startswith("    "):
            continue
        if line.startswith(word):
            yolo_shm_name.append(line)
    return yolo_shm_name

def delete_specified_shm(to_delete = None):
    if to_delete == None:
        return [""]
    if to_delete == ["yolo"]:
        to_delete = current_yolo_shm_names()
        print("to_delete",to_delete)

    with new_file_path.open('r') as file:
        lines = file.readlines()

    delete_block = False
    keep = []
    delete = []
    for line in lines:
        if line.strip() in to_delete or line.strip() + "\n" in to_delete:
            delete_block = True
            delete.append(line.strip())

        if delete_block == False:
            keep.append(line)
            
        if line.strip() == "" and delete_block == True:
            delete_block = False

    with new_file_path.open('w') as file:
        file.writelines(keep)
    return delete


def add_model_outputs(model_path):
    added_vars = []
    if not software_path:
        print("software path not valid")
        return
    
    model = YOLO(model_path)
    
    for value in model.names.values():
        flag = False
        value = "_".join(value.split("-"))
        with new_file_path.open('r') as f:
            for i in f.readlines():
                if i == "yolo_" + value or i == "yolo_" + value + "\n":
                    flag = True
        if flag:
            continue
        
        print("adding ", "yolo_" + value)
        added_vars.append("yolo_" + value)
        with new_file_path.open('a') as file1:
            file1.write("yolo_" + value + "\n")
            file1.write("\tdouble angle\n")
            file1.write("\tdouble area\n")
            file1.write("\tdouble center_x\n")
            file1.write("\tdouble center_y\n")
            file1.write("\tdouble confidence\n")
            file1.write("\tint visible\n")
            file1.write("\tdouble xmax\n")
            file1.write("\tdouble xmin\n")
            file1.write("\tdouble ymax\n")
            file1.write("\tdouble ymin\n")
            file1.write("\tint int_name\n")
            file1.write("\n")

    return added_vars
parser = argparse.ArgumentParser(description="")

parser.add_argument("--print", action="store_true",
                    help="Prints current YOLO SHM vars.")

parser.add_argument("--delete", nargs="*", default=None,
                    help="Deletes specified shm groups. --delete-yolo-shm-vars yolo deletes all yolo shm groups.")

parser.add_argument("--add",nargs=1, default=None,
                    help="Add all extracted yolo shm groups")

args = parser.parse_args()

used_flags = sum([bool(args.print),
                  bool(args.delete),
                  bool(args.add)])

if used_flags > 1:
    print("Error: Only one action flag can be used at a time.")
    sys.exit(1)

if args.print:
    shm_vars = current_yolo_shm_names()
    print('\n'.join(shm_vars))

if args.delete:
    print(args.delete)
    deleted = delete_specified_shm(args.delete)
    print("Deleted SHM vars:")
    print(', '.join(deleted))

if args.add:
    added = add_model_outputs(args.add[0])
    print('Added SHM vars:')
    print('\n'.join(added))



