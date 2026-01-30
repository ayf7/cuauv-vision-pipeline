#!/usr/bin/env python3

import sys
import importlib
import os
import inspect
import cv2
import time
import threading
from vision.modules.base import ModuleBase
from vision.capture_sources.capture_source import CaptureSource

if len(sys.argv) != 3:
    print('Hypertune expects 2 command-line arguments:')
    print('  * the name of the vision module file')
    print('  * the name of the image directory')
    sys.exit(1)

module_name = 'vision.modules.' + sys.argv[1][:-3]
if not os.path.isfile('vision/modules/' + sys.argv[1]):
    print('The file', 'vision/modules/' + sys.argv[1], 'doesn\'t exist.')
    sys.exit(1)

module_file = importlib.import_module(module_name)
module = None
for prop_name in dir(module_file):
    prop = getattr(module_file, prop_name)
    if (inspect.isclass(prop) and issubclass(prop, ModuleBase)
            and not prop is ModuleBase):
        module = prop
        break

if module is None:
    print('Failed to find a subclass of ModuleBase in',
            'vision/modules/' + sys.argv[1] + '.')
    sys.exit(1)
print('Found module class', module.__name__ + '.')

if 'module_options' in dir(module_file):
    module_options = getattr(module_file, 'module_options')
else:
    print('No variable called module_options found. Using an empty list.')
    module_options = []

if not os.path.isdir('vision/hypertune_images/' + sys.argv[2]):
    print('The directory vision/hypertune_images/' + sys.argv[2]
            + ' doesn\'t exist.')
    sys.exit(1)
image_filenames = os.listdir(sys.path[0] + '/hypertune_images/' + sys.argv[2])
if len(image_filenames) == 0:
    print('The directory vision/hypertune_images/' + sys.argv[2]
            + ' is empty.')
    sys.exit(1)
if any(filename.split('.')[1] not in ['jpg', 'jpeg', 'png']
        for filename in image_filenames):
    print('The directory vision/hypertune_images/' + sys.argv[2]
            + ' contains files which are not JPGs, JPEGs, or PNGs.')
    sys.exit(1)
print('Found image directory with ' + str(len(image_filenames)) + ' images.')

class HypertuneCaptureSource(CaptureSource):
    def __init__(self, image_dir_name, image_filename, index):
        super().__init__(direction=f'hypertune-{index}')
        self.image = cv2.imread(sys.path[0] + '/hypertune_images/'
                + image_dir_name + '/' + image_filename)
    
    def acquire_next_image(self):
        return self.image, int(time.time())

block_names = []
for i, filename in enumerate(image_filenames):
    source = HypertuneCaptureSource(sys.argv[2], filename, i)
    threading.Thread(target=source.acquisition_loop).start()
    block_names.append(f'hypertune-{i}')
    print(f'> Started capture source for block hypertune-{i} on {filename}.')

original_args = sys.argv[:]
sys.argv = ['vision/modules/' + sys.argv[1]] + block_names
module_obj = module(block_names, module_options)
original_process = module_obj.process
original_post = module_obj.post

def new_process(*images):
    for i, image in enumerate(images):
        def new_post(tag, image):
            if tag.startswith('ht-'):
                original_post(f'{tag}-{i}', image)
        module_obj.post = new_post
        original_process(image)
module_obj.process = new_process

print('Starting module.')
module_obj()