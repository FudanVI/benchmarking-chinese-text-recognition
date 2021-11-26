import os
import cv2
import json
import numpy as np
from PIL import Image

# Please modify the path to generate the SCUT cropped images
# To build the testing dataset, please simply replace 'train' with 'test'
json_root = './SCUT-HCCDoc_Dataset_Release_v2/hccdoc_train.json' # the path for hccdoc_train.json
image_root = './SCUT-HCCDoc_Dataset_Release_v2/image' # the path for image directory
save_root = './SCUT_train' # the path for saving images, please create directory first

f = open(json_root)
results = json.load(f)
print('Finish loading json')

f = open(os.path.join(save_root, 'gt.txt'), 'w+')
five_keys = results['annotations'].keys()
cnt = 0
for key in five_keys:
    database = results['annotations'][key]
    for data in database:
        file_path = os.path.join(image_root, data['file_path'])
        gts = data['gt']
        for gt in gts:
            point, text = gt['point'], gt['text']
            crop_img = image_process(file_path, point[0:2], point[2:4], point[4:6], point[6:8])

            crop_img.save(os.path.join(save_root, '{}.jpg'.format(cnt)))
            f.write('{} {}\n'.format(os.path.join(save_root, '{}.jpg'.format(cnt)), text.replace(' ','')))

            # display(crop_img)
            cnt += 1
            if cnt % 100 == 0:
                print(cnt)
