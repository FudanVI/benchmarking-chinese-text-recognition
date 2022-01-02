import os
import cv2
import json
import shutil
import numpy as np
from PIL import Image

dataset_path = 'The absolute path of SCUT-HCCDoc dataset' # eg, '/home/dataset/SCUT-HCCDoc_Dataset_Release_v2'
save_path = 'The empty directory for saving images' # eg, '/home/dataset/my_path'

def check_save_path():
    if os.path.exists(save_path):
        answer = input(f'Path [{save_path}] exists! Do you want to remove it? [y/n]')
        if answer.strip() == 'y':
            shutil.rmtree(save_path)
        else:
            assert False,'Please modify the save_path!'

    print('Create new directory for saving image: {}'.format(save_path))
    os.mkdir(save_path)
    os.mkdir(os.path.join(save_path, 'train_image'))
    os.mkdir(os.path.join(save_path, 'validation_image'))
    os.mkdir(os.path.join(save_path, 'test_image'))

# crop text regions
def image_process(img_path, tl, tr, br, bl):
    img = cv2.imread(img_path)
    width = min(tr[0]-tl[0], br[0]-bl[0])
    height = min(bl[1]-tl[1], br[1]-tr[1])
    point_0 = np.float32([tl, tr, bl, br])
    point_i = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
    transform = cv2.getPerspectiveTransform(point_0, point_i)
    img_i = cv2.warpPerspective(img, transform, (width, height))
    return Image.fromarray(img_i)


def generate_train_validation_dataset():
    print('Start to construct the training and validation sets!')
    print('After the construction, the training set should contain 74,603 samples and the validation set should contain 18,551 samples.')
    train_save_path = os.path.join(save_path, 'train_image')
    validation_save_path = os.path.join(save_path, 'validation_image')
    image_path = os.path.join(dataset_path, 'image')

    scut_file = open(os.path.join(dataset_path, 'hccdoc_train.json'))
    results = json.load(scut_file)
    print('Finish loading json file of scut dataset!')

    f_train = open(os.path.join(train_save_path, 'gt.txt'), 'w+')
    f_validtion = open(os.path.join(validation_save_path, 'gt.txt'), 'w+')
    five_keys = results['annotations'].keys()

    cnt = 0
    for key in five_keys:
        database = results['annotations'][key]
        for data in database:
            file_path = os.path.join(image_path, data['file_path'])
            gts = data['gt']
            for index, gt in enumerate(gts):
                point, text = gt['point'], gt['text']
                crop_img = image_process(file_path, point[0:2], point[2:4], point[4:6], point[6:8])

                if index % 5 == 0:
                    crop_img.save(os.path.join(validation_save_path, '{}.jpg'.format(cnt)))
                    f_validtion.write('{} {}\n'.format(os.path.join(train_save_path, '{}.jpg'.format(cnt)), text.replace(' ', '')))
                else:
                    crop_img.save(os.path.join(train_save_path, '{}.jpg'.format(cnt)))
                    f_train.write('{} {}\n'.format(os.path.join(train_save_path, '{}.jpg'.format(cnt)), text.replace(' ', '')))

                cnt += 1
                if cnt % 1000 == 0:
                    print(cnt)

    f_train.close()
    f_validtion.close()


def generate_test_dataset():
    print('Start to construct the testing set!')
    print('After the construction, the testing set should contain 23,389 samples.')
    test_save_path = os.path.join(save_path, 'test_image')
    image_path = os.path.join(dataset_path, 'image')

    scut_file = open(os.path.join(dataset_path, 'hccdoc_test.json'))
    results = json.load(scut_file)
    print('Finish loading json file of scut dataset!')

    f_test = open(os.path.join(test_save_path, 'gt.txt'), 'w+')
    five_keys = results['annotations'].keys()

    cnt = 0
    for key in five_keys:
        database = results['annotations'][key]
        for data in database:
            file_path = os.path.join(image_path, data['file_path'])
            gts = data['gt']
            for index, gt in enumerate(gts):
                point, text = gt['point'], gt['text']
                crop_img = image_process(file_path, point[0:2], point[2:4], point[4:6], point[6:8])

                crop_img.save(os.path.join(test_save_path, '{}.jpg'.format(cnt)))
                f_test.write('{} {}\n'.format(os.path.join(test_save_path, '{}.jpg'.format(cnt)), text.replace(' ', '')))

                cnt += 1
                if cnt % 1000 == 0:
                    print(cnt)

    f_test.close()

if __name__ == '__main__':
    check_save_path()
    generate_train_validation_dataset()
    generate_test_dataset()
    print('Successfully loading!')
