# Data
This directory contains some scripts to read, modify or make the datasets.

## Make lmdb-format handwriting dataset
1. Download the SCUT-HCCDoc dataset from [link](https://github.com/HCIILAB/SCUT-HCCDoc_Dataset_Release).
2. Modify two paths in ```divide_scut.py``` and run ```python divide_scut.py``` to generate the training, validation, and testing sets.
```python
dataset_path = 'The absolute path of SCUT-HCCDoc dataset' # eg, '/home/dataset/SCUT-HCCDoc_Dataset_Release_v2'
save_path = 'The empty directory for saving images' # eg, '/home/dataset/my_path'
```
3. Modify Line 87 and Line 104 in ```lmdbMaker.py``` and run ```python lmdbMaker.py``` in **python2** environment (the lmdbMaker script in python3 is coming soon).
Please note that ```gt.txt``` is stored in ```{save_path}/train_image```,```{save_path}/validation_image```, and ```{save_path}/test_image```.
```python
lines = open('.../gt.txt', 'r').readlines() # Line 87

createDataset('./path_for_saving_lmdb'.format(), imgList, labelList) # Line 104
```
4. Use ```lmdbReader.py``` to examine the generated lmdb-format datasets.

