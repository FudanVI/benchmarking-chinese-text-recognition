# TransOCR

This is the code for TransOCR proposed in "Benchmarking Chinese Text Recognition: Some Baselines and Analyses".

## Dependencies
Create a virtual environment required by the code with the configuration file "TransOCR.yaml"
```python
conda env create -f TransOCR.yaml
```

## Pre-trained Model
Download the pre-trained model for each scenario at [BaiduYunDisk](https://pan.baidu.com/s/1SGuFrmNvim259FcwmuCyog) with password: qaiu


## Training
Please remember to modify the experiment name. You can execute the following command directly or execute the "train.sh" containing the command. So is testing.
```python
CUDA_VISIBLE_DEVICES=GPU_NUM python train.py --train_dataset Path_to_Training_Dataset --test_dataset Path_to_Testing_Dataset --alpha_path Path_to_Alphabet_File --exp_name EXP_NAME 
```

## Testing
```python
CUDA_VISIBLE_DEVICES=GPU_NUM python train.py --test_dataset Path_to_Test_Dataset --exp_name EXP_NAME --resume YOUR_MODEL --test_only
```

## Demo
```python
CUDA_VISIBLE_DEVICES=GPU_NUM python demo.py --image_path Path_to_Testing_Image --alpha_path Path_to_Alphabet_File --resume YOUR_MODEL
```