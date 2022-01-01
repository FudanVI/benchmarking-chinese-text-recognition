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
Execute the following command for training. Please remember to modify the experiment name.
```python
CUDA_VISIBLE_DEVICES=GPU_NUM python train.py --train_dataset Path_to_Training_Dataset --test_dataset Path_to_Testing_Dataset --alpha_path Path_to_Alphabet_File --exp_name EXP_NAME --scenario CORRESPONDING_SCENARIO
```

We set the input size of scene, web, document, handwriting datasets to 64×200, 64×200, 64×800, 64×1200. Therefore, the proper input size should be set when testing the corresponding dataset. Please note that only the first down-sampling layer of backbone is used in web and scene scenarios while all down-sampling layers are used in document and scut scenarios.
## Testing
```python
CUDA_VISIBLE_DEVICES=GPU_NUM python train.py --test_dataset Path_to_Test_Dataset --imageH Height_of_Input_Image --imageW Width_of_Input_Image --exp_name EXP_NAME --resume YOUR_MODEL --scenario CORRESPONDING_SCENARIO --test_only
```

## Demo
```python
CUDA_VISIBLE_DEVICES=GPU_NUM python demo.py --image_path Path_to_Testing_Image --imageH Height_of_Input_Image --imageW Width_of_Input_Image --alpha_path Path_to_Alphabet_File --resume YOUR_MODEL --scenario CORRESPONDING_SCENARIO
```