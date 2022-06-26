# Baseline
We manually select six representative methods as baselines and refer to the following off-the-shelf PyTorch implementations to complete all experiments.
* CRNN: https://github.com/meijieru/crnn.pytorch
* ASTER: https://github.com/ayumiymk/aster.pytorch
* MORAN: https://github.com/Canjie-Luo/MORAN_v2
* SAR: https://github.com/liuch37/sar-pytorch
* SEED: https://github.com/Pay20Y/SEED
* TransOCR: https://github.com/FudanVI/FudanOCR/tree/main/scene-text-telescope

## Dependencies
Use the configuration file "model_name.yaml" to create the environment of the corresponding baseline. When creating the environment for CRNN, the [warp_ctc_pytorch](https://github.com/SeanNaren/warp-ctc/tree/pytorch_bindings/pytorch_binding) should be installed additionally.
```python
conda env create -f model_name/model_name.yaml
```

## Pre-trained Model
Download the trained weights of each baseline at [GoogleDrive](https://drive.google.com/drive/folders/14v3hHhq4AOVEYY1hQfA1d1vI2HtZQZey?usp=sharing)

## Training
Use the following command for training. Please remember to modify the parameters (*e.g.*, the path to training datasets, the experimental name, etc.).
```python
sh model_name/train.sh
```

## Testing
Use the following command for testing.
```python
sh model_name/test.sh
```