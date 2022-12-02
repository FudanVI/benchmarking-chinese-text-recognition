# Benchmarking-Chinese-Text-Recognition
![](https://img.shields.io/badge/Team-FudanVI-red) ![](https://img.shields.io/badge/Maintained-Yes-green) ![](https://img.shields.io/badge/License-MIT-blue)


This repository contains datasets and baselines for benchmarking Chinese text recognition. Please see the corresponding [paper](https://arxiv.org/abs/2112.15093) for more details regarding the datasets, baselines, the empirical study, etc.

## Highlights
:star2: All datasets are transformed to lmdb format for convenient usage.

:star2: The experimental results of all baselines are available at [link](https://github.com/FudanVI/benchmarking-chinese-text-recognition/tree/main/predictions) with format (*index* *[pred]* *[gt]*).

:star2: The code and trained weights of all baselines are available at [link](https://github.com/FudanVI/benchmarking-chinese-text-recognition/tree/main/model) for direct use.

## Updates
Dec 2, 2022: An updated version of the corresponding paper is available at arXiv.

Aug 22, 2022: We upload the lmdb datasets of hard cases.

Jun 15, 2022: The experimental settings are modified. We upload the code and trained weights of all baselines.

Jan 3, 2022: This repo is made publicly available. The corresponding paper is available at arXiv.

Nov 26, 2021: We upload the lmdb datasets publicly to Google Drive and BaiduCloud.

## Download
* The *lmdb* scene, web and document datasets are available in [BaiduCloud](https://pan.baidu.com/s/1OlAAvSOUl8mA2WBzRC8RCg) (psw:v2rm) and [GoogleDrive](https://drive.google.com/drive/folders/1J-3klWJasVJTL32FOKaFXZykKwN6Wni5?usp=sharing).

* The lmdb datasets of hard cases can be downloaded from [BaiduCloud](https://pan.baidu.com/s/1HjY_LuQPpBiol6Sc7noUDQ) (psw:n6nu) and [GoogleDrive](https://drive.google.com/drive/folders/1J-3klWJasVJTL32FOKaFXZykKwN6Wni5?usp=sharing); the lmdb dataset for examples of synthetic CTR data is available in [BaiduCloud](https://pan.baidu.com/s/1ON3mwSJyXiWxZ00DxoHCxA) (psw:c4sl).

* The lmdb dataset of hard cases can be downloaded from [BaiduCloud](https://pan.baidu.com/s/1HjY_LuQPpBiol6Sc7noUDQ) (psw:n6nu).

* For the handwriting setting, please first download it at [SCUT-HCCDoc](https://github.com/HCIILAB/SCUT-HCCDoc_Dataset_Release) and divide it into training, validation, and testing sets following [link](https://github.com/FudanVI/benchmarking-chinese-text-recognition/tree/main/data).

* We also collected HWDB2.0-2.2 and ICDAR2013 handwriting datasets from [CASIA](http://www.nlpr.ia.ac.cn/databases/handwriting/Download.html) and ICDAR2013 competition for futher research. Datasets are available at [BaiduCloud](https://pan.baidu.com/s/1q_x3L1lZBRykoY-AwhtoXw) (psw:lfaq) and [GoogleDrive](https://drive.google.com/drive/folders/1_xLYnEtoVo-RvPL9m79f0HgERwtR1Wc-?usp=sharing).

## Datasets
![Alt text](./images/dataset.png)
The image demonstrates the four datasets used in our benchmark including *Scene*, *Web*, *Document*, and *Handwriting* datasets, each of which is introduced next.

### Scene Dataset
We first collect the publicly available scene datasets including **RCTW**, **ReCTS**, **LSVT**, **ArT**, **CTW** resulting in 636,455 samples, which are randomly shuffled and then divided at a ratio of 8:1:1 to construct the training, validation, and testing datasets. Details of each scene datasets are introduced as follows:
- **RCTW** [1] provides 12,263 annotated Chinese text images from natural scenes. We derive 44,420 text lines from the training set and use them in our benchmark. The testing set of RCTW is not used as the text labels are not available. 
- **ReCTS** [2] provides 25,000 annotated street-view Chinese text images, mainly derived from natural signboards. We only adopt the training set and crop 107,657 text samples in total for our benchmark. 
- **LSVT** [3] is a large scale Chinese and English scene text dataset, providing 50,000 full-labeled (polygon boxes and text labels) and 400,000 partial-labeled (only one text instance each image) samples. We only utilize the full-labeled training set and crop 243,063 text line images for our benchmark.
- **ArT** [4] contains text samples captured in natural scenes with various text layouts (e.g., rotated text and curved texts). Here we obtain 49,951 cropped text images from the training set, and use them in our benchmark.
- **CTW** [5] contains annotated 30,000 street view images with rich diversity including planar, raised, and poorly-illuminated text images. Also, it provides not only character boxes and labels, but also character attributes like background complexity, appearance, etc. Here we crop 191,364 text lines from both the training and testing sets.

We combine all the subdatasets, resulting in 636,455 text samples. We randomly shuffle these samples and split them at a ratio of 8:1:1, leading to 509,164 samples for training, 63,645 samples for validation, and 63,646 samples for testing. 

### Web Dataset
To collect the web dataset, we utilize **MTWI [6]** that contains 20,000 Chinese and English web text images from 17 different categories on the Taobao website. The text samples are appeared in various scenes, typography and designs. We derive 140,589 text images from the training set, and manually divide them at a ratio of 8:1:1, resulting in 112,471 samples for training, 14,059 samples for validation, and 14,059 samples for testing.

### Document Dataset
We use the public repository **Text Render [7]** to generate some document-style synthetic text images. More specifically, we uniformly sample the length of text varying from 1 to 15. The corpus comes from wiki, films, amazon, and baike. The dataset contains 500,000 in total and is randomly divided into training, validation, and testing sets with a proportion of 8:1:1 (400,000 v.s. 50,000 v.s. 50,000).

### Handwriting Dataset
We collect the handwriting dataset based on **SCUT-HCCDoc [8]**, which captures the Chinese handwritten image with cameras in unconstrained environments. Following the official settings, we derive 93,254 text lines for training and 23,389 for testing, respectively. To pursue more rigorous research, we manually split the original training set into two sets at a ratio of 4:1, resulting in 74,603 samples for training and 18,651 samples for validation. For convenience, we continue to use the original 23,389 samples for testing.

Overall, the amount of text samples for each dataset is shown as follows:
<table><tbody>
    <tr>
        <th>&nbsp;&nbsp;Setting&nbsp;&nbsp;</th>
        <th>&nbsp;&nbsp;Dataset&nbsp;&nbsp;</th>
        <th>&nbsp;&nbsp;Sample Size&nbsp;&nbsp;</th>
        <th>&nbsp;&nbsp;Setting&nbsp;&nbsp;</th>
        <th>&nbsp;&nbsp;Dataset&nbsp;&nbsp;</th>
        <th>&nbsp;&nbsp;Sample Size&nbsp;&nbsp;</th>
    </tr>
    <tr>
        <td rowspan="3" align="center">Scene</td>
        <td align="center">Training</td>
        <td align="center">509,164</td>
        <td rowspan="3" align="center">Web</td>
        <td align="center">Training</td>
        <td align="center">112,471</td>
    </tr>
    <tr>
        <td align="center">Validation</td>
        <td align="center">63,645</td>
        <td align="center">Validation</td>
        <td align="center">14,059</td>
    </tr>
    <tr>
        <td align="center">Testing</td>
        <td align="center">63,646</td>
        <td align="center">Testing</td>
        <td align="center">14,059</td>
    </tr>
    <tr>
        <td rowspan="3" align="center">Document</td>
        <td align="center">Training</td>
        <td align="center">400,000</td>
        <td rowspan="3" align="center">Handwriting</td>
        <td align="center">Training</td>
        <td align="center">74,603</td>
    </tr>
    <tr>
        <td align="center">Validation</td>
        <td align="center">50,000</td>
        <td align="center">Validation</td>
        <td align="center">18,651</td>
    </tr>
    <tr>
        <td align="center">Testing</td>
        <td align="center">50,000</td>
        <td align="center">Testing</td>
        <td align="center">23,389</td>
    </tr>
</table>


## Baselines
We manually select six representative methods as baselines, which will be introduced as follows.

* **CRNN [9]** is a typical CTC-based method and it is widely used in academia and industry. It first sends the text image to a CNN to extract the image features, then adopts a two-layer LSTM to encode the sequential features. Finally, the output of LSTM is fed to a CTC (Connectionist Temperal Classification) decoder to maximize the probability of all the paths towards the ground truth. 

* **ASTER [10]** is a typical rectification-based method aiming at tackling irregular text images. It introduces a Spatial Transformer Network (STN) to rectify the given text image into a more recognizable appearance. Then the rectified text image is sent to a CNN and a two-layer LSTM to extract the features. In particular, ASTER takes advantage of the attention mechanism to predict the final text sequence. 

* **MORAN [11]** is a representative rectification-based method. It first adopts a multi-object rectification network (MORN) to predict rectified pixel offsets in a weak supervision way (distinct from ASTER that utilizes STN). The output pixel offsets are further used for generating the rectified image, which is further sent to the attention-based decoder (ASRN) for text recognition.

* **SAR [12]** is a representative method that takes advantage of 2-D feature maps for more robust decoding. In particular, it is mainly proposed to tackle irregular texts. On one hand, SAR adopts more powerful residual blocks in the CNN encoder for learning stronger image representation. On the other hand, different from CRNN, ASTER, and MORAN compressing the given image into a 1-D feature map, SAR adopts 2-D attention on the spatial dimension of the feature maps for decoding, resulting in a stronger performance in curved and oblique texts.

* **SEED [13]** is a representative semantics-based method. It introduces a semantics module to extract global semantics embedding and utilize it to initialize the first hidden state of the decoder. Specifically, while inheriting the structure of ASTER, the decoder of SEED intakes the semantic embedding to provide prior for the recognition process, thus showing superiority in recognizing low-quality text images.

* **TransOCR [14]** is one of the representative Transformer-based methods. It is originally designed to provide text priors for the super-resolution task. It employs ResNet-34 as the encoder and self-attention modules as the decoder. Distinct from the RNN-based decoders, the self-attention modules are more efficient to capture semantic features of the given text images.

Here are the results of the baselines on four datasets. ACC / NED follow the percentage format and decimal format, respectively. Please click the hyperlinks to see the detailed experimental results, following the format of (*index* *[pred]* *[gt]*).
<table><tbody>
    <tr>
        <th rowspan="2">&nbsp;&nbsp;Baseline&nbsp;&nbsp;</th>
        <th rowspan="2">&nbsp;&nbsp;Year&nbsp;&nbsp;</th>
        <th colspan="4">Dataset</th>
    </tr>
    <tr>
        <th align="center">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Scene&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</th>
        <th align="center">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Web&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</th>
        <th align="center">&nbsp;&nbsp;Document&nbsp;&nbsp;</th>
        <th align="center">&nbsp;Handwriting&nbsp;</th>
    </tr>
    <tr>
        <td align="center">CRNN [9]</td>
        <td align="center">2016</td>
        <td align="center"><a href="./predictions/CRNN/CRNN_scene.txt" >54.94 / 0.742</a></td>
        <td align="center"><a href="./predictions/CRNN/CRNN_web.txt" >56.21 / 0.745</a></td>
        <td align="center"><a href="./predictions/CRNN/CRNN_document.txt">97.41 / 0.995</a></td>
        <td align="center"><a href="./predictions/CRNN/CRNN_handwriting.txt">48.04 / 0.843</a></td>
    </tr>
    <tr>
        <td align="center">ASTER [10]</td>
        <td align="center">2018</td>
        <td align="center"><a href="./predictions/ASTER/ASTER_scene.txt">59.37 / 0.801</a></td>
        <td align="center"><a href="./predictions/ASTER/ASTER_web.txt">57.83 / 0.782</a></td>
        <td align="center"><a href="./predictions/ASTER/ASTER_document.txt">97.59 / 0.995</a></td>
        <td align="center"><a href="./predictions/ASTER/ASTER_handwriting.txt">45.90 / 0.819</a></td>
    </tr>
    <tr>
        <td align="center">MORAN [11]</td>
        <td align="center">2019</td>
        <td align="center"><a href="./predictions/MORAN/MORAN_scene.txt">54.68 / 0.710</a></td>
        <td align="center"><a href="./predictions/MORAN/MORAN_web.txt">49.64 / 0.679</a></td>
        <td align="center"><a href="./predictions/MORAN/MORAN_document.txt">91.66 / 0.984</a></td>
        <td align="center"><a href="./predictions/MORAN/MORAN_handwriting.txt">30.24 / 0.651</a></td>
    </tr>
    <tr>
        <td align="center">SAR [12]</td>
        <td align="center">2019</td>
        <td align="center"><a href="./predictions/SAR/SAR_scene.txt">53.80 / 0.738</a></td>
        <td align="center"><a href="./predictions/SAR/SAR_web.txt">50.49 / 0.705</a></td>
        <td align="center"><a href="./predictions/SAR/SAR_document.txt">96.23 / 0.993</a></td>
        <td align="center"><a href="./predictions/SAR/SAR_handwriting.txt" >30.95 / 0.732</a></td>
    </tr>
    <tr>
        <td align="center">SEED [13]</td>
        <td align="center">2020</td>
        <td align="center"><a href="./predictions/SEED/SEED_scene.txt">45.37 / 0.708</a></td>
        <td align="center"><a href="./predictions/SEED/SEED_web.txt">31.35 / 0.571</a></td>
        <td align="center"><a href="./predictions/SEED/SEED_document.txt">96.08 / 0.992</a></td>
        <td align="center"><a href="./predictions/SEED/SEED_handwriting.txt">21.10 / 0.555</a></td>
    </tr>
    <tr>
        <td align="center">TransOCR [14]</td>
        <td align="center">2021</td>
        <td align="center"><a href="./predictions/TransOCR/TransOCR_scene.txt">67.81 / 0.817</a></td>
        <td align="center"><a href="./predictions/TransOCR/TransOCR_web.txt">62.74 / 0.782</a></td>
        <td align="center"><a href="./predictions/TransOCR/TransOCR_document.txt">97.86 / 0.996</a></td>
        <td align="center"><a href="./predictions/TransOCR/TransOCR_handwriting.txt">51.67 / 0.835</a></td>
    </tr>
</table>

## References

### Datasets
[1] Shi B, Yao C, Liao M, et al. ICDAR2017 competition on reading chinese text in the wild (RCTW-17). ICDAR, 2017. 

[2] Zhang R, Zhou Y, Jiang Q, et al. Icdar 2019 robust reading challenge on reading chinese text on signboard. ICDAR, 2019. 

[3] Sun Y, Ni Z, Chng C K, et al. ICDAR 2019 competition on large-scale street view text with partial labeling-RRC-LSVT. ICDAR, 2019. 

[4] Chng C K, Liu Y, Sun Y, et al. ICDAR2019 robust reading challenge on arbitrary-shaped text-RRC-ArT. ICDAR, 2019. 

[5] Yuan T L, Zhu Z, Xu K, et al. A large chinese text dataset in the wild. Journal of Computer Science and Technology, 2019.

[6] He M, Liu Y, Yang Z, et al. ICPR2018 contest on robust reading for multi-type web images. ICPR, 2018. 

[7] text_render: [https://github.com/Sanster/text_renderer](https://github.com/Sanster/text_renderer)

[8] Zhang H, Liang L, Jin L. SCUT-HCCDoc: A new benchmark dataset of handwritten Chinese text in unconstrained camera-captured documents. Pattern Recognition, 2020. 


### Methods
[9] Shi B, Bai X, Yao C. An end-to-end trainable neural network for image-based sequence recognition and its application to scene text recognition. TPAMI, 2016.

[10] Shi B, Yang M, Wang X, et al. Aster: An attentional scene text recognizer with flexible rectification. TPAMI, 2018.

[11] Luo C, Jin L, Sun Z. Moran: A multi-object rectified attention network for scene text recognition. PR, 2019.

[12] Li H, Wang P, Shen C, et al. Show, attend and read: A simple and strong baseline for irregular text recognition. AAAI, 2019.

[13] Qiao Z, Zhou Y, Yang D, et al. Seed: Semantics enhanced encoder-decoder framework for scene text recognition. CVPR, 2020.

[14] Chen J, Li B, Xue X. Scene Text Telescope: Text-Focused Scene Image Super-Resolution. CVPR, 2021.

## Citation
Please consider citing this paper if you find it useful in your research. The bibtex-format citations of all relevant datasets and baselines are at [link](https://github.com/FudanVI/benchmarking-chinese-text-recognition/tree/main/bibtex).

```
@article{chen2021benchmarking,
  title={Benchmarking Chinese Text Recognition: Datasets, Baselines, and an Empirical Study},
  author={Chen, Jingye and Yu, Haiyang and Ma, Jianqi and Guan, Mengnan and Xu, Xixi and Wang, Xiaocong and Qu, Shaobo and Li, Bin and Xue, Xiangyang},
  journal={arXiv preprint arXiv:2112.15093},
  year={2021}
}
```

## Acknowledgements
We sincerely thank those researchers who collect the subdatasets for Chinese text recognition. Besides, we would like to thank Teng Fu,
Nanxing Meng, Ke Niu and Yingjie Geng for their feedbacks on this benchmark. 

## Copyright
The team includes Jingye Chen, Haiyang Yu, Jianqi Ma, Mengnan Guan, Xixi Xu, Xiaocong Wang, and Shaobo Qu, advised by Prof. [Bin Li](https://aimpressionist.github.io) and Prof. Xiangyang Xue.

Copyright © 2021 Fudan-FudanVI. All Rights Reserved.

![Alt text](./images/logo.png)
