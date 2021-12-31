# Benchmarking-Chinese-Text-Recognition
![](https://img.shields.io/badge/Team-FudanVI-red) ![](https://img.shields.io/badge/Maintained-Yes-green) ![](https://img.shields.io/badge/License-MIT-blue)


This repository contains datasets and baselines for benchmarking Chinese text recognition. Please see the corresponding [paper]() for more details regarding the datasets, baselines, the empirical study, etc.

## Highlights
:star2: All datasets are transformed to lmdb format for convenient usage.

:star2: The experimental results of all baselines are available and stored in txt format.

:star2: The code and trained weights of TransOCR (one of the baselines) are available for direct use.

## Updates
Jan 1, 2022: This repo is made publicly available.

Nov 31, 2021: The paper is submitted to arXiv. 

Nov 26, 2021: We upload the lmdb datasets publicly to Google Drive and BaiduCloud.

## Download
* The *lmdb* scene, web and document datasets are available in [BaiduCloud](https://pan.baidu.com/s/1OlAAvSOUl8mA2WBzRC8RCg) (psw:v2rm) and [GoogleDrive](https://drive.google.com/drive/folders/1J-3klWJasVJTL32FOKaFXZykKwN6Wni5?usp=sharing).

* For the handwriting setting, please first download it at [SCUT-HCCDoc](https://github.com/HCIILAB/SCUT-HCCDoc_Dataset_Release) and manually divide it into training, validation, and testing sets. Please transform three sets to *lmdb* formats following XXX.

Please use ```data/lmdbReader.py``` and ```data/lmdbMaker.py``` to read or make your own dataset.

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
        <td align="center"><a href="./predictions/CRNN/CRNN_scene.txt" style="color:black;">52.8</a></td>
        <td align="center"><a href="./predictions/CRNN/CRNN_web.txt" style="color:black;">54.1</a></td>
        <td align="center"><a href="./predictions/CRNN/CRNN_document.txt" style="color:black;">93.4</a></td>
        <td align="center">ing</td>
    </tr>
    <tr>
        <td align="center">ASTER [10]</td>
        <td align="center">2018</td>
        <td align="center"><a href="./predictions/ASTER/ASTER_scene.txt" style="color:black;">54.1</a></td>
        <td align="center"><a href="./predictions/ASTER/ASTER_web.txt" style="color:black;">52.0</a></td>
        <td align="center"><a href="./predictions/ASTER/ASTER_document.txt" style="color:black;">92.9</a></td>
        <td align="center">ing</td>
    </tr>
    <tr>
        <td align="center">MORAN [11]</td>
        <td align="center">2019</td>
        <td align="center"><a href="./predictions/MORAN/MORAN_scene.txt" style="color:black;">51.3</a></td>
        <td align="center"><a href="./predictions/MORAN/MORAN_web.txt" style="color:black;">49.6</a></td>
        <td align="center"><a href="./predictions/MORAN/MORAN_document.txt" style="color:black;">95.6</a></td>
        <td align="center"><a href="./predictions/MORAN/MORAN_handwriting.txt" style="color:black;">37.2</a></td>
    </tr>
    <tr>
        <td align="center">SAR [12]</td>
        <td align="center">2019</td>
        <td align="center"><a href="./predictions/SAR/SAR_scene.txt" style="color:black;">61.8</a></td>
        <td align="center"><a href="./predictions/SAR/SAR_web.txt" style="color:black;">54.0</a></td>
        <td align="center">ing</td>
        <td align="center">ing</td>
    </tr>
    <tr>
        <td align="center">SEED [13]</td>
        <td align="center">2020</td>
        <td align="center"><a href="./predictions/SEED/SEED_scene.txt" style="color:black;">49.2</a></td>
        <td align="center"><a href="./predictions/SEED/SEED_web.txt" style="color:black;">46.0</a></td>
        <td align="center"><a href="./predictions/SEED/SEED_document.txt" style="color:black;">93.6</a></td>
        <td align="center">ing</td>
    </tr>
    <tr>
        <td align="center">SRN [14]</td>
        <td align="center">2020</td>
        <td align="center"><a href="./predictions/SRN/SRN_scene.txt" style="color:black;">59.2</a></td>
        <td align="center"><a href="./predictions/SRN/SRN_web.txt" style="color:black;">49.7</a></td>
        <td align="center"><a href="./predictions/SRN/SRN_document.txt" style="color:black;">96.1</a></td>
        <td align="center">ing</td>
    </tr>
    <tr>
        <td align="center">TransOCR [15]</td>
        <td align="center">2021</td>
        <td align="center"><a href="./predictions/SRN/SRN_scene.txt" style="color:black;">59.2</a></td>
        <td align="center"><a href="./predictions/SRN/SRN_web.txt" style="color:black;">49.7</a></td>
        <td align="center"><a href="./predictions/SRN/SRN_document.txt" style="color:black;">96.1</a></td>
        <td align="center">ing</td>
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

[14] Yu D, Li X, Zhang C, et al. Towards accurate scene text recognition with semantic reasoning networks. CVPR, 2020.

[15] Chen J, Li B, Xue X. Scene Text Telescope: Text-Focused Scene Image Super-Resolution. CVPR, 2021.

## Citation
Please consider cite this paper if you find it useful in your research.

```
to be filled
```

## Acknowledgements


## Copyright
The team includes Jingye Chen **(Leader)**, Mengnan Guan, Haiyang Yu, Shaobo Qu, Xiaocong Wang, Xixi Xu, and Jianqi Ma, advised by Prof. Bin Li and Prof. Xiangyang Xue.

Copyright © 2021 Fudan-FudanVI. All Rights Reserved.

![Alt text](./images/logo.png)
