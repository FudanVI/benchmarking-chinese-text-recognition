# Benchmarking-Chinese-Text-Recognition
![](https://img.shields.io/badge/Team-FudanVI-red) ![](https://img.shields.io/badge/Maintained-Yes-green) ![](https://img.shields.io/badge/License-MIT-blue)


This repository contains datasets and baselines for benchmarking Chinese text recognition. Please see the corresponding [paper]() for more details regarding the dataset divisions, experiments, etc.


## Updates
Nov 26, 2021: We make the datasets publicly available in Google Drive.

Nov 25, 2021: We upload the Chinese text datasets used in our benchmark to BaiduCloud.

## Todo List
- [x] Upload related datasets to BaiduCloud and Google Drive.
- [ ] Add description for the datasets (**Jianqi Ma**)
- [ ] Fix the experiment settings, e.g., dataset division, input size, etc. (**Jingye Chen**)
- [ ] Complete a series of baselines (**All**)
- [ ] Analyze the difference between Chinese texts and English texts(**Xixi Xu**)
- [ ] Train a model to provide the baseline, and publish the code and pre-trained weight (**TBD**)
- [ ] Write a paper, and publish it at arXiv!!!

## Download
The *lmdb* datasets for three benchmark settings (scene/web/document) are available in [BaiduCloud](https://pan.baidu.com/s/1OlAAvSOUl8mA2WBzRC8RCg) (psw:v2rm) and [GoogleDrive](https://drive.google.com/drive/folders/1J-3klWJasVJTL32FOKaFXZykKwN6Wni5?usp=sharing).
For the handwriting setting, please first download the dataset at [SCUT-HCCDoc](https://github.com/HCIILAB/SCUT-HCCDoc_Dataset_Release), then manually divide it and construct *lmdb* files following XXX.

The *lmdb* sub-datasets (RCTW/ReCTS/LSVT/ArT/CTW) used to construct the scene settings are available in [BaiduCloud](https://pan.baidu.com/s/1OSDNyb_6f1mJVtQad7SdhA) (psw:gkta) and [GoogleDrive](https://drive.google.com/drive/folders/1oYZPLjTADqmrS2cqeZgURhG2KT2sCCIh?usp=sharing).

Please use ```data/lmdbReader.py``` and ```data/lmdbMaker.py``` to read or make your own dataset.




## Datasets
![Alt text](./images/dataset.png)
The above image demonstrates the four settings used in our benchmark including *Scene*, *Web*, *Document*, and *Handwriting* settings, each of which will be detailed next.

### Scene Setting
We first collect the publicly available scene datasets including **RCTW**, **ReCTS**, **LSVT**, **ArT**, **CTW** resulting in 636,455 samples, which are randomly shuffled and then divided at a ratio of 8:1:1 to construct the training, validation, and testing datasets. Details of each scene datasets are introduced as follows:
- **RCTW** [] : 
- **ReCTS** [] :
- **LSVT** [] :
- **ArT** [] :
- **CTW** [] :

### Web Setting
We collect the web text images from the [ICPR-MTWI Competition](https://tianchi.aliyun.com/competition/entrance/231684/information) held in 2018. Since the labels for the testing dataset are not available, we randomly shuffle the training cropped text images and divide it at a ratio of 8:1:1 to construct the training, validation, and testing datasets. **MTWI** [] is xxx ...

### Document Setting
Since there does not exist any suitable publicly available document text datasets, we manually construct **Document50k** using the text render [x]. The text render ... We collect the corpus from ... The length of each text ... 

### Handwriting Setting
We utilize the **SCUT-HCCDoc** [x] dataset for this setting. The original SCUT-HCCDoc contains 93,254 samples for training and 23,389 for testing. To fairly evaluate the effectiveness of the baselines, we choose the last 13,254 samples in the original training dataset for validation. The SCUT-HCCDoc is xxx ...


Overall, the samples size for each dataset is shown as follows:
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
        <td align="center">63,645</td>
        <td align="center">Testing</td>
        <td align="center">14,059</td>
    </tr>
    <tr>
        <td rowspan="3" align="center">Document</td>
        <td align="center">Training</td>
        <td align="center">400,000</td>
        <td rowspan="3" align="center">Handwriting</td>
        <td align="center">Training</td>
        <td align="center">80,000</td>
    </tr>
    <tr>
        <td align="center">Validation</td>
        <td align="center">50,000</td>
        <td align="center">Validation</td>
        <td align="center">13,254</td>
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
    <tr bgcolor="#CCCCCC">
        <th rowspan="2">&nbsp;&nbsp;Method&nbsp;&nbsp;</th>
        <th rowspan="2">&nbsp;&nbsp;Source&nbsp;&nbsp;</th>
        <th rowspan="2">&nbsp;&nbsp;Time&nbsp;&nbsp;</th>
        <th colspan="4">Dataset</th>
    </tr>
    <tr bgcolor="#CCCCCC">
        <th align="center">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Scene&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</th>
        <th align="center">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Web&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</th>
        <th align="center">&nbsp;&nbsp;Document&nbsp;&nbsp;</th>
        <th align="center">&nbsp;Handwriting&nbsp;</th>
    </tr>
    <tr>
        <td align="center">CRNN [11]</td>
        <td align="center">TPAMI</td>
        <td align="center">2016</td>
        <td align="center">52.8</td>
        <td align="center">54.1</td>
        <td align="center">93.4</td>
        <td align="center">ing</td>
    </tr>
    <tr>
        <td align="center">ASTER [x]</td>
        <td align="center">TPAMI</td>
        <td align="center">2018</td>
        <td align="center"><a href="./predictions/ASTER/ASTER_scene.txt" style="color:black;">54.1</a></td>
        <td align="center"><a href="./predictions/ASTER/ASTER_web.txt" style="color:black;">52.0</a></td>
        <td align="center"><a href="./predictions/ASTER/ASTER_document.txt" style="color:black;">92.9</a></td>
        <td align="center">ing</td>
    </tr>
    <tr bgcolor="#CCCCCC">
        <td align="center">MORAN [12]</td>
        <td align="center">PR</td>
        <td align="center">2019</td>
        <td align="center"><a href="./predictions/MORAN/MORAN_scene.txt" style="color:black;">51.3</a></td>
        <td align="center"><a href="./predictions/MORAN/MORAN_web.txt" style="color:black;">49.6</a></td>
        <td align="center"><a href="./predictions/MORAN/MORAN_document.txt" style="color:black;">95.6</a></td>
        <td align="center">ing</td>
    </tr>
    <tr>
        <td align="center">SEED [13]</td>
        <td align="center">CVPR</td>
        <td align="center">2020</td>
        <td align="center"><a href="./predictions/SEED/SEED_scene.txt" style="color:black;">49.2</a></td>
        <td align="center"><a href="./predictions/SEED/SEED_web.txt" style="color:black;">46.0</a></td>
        <td align="center"><a href="./predictions/SEED/SEED_document.txt" style="color:black;">93.6</a></td>
        <td align="center">ing</td>
    </tr>
    <tr  bgcolor="#CCCCCC">
        <td align="center">SRN [14]</td>
        <td align="center">CVPR</td>
        <td align="center">2020</td>
        <td align="center"><a href="./predictions/SRN/SRN_scene.txt" style="color:black;">59.2</a></td>
        <td align="center"><a href="./predictions/SRN/SRN_web.txt" style="color:black;">49.7</a></td>
        <td align="center"><a href="./predictions/SRN/SRN_document.txt" style="color:black;">96.1</a></td>
        <td align="center">ing</td>
    </tr>
</table>

## References

### Datasets
[1] Yuan T L, Zhu Z, Xu K, et al. A large chinese text dataset in the wild. Journal of Computer Science and Technology, 2019. [paper](https://github.com/FudanVI/benchmarking-chinese-text-recognition/blob/main/papers/Scene/%E3%80%90CTW%E3%80%91(JCS2019)A%20Large%20Chinese%20Text%20Dataset%20in%20the%20Wild.pdf)

[2] Chng C K, Liu Y, Sun Y, et al. ICDAR2019 robust reading challenge on arbitrary-shaped text-RRC-ArT. ICDAR, 2019. [paper](https://github.com/FudanVI/benchmarking-chinese-text-recognition/blob/main/papers/Scene/%E3%80%90ArT%E3%80%91(ICDAR2019)ICDAR2019%20Robust%20Reading%20Challenge%20on%20Arbitrary-Shaped%20Text%20-%20RRC-ArT.pdf)

[3] Sun Y, Ni Z, Chng C K, et al. ICDAR 2019 competition on large-scale street view text with partial labeling-RRC-LSVT. ICDAR, 2019. [paper](https://github.com/FudanVI/benchmarking-chinese-text-recognition/blob/main/papers/Scene/%E3%80%90LSVT%E3%80%91(ICDAR2019)ICDAR%202019%20Competition%20on%20Large-scale%20Street%20View%20Text%20with%20Partial%20Labeling%20-%20RRC-LSVT.pdf)

[4] Shi B, Yao C, Liao M, et al. ICDAR2017 competition on reading chinese text in the wild (RCTW-17). ICDAR, 2017. [paper](https://github.com/FudanVI/benchmarking-chinese-text-recognition/blob/main/papers/Scene/%E3%80%90RCTW%E3%80%91(ICDAR2017)ICDAR2017%20Competition%20on%20Reading%20Chinese%20Text%20in%20the%20Wild%20(RCTW-17).pdf)

[5] Zhang R, Zhou Y, Jiang Q, et al. Icdar 2019 robust reading challenge on reading chinese text on signboard. ICDAR, 2019. [paper](https://github.com/FudanVI/benchmarking-chinese-text-recognition/blob/main/papers/Scene/%E3%80%90ReCTS%E3%80%91(ICDAR2019)ICDAR%202019%20Robust%20Reading%20Challenge%20on%20Reading%20Chinese%20Text%20on%20Signboard.pdf)

[6] He M, Liu Y, Yang Z, et al. ICPR2018 contest on robust reading for multi-type web images. ICPR, 2018. [paper](https://github.com/FudanVI/benchmarking-chinese-text-recognition/blob/main/papers/Web/%E3%80%90MTWI%E3%80%91(ICPR2018)ICPR2018%20Contest%20on%20Robust%20Reading%20for%20Multi-Type%20Web%20Images.pdf)

[7] Zhang H, Liang L, Jin L. SCUT-HCCDoc: A new benchmark dataset of handwritten Chinese text in unconstrained camera-captured documents. Pattern Recognition, 2020. [paper](https://github.com/FudanVI/benchmarking-chinese-text-recognition/blob/main/papers/Handwriting/%E3%80%90SCUT%E3%80%91(PR2020)SCUT-HCCDoc-%20A%20new%20benchmark%20dataset%20of%20handwritten%20Chinese%20text%20in%20unconstrained%20camera-captured%20documents.pdf)

[8] Yin F, Wang Q F, Zhang X Y, et al. ICDAR 2013 Chinese handwriting recognition competition. ICDAR, 2013. [paper](https://github.com/FudanVI/benchmarking-chinese-text-recognition/blob/main/papers/Handwriting/%E3%80%90ICDAR2013%E3%80%91(ICDAR2013)ICDAR%202013%20Chinese%20Handwriting%20Recognition%20Competition%20.pdf)

[9] Liu C L, Yin F, Wang D H, et al. CASIA online and offline Chinese handwriting databases. ICDAR, 2011. [paper](https://github.com/FudanVI/benchmarking-chinese-text-recognition/blob/main/papers/Handwriting/%E3%80%90HWDB%E3%80%91(ICDAR2011)CASIA%20Online%20and%20Offline%20Chinese%20Handwriting%20Databases.pdf)

[10] text_render: [https://github.com/Sanster/text_renderer](https://github.com/Sanster/text_renderer)

### Methods
[11] Shi B, Bai X, Yao C. An end-to-end trainable neural network for image-based sequence recognition and its application to scene text recognition. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2016.

[12] Luo C, Jin L, Sun Z. Moran: A multi-object rectified attention network for scene text recognition. Pattern Recognition, 2019.

[13] Qiao Z, Zhou Y, Yang D, et al. Seed: Semantics enhanced encoder-decoder framework for scene text recognition. CVPR, 2020.

[14] Yu D, Li X, Zhang C, et al. Towards accurate scene text recognition with semantic reasoning networks. CVPR, 2020.

## Citation
Please cite the following paper if you are using the code/model in your research paper.

```
to be filled
```


## Copyright
The team includes Jingye Chen **(Leader)**, Mengnan Guan, Haiyang Yu, Shaobo Qu, Xiaocong Wang, Xixi Xu, and Jianqi Ma, advised by Prof. Bin Li and Prof. Xiangyang Xue.

Copyright © 2021 Fudan-FudanVI. All Rights Reserved.

![Alt text](./images/logo.png)
