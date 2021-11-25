# Benchmarking-Chinese-Text-Recognition
This is a repository containing datasets and baselines for benchmarking Chinese text recognition. Please see the [paper]() for more details regarding the dataset divisions and experiments.

## Updates
Nov 25, 2021: We upload the Chinese text datasets used in our benchmark to BaiduCloud.

## Todo List
- [ ] Upload related datasets to BaiduCloud and Google Drive （**Jingye Chen**）
- [ ] Add description for the datasets （**Jianqi Ma**）
- [ ] Fix the experiment settings, e.g., dataset division, input size, etc. (**Jingye Chen**)
- [ ] Complete a series of baselines (**All**)
- [ ] Analyze the difference between Chinese texts and English texts（**Xixi Xu**）
- [ ] Write a paper, and publish it at arXiv!!!

## Datasets
### Scene
- [x] LSVT 
- [x] RCTW
- [x] CTW
- [x] ArT
- [x] ReCTS
### Web
- [x] MTWI
### Document
- [x] Synthesize500k
### Handwritting
- [x] ICDAR2013
- [x] SCUT
- [x] HWDB2.0-2.2

## Model List
- [x] CRNN(**Haiyang Yu**)
- [ ] ASTER(**Mengnan Guan and Shaobo Qu**)
- [x] MORAN(**Jingye Chen**)
- [x] SEED(**Mengnan Guan**)
- [x] SRN(**Xiaocong Wang**)
- [ ] Transformer-STR(**Shaobo Qu**)
- [ ] SAR(**Xixi Xu**)
- [ ] DAN(**Xiaocong Wang**)



## References
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



## Teammates
Jingye Chen, Mengnan Guan, Haiyang Yu, Shaobo Qu, Xiaocong Wang, Xixi Xu, Jianqi Ma

Advisors: Bin Li and Xiangyang Xue
