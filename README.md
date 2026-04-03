# MSU-TL: U-Net-Driven EEG Epilepsy Detection under Mask Self-supervised and Transfer Learning Paradigms

```
    Mingzhen Wen¹,  Hailing Wang¹*,  Jinghao Liu¹,  Jizhen Luo¹,  Lei Meng¹,  Wei Sun²*
¹ Shanghai University of Engineering Science, School of Electric and Electronic Engineering, Shanghai, China
² Institute of Software, Chinese Academy of Sciences, Beijing, China 
```

## Abstract: 
 Epilepsy is a neurological disorder characterized by abnormal neuronal discharges in the brain, typically diagnosed via electroencephalography (EEG). Predicting epileptic seizures is crucial for alleviating patient burden and informing clinical interventions. However, developing supervised models for seizure prediction demands extensive labeled data, which is labor-intensive and costly to acquire. Moreover, due to the high inter-subject variability of EEG signals, such models often suffer from poor generalization in real-world cross-subject scenarios. To address these challenges, we propose MSU-TL, a U-Net-based framework for EEG-based epilepsy detection that integrates mask-based self-supervised learning and transfer learning. Unlike conventional self-supervised methods, our approach incorporates multi-scale feature extraction within the U-Net architecture and employs a masked reconstruction module, enabling the model to learn transferable and robust representations from unlabeled EEG data. After pre-training, the model parameters are frozen, and a lightweight Multilayer Perceptron (MLP) classifier is appended for downstream epilepsy detection. Evaluated on the CHB-MIT database, MSU-TL achieved accuracies of 94.79% in within-subject experiments and 88.62% in cross-subject settings. Notably, to emulate real-world scenarios where limited labeled data from a few subjects must generalize to many, we conducted few-sample experiments, in which MSU-TL continued to demonstrate strong performance compared with state-of-art methods.

## Overall Architecture
![Overall Architecture](https://github.com/ONE-WMZ/MSU-TL/blob/main/Picture/Overall%20Architecture.png)

## Pretrained models
![Pretrained models]()
