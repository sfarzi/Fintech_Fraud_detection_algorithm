![GitHub Repo](https://img.shields.io/badge/Research-Paper-blue)
# **Fraud detection in financial statements using data mining and GAN models**
## 📜 Abstract
<p align="justify">
Financial statements are analytical reports published periodically by financial institutions explaining their performance from different perspectives. As these reports are the fundamental source for decision-making by many stakeholders, creditors, investors, and even auditors, some institutions may manipulate them to mislead people and commit fraud. Fraud detection in financial statements aims to discover anomalies caused by these distortions and discriminate fraud-prone reports from non-fraudulent ones. Although binary classification is one of the most popular data mining approaches in this area, it requires a standard labeled dataset, which is often unavailable in the real world due to the rarity of fraudulent samples. This paper proposes a novel approach based on the generative adversarial networks (GAN) and ensemble models that is able to not only resolve the lack of non-fraudulent samples but also handle the high-dimensionality of feature space. A new dataset is also constructed by collecting the annual financial statements of ten Iranian banks and then extracting three types of features suggested in this study. Experimental results on this dataset demonstrate that the proposed method performs well in generating synthetic fraud-prone samples. Moreover, it attains comparative performance with supervised models and better performance than unsupervised ones in accurately distinguishing fraud-prone samples.
</p>

## 📊 Results
<p align="justify">
  The table below shows the performance comparison of several outlier detection models in each segment of the data, in terms of precision, recall, and accuracy.

| Segment  | Model   | Train Precision (-) | Train Precision (+) | Train Recall (-) | Train Recall (+) | Train Accuracy | Test Precision (-) | Test Precision (+) | Test Recall (-) | Test Recall (+) | Test Accuracy |
|----------|---------|---------------------|---------------------|------------------|------------------|----------------|--------------------|--------------------|-----------------|-----------------|---------------|
| Loans    | iForest | 0.560 | 0.971 | 0.994 | 0.219 | 0.606 | 0.592 | 1 | 1 | 0.292 | 0.652 | 
|          | LSCP    | 0.554 | 0.943 | 0.988 | 0.212 | 0.599 | 0.556 | 0.800 | 1 | 0.187 | 0.600 | 
|          | KNN     | 0.478 | 0.256 | 0.880 | 0.044 | 0.461 | 0.474 | 0.233 | 0.844 | 0.0694 | 0.460 | 
|          | ECOD    | 0.543 | 1 | 1 | 0.166 | 0.583 | 0.600 | 1 | 1 | 0.295 | 0.652 |
|          | COPOD   | 0.546 | 0.971 | 0.994 | 0.179 | 0.586 | 0.558 | 0.900 | 0.976 | 0.220 | 0.600 |
|          | SVM     | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
|          | LR      | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
|          | XGBOD   | 1 | 1 | 1 | 1 | 1 | 0.976 | 0.976 | 0.972 | 0.978 | 0.974 |
| Deposits | iForest | 0.558 | 1 | 1 | 0.205 | 0.602 | 0.570 | 1 | 1 | 0.264 | 0.628 | 
|          | LSCP    | 0.558 | 1 | 1 | 0.205 | 0.602 | 0.556 | 0.800 | 1 | 0.220 | 0.610 |
|          | KNN     | 0.526 | 0.739 | 0.948 | 0.142 | 0.546 | 0.518 | 0.633 | 0.944 | 0.136 | 0.546 |
|          | ECOD    | 0.560 | 1 | 1 | 0.205 | 0.602 | 0.564 | 0.800 | 1 | 0.239 | 0.618 |
|          | COPOD   | 0.560 | 1 | 1 | 0.195 | 0.597 | 0.550 | 0.800 | 1 | 0.189 | 0.598 |
|          | SVM     | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
|          | LR      | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
|          | XGBOD   | 1 | 1 | 1 | 1 | 1 | 0.968 | 0.984 | 0.976 | 0.946 | 0.970 |
| Incomes  | iForest | 0.552 | 0.950 | 0.990 | 0.195 | 0.592 | 0.632 | 1 | 1 | 0.397 | 0.698 |
|          | LSCP    | 0.558 | 1 | 1 | 0.205 | 0.602 | 0.584 | 0.800 | 1 | 0.275 | 0.626 |
|          | KNN     | 0.502 | 0.494 | 0.914 | 0.087 | 0.500 | 0.504 | 0.533 | 0.906 | 0.110 | 0.504 |
|          | ECOD    | 0.562 | 1 | 1 | 0.215 | 0.607 | 0.606 | 0.800 | 1 | 0.325 | 0.646 |
|          | COPOD   | 0.560 | 1 | 1 | 0.206 | 0.602 | 0.584 | 0.800 | 1 | 0.275 | 0.626 |
|          | SVM     | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
|          | LR      | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
|          | XGBOD   | 1 | 1 | 1 | 1 | 1 | 0.980 | 1 | 1 | 0.982 | 0.990 |
| Costs    | iForest | 0.530 | 0.775 | 0.954 | 0.160 | 0.556 | 0.540 | 0.900 | 0.960 | 0.186 | 0.572 |
|          | LSCP    | 0.556 | 1 | 1 | 0.205 | 0.602 | 0.568 | 1 | 1 | 0.236 | 0.616 |
|          | KNN     | 0.520 | 0.693 | 0.938 | 0.140 | 0.538 | 0.522 | 0.733 | 0.922 | 0.152 | 0.532 |
|          | ECOD    | 0.564 | 1 | 1 | 0.222 | 0.610 | 0.588 | 1 | 1 | 0.286 | 0.636 |
|          | COPOD   | 0.562 | 1 | 1 | 0.216 | 0.607 | 0.580 | 1 | 1 | 0.261 | 0.626 |
|          | SVM     | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
|          | LR      | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
|          | XGBOD   | 1 | 1 | 1 | 1 | 1 | 0.982 | 1 | 1 | 0.980 | 0.990 |

**Notes:**  
- `-` represents the non-fraudulent class.  
- `+` represents the fraudulent class.  
</p>

## 📌 Citation

If you use this work, please cite our [paper](https://www.sciencedirect.com/science/article/abs/pii/S0957417423006462) as follows:

```bibtex
@article{AFTABI2023120144,
author = {Seyyede Zahra Aftabi and Ali Ahmadi and Saeed Farzi},
title = {Fraud detection in financial statements using data mining and GAN models},
journal = {Expert Systems with Applications},
volume = {227},
pages = {120144},
year = {2023},
issn = {0957-4174},
doi = {https://doi.org/10.1016/j.eswa.2023.120144},
url = {https://www.sciencedirect.com/science/article/pii/S0957417423006462},
}
```
