![GitHub Repo](https://img.shields.io/badge/Research-Paper-blue)
# **Fraud detection in financial statements using data mining and GAN models**
## 📜 Abstract
<p align="justify">
Financial statements are analytical reports published periodically by financial institutions explaining their performance from different perspectives. As these reports are the fundamental source for decision-making by many stakeholders, creditors, investors, and even auditors, some institutions may manipulate them to mislead people and commit fraud. Fraud detection in financial statements aims to discover anomalies caused by these distortions and discriminate fraud-prone reports from non-fraudulent ones. Although binary classification is one of the most popular data mining approaches in this area, it requires a standard labeled dataset, which is often unavailable in the real world due to the rarity of fraudulent samples. This paper proposes a novel approach based on the generative adversarial networks (GAN) and ensemble models that is able to not only resolve the lack of non-fraudulent samples but also handle the high-dimensionality of feature space. A new dataset is also constructed by collecting the annual financial statements of ten Iranian banks and then extracting three types of features suggested in this study. Experimental results on this dataset demonstrate that the proposed method performs well in generating synthetic fraud-prone samples. Moreover, it attains comparative performance with supervised models and better performance than unsupervised ones in accurately distinguishing fraud-prone samples.
</p>

## 📊 Results
<p align="justify">
  The table below shows the performance comparison of several outlier detection models in each segment of the data, in terms of precision, recall, and accuracy.

<table align="center">
  <thead>
    <tr>
      <th rowspan="3">Segment</th>
      <th rowspan="3">Model</th>
      <th colspan="5">Train</th>
      <th colspan="5">Test</th>
    </tr>
    <tr>
      <th colspan="2">Precision</th>
      <th colspan="2">Recall</th>
      <th rowspan="2">Accuracy</th>
      <th colspan="2">Precision</th>
      <th colspan="2">Recall</th>
      <th rowspan="2">Accuracy</th>
    </tr>
    <tr>
      <th> - </th>
      <th> + </th>
      <th> - </th>
      <th> + </th>
      <th> - </th>
      <th> + </th>
      <th> - </th>
      <th> + </th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="8">Loans</td>
      <td>iForest</td>
      <td>0.560</td>
      <td>0.971</td>
      <td>0.994</td>
      <td>0.219</td>
      <td>0.606</td>
      <td>0.592</td>
      <td>1</td>
      <td>1</td>
      <td>0.292</td>
      <td>0.652</td>
    </tr>
    <tr>
      <td>LSCP</td>
      <td>0.554</td>
      <td>0.943</td>
      <td>0.988</td>
      <td>0.212</td>
      <td>0.599</td>
      <td>0.556</td>
      <td>0.800</td>
      <td>1</td>
      <td>0.187</td>
      <td>0.600</td>
    </tr>
    <tr>
      <td>KNN</td>
      <td>0.478</td>
      <td>0.256</td>
      <td>0.880</td>
      <td>0.044</td>
      <td>0.461</td>
      <td>0.474</td>
      <td>0.233</td>
      <td>0.844</td>
      <td>0.0694</td>
      <td>0.460</td>
    </tr>
    <tr>
      <td>ECOD</td>
      <td>0.543</td>
      <td>1</td>
      <td>1</td>
      <td>0.166</td>
      <td>0.583</td>
      <td>0.600</td>
      <td>1</td>
      <td>1</td>
      <td>0.295</td>
      <td>0.652</td>
    </tr>
    <tr>
      <td>COPOD</td>
      <td>0.546</td>
      <td>0.971</td>
      <td>0.994</td>
      <td>0.179</td>
      <td>0.586</td>
      <td>0.558</td>
      <td>0.900</td>
      <td>0.976</td>
      <td>0.220</td>
      <td>0.600</td>
    </tr>
    <tr>
      <td>SVM</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <td>LR</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <td>XGBOD</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0.976</td>
      <td>0.976</td>
      <td>0.972</td>
      <td>0.978</td>
      <td>0.974</td>
    </tr>
    <tr>
      <td rowspan="8">Deposits</td>
      <td>iForest</td>
      <td>0.558</td>
      <td>1</td>
      <td>1</td>
      <td>0.205</td>
      <td>0.602</td>
      <td>0.570</td>
      <td>1</td>
      <td>1</td>
      <td>0.264</td>
      <td>0.628</td>
    </tr>
    <tr>
      <td>LSCP</td>
      <td>0.558</td>
      <td>1</td>
      <td>1</td>
      <td>0.205</td>
      <td>0.602</td>
      <td>0.556</td>
      <td>0.800</td>
      <td>1</td>
      <td>0.220</td>
      <td>0.610</td>
    </tr>
    <tr>
      <td>KNN</td>
      <td>0.526</td>
      <td>0.739</td>
      <td>0.948</td>
      <td>0.142</td>
      <td>0.546</td>
      <td>0.518</td>
      <td>0.633</td>
      <td>0.944</td>
      <td>0.136</td>
      <td>0.546</td>
    </tr>
    <tr>
      <td>ECOD</td>
      <td>0.560</td>
      <td>1</td>
      <td>1</td>
      <td>0.205</td>
      <td>0.602</td>
      <td>0.564</td>
      <td>0.800</td>
      <td>1</td>
      <td>0.239</td>
      <td>0.618</td>
    </tr>
    <tr>
      <td>COPOD</td>
      <td>0.560</td>
      <td>1</td>
      <td>1</td>
      <td>0.195</td>
      <td>0.597</td>
      <td>0.550</td>
      <td>0.800</td>
      <td>1</td>
      <td>0.189</td>
      <td>0.598</td>
    </tr>
    <tr>
      <td>SVM</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <td>LR</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <td>XGBOD</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0.968</td>
      <td>0.984</td>
      <td>0.976</td>
      <td>0.946</td>
      <td>0.970</td>
    </tr>
        <tr>
      <td rowspan="8">Incomes</td>
      <td>iForest</td>
      <td>0.552</td>
      <td>0.950</td>
      <td>0.990</td>
      <td>0.195</td>
      <td>0.592</td>
      <td>0.632</td>
      <td>1</td>
      <td>1</td>
      <td>0.397</td>
      <td>0.698</td>
    </tr>
    <tr>
      <td>LSCP</td>
      <td>0.558</td>
      <td>1</td>
      <td>1</td>
      <td>0.205</td>
      <td>0.602</td>
      <td>0.584</td>
      <td>0.800</td>
      <td>1</td>
      <td>0.275</td>
      <td>0.626</td>
    </tr>
    <tr>
      <td>KNN</td>
      <td>0.502</td>
      <td>0.494</td>
      <td>0.914</td>
      <td>0.087</td>
      <td>0.500</td>
      <td>0.504</td>
      <td>0.533</td>
      <td>0.906</td>
      <td>0.110</td>
      <td>0.504</td>
    </tr>
    <tr>
      <td>ECOD</td>
      <td>0.562</td>
      <td>1</td>
      <td>1</td>
      <td>0.215</td>
      <td>0.607</td>
      <td>0.606</td>
      <td>0.800</td>
      <td>1</td>
      <td>0.325</td>
      <td>0.646</td>
    </tr>
    <tr>
      <td>COPOD</td>
      <td>0.560</td>
      <td>1</td>
      <td>1</td>
      <td>0.206</td>
      <td>0.602</td>
      <td>0.584</td>
      <td>0.800</td>
      <td>1</td>
      <td>0.275</td>
      <td>0.626</td>
    </tr>
    <tr>
      <td>SVM</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <td>LR</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <td>XGBOD</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0.980</td>
      <td>1</td>
      <td>1</td>
      <td>0.982</td>
      <td>0.990</td>
    </tr>
    <tr>
      <td rowspan="8">Costs</td>
      <td>iForest</td>
      <td>0.530</td>
      <td>0.775</td>
      <td>0.954</td>
      <td>0.160</td>
      <td>0.556</td>
      <td>0.540</td>
      <td>0.900</td>
      <td>0.960</td>
      <td>0.186</td>
      <td>0.572</td>
    </tr>
    <tr>
      <td>LSCP</td>
      <td>0.556</td>
      <td>1</td>
      <td>1</td>
      <td>0.205</td>
      <td>0.602</td>
      <td>0.568</td>
      <td>1</td>
      <td>1</td>
      <td>0.236</td>
      <td>0.616</td>
    </tr>
    <tr>
      <td>KNN</td>
      <td>0.520</td>
      <td>0.693</td>
      <td>0.938</td>
      <td>0.140</td>
      <td>0.538</td>
      <td>0.522</td>
      <td>0.733</td>
      <td>0.922</td>
      <td>0.152</td>
      <td>0.532</td>
    </tr>
    <tr>
      <td>ECOD</td>
      <td>0.564</td>
      <td>1</td>
      <td>1</td>
      <td>0.222</td>
      <td>0.610</td>
      <td>0.588</td>
      <td>1</td>
      <td>1</td>
      <td>0.286</td>
      <td>0.636</td>
    </tr>
    <tr>
      <td>COPOD</td>
      <td>0.562</td>
      <td>1</td>
      <td>1</td>
      <td>0.216</td>
      <td>0.607</td>
      <td>0.580</td>
      <td>1</td>
      <td>1</td>
      <td>0.261</td>
      <td>0.626</td>
    </tr>
    <tr>
      <td>SVM</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <td>LR</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <td>XGBOD</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0.982</td>
      <td>1</td>
      <td>1</td>
      <td>0.980</td>
      <td>0.990</td>
    </tr>
  </tbody>
</table>


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
