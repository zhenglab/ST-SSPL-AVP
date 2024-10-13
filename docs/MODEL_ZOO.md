# Model Zoo

We provide the experimental results of the MGMA method in SSPL framework for AVP. We implement two MGMA-Nets, including the MGMA-ResNet (RNB, built upon the classic ResNet architecture) and the MGMA-ShuffleNet (SFNB, built upon the lightweight ShuffleNet architecture), used for conducting experiments. We present the performance (RMSE, MAE) and efficiency (Params, FLOPs) metrics of our MGMA-SSPL models in predicting ERA5 variables (T2M, UV10, R and TCC). All the models are trained for 50 epochs, and they can be downloaded via the [Google Drive](https://drive.google.com/file/d/1cPtZQz8UlrXZeZwEdNiht3RQC_fMF5hT/view?usp=drive_link) or [Baidu Netdisk](https://pan.baidu.com/s/11SuqwfydL8JLBBfk3zNXnw?pwd=avp1) links.

## The Efficiency of ST-SSPL Models

| Method | Params | FLOPs | FPS |
| :----: | :----: | :---: | :-: |
| SFNB-Base | 0.37M | 2.98G | 993 |
| RNB-Base | 0.48M | 3.66G | 931 |
| SFNB-MGMA | 0.55M | 4.14G | 407 |
| RNB-MGMA | 0.66M | 4.81G | 396 |

## The Results of ST-SSPL Models on Temperature (t2m)

| Method | Variable | RMSE | MAE | Config |
| :----: | :------: | :--: | :-: | :----: |
| SFNB-Base | t2m | 1.1154 | 0.7133 | [configs/weather/t2m_5_625/MGMA_ShuffleV2_NONE.py](../configs/weather/t2m_5_625/MGMA_ShuffleV2_NONE.py) |
| RNB-Base | t2m | 1.1348 | 0.7339 | [configs/weather/t2m_5_625/MGMA_Bottleneck_NONE.py](../configs/weather/t2m_5_625/MGMA_Bottleneck_NONE.py) |
| SFNB-MGMA | t2m | 1.0831 | 0.6760 | [configs/weather/t2m_5_625/MGMA_ShuffleV2.py](../configs/weather/t2m_5_625/MGMA_ShuffleV2.py) |
| RNB-MGMA | t2m | **1.0726** | **0.6689** | [configs/weather/t2m_5_625/MGMA_Bottleneck.py](../configs/weather/t2m_5_625/MGMA_Bottleneck.py) |

## The Results of ST-SSPL Models on Wind Component (uv10)

| Method | Variable | RMSE | MAE | Config |
| :----: | :------: | :--: | :-: | :----: |
| SFNB-Base | uv10 | 1.3692 | 0.9430 | [configs/weather/uv10_5_625/MGMA_ShuffleV2_NONE.py](../configs/weather/uv10_5_625/MGMA_ShuffleV2_NONE.py) |
| RNB-Base | uv10 | 1.3606 | 0.9381 | [configs/weather/uv10_5_625/MGMA_Bottleneck_NONE.py](../configs/weather/uv10_5_625/MGMA_Bottleneck_NONE.py) |
| SFNB-MGMA | uv10 | 1.2938 | 0.8660 | [configs/weather/uv10_5_625/MGMA_ShuffleV2.py](../configs/weather/uv10_5_625/MGMA_ShuffleV2.py) |
| RNB-MGMA | uv10 | **1.2855** | **0.8600** | [configs/weather/uv10_5_625/MGMA_Bottleneck.py](../configs/weather/uv10_5_625/MGMA_Bottleneck.py) |

## The Results of ST-SSPL Models on Humidity (r)

| Method | Variable | RMSE | MAE | Config |
| :----: | :------: | :--: | :-: | :----: |
| SFNB-Base | r | 5.8830 | 4.1028 | [configs/weather/r_5_625/MGMA_ShuffleV2_NONE.py](../configs/weather/r_5_625/MGMA_ShuffleV2_NONE.py) |
| RNB-Base | r | 5.8999 | 4.1061 | [configs/weather/r_5_625/MGMA_Bottleneck_NONE.py](../configs/weather/r_5_625/MGMA_Bottleneck_NONE.py) |
| SFNB-MGMA | r  | 5.6384 | **3.8036** | [configs/weather/r_5_625/MGMA_ShuffleV2.py](../configs/weather/r_5_625/MGMA_ShuffleV2.py) |
| RNB-MGMA | r | **5.6376** | 3.8242 | [configs/weather/r_5_625/MGMA_Bottleneck.py](../configs/weather/r_5_625/MGMA_Bottleneck.py) |

## The Results of ST-SSPL Models on Cloud Cover (tcc)

| Method | Variable | RMSE | MAE | Config |
| :----: | :------: | :--: | :-: | :----: |
| SFNB-Base | tcc | 0.2250 | 0.1588 | [configs/weather/tcc_5_625/MGMA_ShuffleV2_NONE.py](../configs/weather/tcc_5_625/MGMA_ShuffleV2_NONE.py) |
| RNB-Base | tcc | 0.2253  | 0.1577 | [configs/weather/tcc_5_625/MGMA_Bottleneck_NONE.py](../configs/weather/tcc_5_625/MGMA_Bottleneck_NONE.py) |
| SFNB-MGMA | tcc | **0.2150** | **0.1461** | [configs/weather/tcc_5_625/MGMA_ShuffleV2.py](../configs/weather/tcc_5_625/MGMA_ShuffleV2.py) |
| RNB-MGMA | tcc | **0.2150** | 0.1467 | [configs/weather/tcc_5_625/MGMA_Bottleneck.py](../configs/weather/tcc_5_625/MGMA_Bottleneck.py) |

