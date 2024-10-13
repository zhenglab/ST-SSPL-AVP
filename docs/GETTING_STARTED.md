# Getting Started

We use the t2m model as an example to show how the experiments are implemented, including the running commands for training, testing, and visualizing. The other models (wind speed, humidity, and total cloud cover) can be implemented by setting the corresponding config files. We use only one GPU for all the experiments. A detailed description of the arguments can be found in [parser.py](../lib/utils/parser.py).

#### Training

```shell
python tools/train.py -d weather_t2m_5_625 -c configs/weather/t2m_5_625/MGMA_Bottleneck.py --ex_name weather_t2m_5_625_MGMA_Bottleneck --epoch 50 --fps
```

#### Testing

```shell
python tools/test.py -d weather_t2m_5_625 -c configs/weather/t2m_5_625/MGMA_Bottleneck.py --ex_name weather_t2m_5_625_MGMA_Bottleneck
```

#### Visualizing

```shell
python tools/vis.py -d weather_t2m_5_625 -c configs/weather/t2m_5_625/MGMA_Bottleneck.py --ex_name weather_t2m_5_625_MGMA_Bottleneck 
```
