# Preparation

## Installation

<details close>
<summary>Requirements</summary>

* Linux (Windows is not officially supported)
* Python 3.7+
* PyTorch 1.8 or higher
* CUDA 10.1 or higher
* NCCL 2
* GCC 4.9 or higher
</details>

<details close>
<summary>Dependencies</summary>

* dask
* decord
* future
* fvcore
* hickle
* lpips
* matplotlib
* nni
* netcdf4
* numpy
* opencv-python
* packaging
* pandas
* scikit-image<=0.19.3
* six
* scikit-learn
* timm>=0.5.4,<=0.6.11
* torch
* torchvision
* tqdm
* xarray==0.19.0
</details>

#### Clone the ST-SSPL-AVP repository

```shell
git clone https://github.com/zhenglab/ST-SSPL-AVP
```

#### Install the Python and PyTorch Environments

Install the corresponding versions of [Python](https://www.anaconda.com) and [PyTorch](https://pytorch.org), and also setup the conda environment.

```shell
conda env create -f environment.yml
conda activate st_sspl_avp
```

#### Install the Dependency Packages
```shell
python setup.py develop
```

## Dataset Preparation

- Download the corresponding datasets of ERA5 via [WeatherBench Github Repo](https://github.com/pangeo-data/WeatherBench).

- Unzip and copy the dataset files to `$ST-SSPL-AVP/data` directory as following shows:

```
ST-SSPL-AVP
├── configs
└── data
    |── weather
    |   ├── 2m_temperature
    |   ├── 10m_u_component_of_wind
    |   ├── 10m_v_component_of_wind
    |   ├── relative_humidity
    |   ├── total_cloud_cover
```
