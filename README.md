# TradingRL

This project aims to apply Reinforcement Learning (Transformer-based Actor-Critic models) methodologies for optimal trade execution on Korean stock market.


## Project Goal

To be updated


## Preliminary

To begin with, clone this repo using the following command.
```
git clone https://github.com/JihoonBrianJun/TradingRL.git
cd AutoBinance
```

Python packages used for this project can be replicated by the following command.
```
pip install -r requirements.txt
```

## Data

Raw data files used for training in this project are stored within `data/KR` directory, and they are downloaded using the following command.
```
python3 data/download_yfin_.py
```

## Data Preprocessing Pipeline

To be updated


## Train result

You can train the model using the following command:
```
python3 train_rl.py
```

Model configurations are shown in the table below.

|Hidden dimension (Transformer)|# heads|# Enc/Dec layers (Each)|
|---|---|---|
|64|2|2|

Train hyperparameters is summarized in the following table.

|# epoch|Episode Sample Size|Batch Size|Learning Rate|Gamma (for StepLR)|
|---|---|---|---|
|100|1024|128|1e-5|0.999|

After training, metrics evaluated on validation data were as follows:

To-be-updated


## Metrics

To-be-updated