# TimeGAN-PyTorch
Unofficial implementation of TimeGAN (Yoon et al., NIPS 2019) in PyTorch 2.

Jinsung Yoon, Daniel Jarrett, Mihaela van der Schaar, 
"Time-series Generative Adversarial Networks," 
Neural Information Processing Systems (NeurIPS), 2019

Original Codebase: https://github.com/jsyoon0823/TimeGAN.git

### Data Set Reference
-  Stock data: https://finance.yahoo.com/quote/GOOG/history?p=GOOG
-  Energy data: http://archive.ics.uci.edu/ml/datasets/Appliances+energy+prediction

### Version Notes
The model was implemented and tested using `Python==3.11.9`. Further, the following modules were utilized (see [Requirements File](./requirements.txt)):
```
matplotlib==3.10.0
numpy==2.2.1
pandas==2.2.3
scikit-learn==1.6.0
torch==2.5.1
tqdm==4.67.1
```

### Usage
To conduct the experiments, the easiest way to get started is by cloning this repository and use the [notebook](./timegan.ipynb).

Alternatively, run it from the terminal.
```bash
py main_timegan.py --data=data/stock_data.csv --seq_len=24 --module=gru --hidden_dim=24 --num_layers=3 --epochs=10000 --batch_size=128 --metric_iteration=10 --learning_rate=1e-3
```

### Results

##### Stock Data

1. **Visualization**

<p float="left">
  <img src="../assets/pca.png" alt="PCA plot" width="400" />
  <img src="../assets/tsne.png" alt=="TSNE plot" width="400" />
</p>
