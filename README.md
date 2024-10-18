# TimeGAN-PyTorch
  Unofficial implementation of TimeGAN (Yoon et al., NIPS 2019) in PyTorch.

  Full reference of the authors' work:
  
  Jinsung Yoon, Daniel Jarrett, Mihaela van der Schaar, 
  "Time-series Generative Adversarial Networks," 
  Neural Information Processing Systems (NeurIPS), 2019.
  
  Link to their paper:
  
  https://papers.nips.cc/paper/8789-time-series-generative-adversarial-networks

### Motivation
  The initial TimeGAN code is based on Tensorflow 1, now being deprecated.

  Original TimeGAN codebase:

  https://github.com/jsyoon0823/TimeGAN.git

### Results

##### Distribution Estimate (PCA)
![result_pca](../assets/pca.png)

##### Distribution Estimate (TSNE)
![result_tsne](../assets/tsne.png)

### Notebook
  The provided notebook is organized in chapters of subsequent tasks. It aims at presenting the complex framework in a simple and comprehensible way.

##### GPU-Support
  As extensive training over a considerable number of iterations (epochs) demands vast computional power it is advisable and practical to rely on GPU (or other       resources that offer accelerated computations).

  For example, Google Colab (https://colab.research.google.com/) grants limited access to GPU resources free of charge.
