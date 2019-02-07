# Pytorch-OCR-Fully-Convolutional

This repository is a work in progress implementation of: 
[Accurate, Data-Efficient, Unconstrained Text Recognition with Convolutional Neural Networks](https://arxiv.org/abs/1812.11894) I am not affiliated with the authors of the paper. 

I also implemented the excellent data generator of [@Belval](https://github.com/Belval/TextRecognitionDataGenerator) as a pytorch dataset. I rely on the pytorch implementation of baidus Warp-CTC loss from [@jpuigcerver](https://github.com/jpuigcerver/pytorch-baidu-ctc). I found the performance from warp-CTC to be better than the ctc loss native to pytorch. 

## code organization
### Pytorch Datasets:
- fake_texts:
  A folder that includes python files to set up a pytroch dataset using @Belval synthetic data generator. The pytorch dataset is defined in 'pytorch_dataset_fake.py'
  I load the data during the init of the dataset, with a particular set of parameters. The generation has many parameters most of them are hard set inside of the init. 
- IAM_dataset:
  Contains a simple Pytorch dataset to load the IAM handwritten offline dataset, on a line by line basis. 
  
### Notebooks:
- OCR_Training_synthetic.ipynb
  Trains a model on the synthetic dataset
-  OCR_training_handwritten.ipynb
  Trains a model on the IAM offline handwritten line segment dataset. 
- inference_demo.ipynb
  An example of how to go from images to text predictions. 

## Progress: 
Implement performance on benchmark datasets. (IAM etc) 
Implement Batch-Renorm for Pytorch

- [x] Implement Model in Pytorch
- [x] Implement CTC Loss
- [x] Implement Synthetic Data Generator
- [x] Implement synthetic training script
- [x] Implement IAM dataset
- [x] Implement IAM training script
- [ ] Compare performance on benchmark dataset
- [ ] Implement Batch-Renorm for Pytorch
