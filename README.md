# Pytorch-OCR-Fully-Conv

This repository contains code of a pytorch implementation of: 
[Accurate, Data-Efficient, Unconstrained Text Recognition with Convolutional Neural Networks](https://arxiv.org/abs/1812.11894) I am not affiliated with the authors of the paper. 

I also implemented the excellent data generator of [@Belval](https://github.com/Belval/TextRecognitionDataGenerator) as a pytorch dataset. I reyl on the pytorch implementation of the ctc loss from [@jpuigcerver](https://github.com/jpuigcerver/pytorch-baidu-ctc). I found the performance from the warp-CTC to be better than the ones native to pytorch. 

## code organization
### Pytorch Datasets:
- fake_texts:
  A folder that includes python files to set up a pytroch dataset using @Belval synthetic data generator. The pytorch dataset is defined in 'pytorch_dataset_fake.py'
  I load the data during the init of the dataset, with a particular set of parameters. The generation has many parameters most of them are hard set inside of the init. 
- IAM_dataset:
  Contains a simple Pytorch dataset to load the IAM handwritten offline dataset, on a line by line basis. 
### Training Notebooks:
- `OCR_Training_synthetic.ipynb'
  Trains a model on the synthetic dataset
-  `OCR_training_handwritten.ipynb'
  Trains a model on the IAM offline handwritten line segment dataset. 


## Notices
In my experience this model can be hard to train. The problem is that the ctc loss gets stuck in a local minimum, where the model only predicts the "blank" or any repeated symbol. You can see that in tensorboardx by watching the CER metric. 

Initial parameters, that usually work fine: 
- Start without image augmentations and easy examples.
- Use a Learning rate of 5e-5 

## Todo: 
Implement performance on benchmark datasets. (IAM etc) 
Implement Batch-Renorm for Pytorch

