# Pytorch-OCR-Fully-Conv

This repository contains code of a pytorch implementation of: 
[Accurate, Data-Efficient, Unconstrained Text Recognition with Convolutional Neural Networks](https://arxiv.org/abs/1812.11894) I am not affiliated with the authors of the paper. 

I also implemented the excellent data generator of [@Belval](https://github.com/Belval/TextRecognitionDataGenerator) as a pytorch dataset. 

## code organization
- fake_texts
  A folder that includes python files to set up a pytroch dataset using @Belval synthetic data generator. 

  I load the data during the init, with a particular set of parameters. The generation has many parameters most of them are hard_set inside of the init. 


## Notices
Sometimes the training doesn't quite start. The problem is that the ctc loss gets stuck in a local minimum, where the model only predicts the "blank" or any repeated symbol. This would then result in an infinite CER. 

I also tried out the CTC Loss from Pytorch but I did not get it to train properly. 

Initial parameters, that usually work fine: 
- Start without image augmentations
- Use a Learning rate of 5e-5 
- use a network with 8 layers and 64 units also 0.5 and 0.2  droppout 



Todo: 
Implement performance on benchmark datasets. (IAM etc) 
Implement Batch-Renorm for Pytorch

