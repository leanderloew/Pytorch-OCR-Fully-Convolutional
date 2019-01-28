# Pytorch-OCR-Fully-Conv

This repository contains code of a pytorh implementation of: 
Accurate, Data-Efficient, Unconstrained Text Recognition with Convolutional Neural Networks https://arxiv.org/abs/1812.11894
I am not affiliated with the authors of the paper. 

I also implemented the excellent data generator of: https://github.com/Belval/TextRecognitionDataGenerator as a pytorch dataset. 

Notices: 
Somethimes the training doesn't quite start. The problem is that the ctc loss gets stuck in a local minimum, where the model only predicts the "blank" or any repeated symbol, that then gets removed. 

Initial parameters, that usually work fine: 
1.) Start without image augmentations
2.) Use a Learning rate of 5e-5 
3.) use a network with 8 layers and 64 units also 0.5 and 0.2  droppout 

Todo: 
Implement performance on benchmark datasets. (IAM etc) 

