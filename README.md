# Pytorch-OCR-Fully-Conv

This repository contains code of a pytorh implementation of: 
Accurate, Data-Efficient, Unconstrained Text Recognition with Convolutional Neural Networks https://arxiv.org/abs/1812.11894
I am not affiliated with the authors of the paper. 

I also implemented the excellent data generator of: https://github.com/Belval/TextRecognitionDataGenerator as a pytorch dataset. 

Notices: 
Somethimes the training doesn't quite start. The problem is that the ctc loss gets stuck in a local minimum, where the model only predicts the "blank" or any repeated symbol, that then gets removed. 

Todo: 
Implement performance on benchmark datasets. 
