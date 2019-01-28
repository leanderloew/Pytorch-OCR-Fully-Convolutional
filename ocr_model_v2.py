import torch
import torch.nn as nn
import torchvision.models as models

import numpy as np

from torch.nn.modules.normalization import LayerNorm,LocalResponseNorm
from torch.nn import BatchNorm2d,MaxPool2d

#This tells us which dimension we want to layernomalize over, by default, if we give it one value it will normalize over the
#channels dimension, it has to be a fixed dimension, so the only alternative is if we would use the the height to normalize, we can try tht by exchanging the last two dimnsions like this (0,1,3,2) 
chan_norm=(0,3,2,1)

def lnrom(x,dim=1):
    means=torch.mean(x,dim=1).unsqueeze(1)
    sts=torch.std(x,dim=1).unsqueeze(1)
    out=(x-means)/(sts+0.0001)
    return out 

class depthwise_separable_conv_bn(nn.Module):
    def __init__(self, nin, nout,ks=3,p=1):
        super(depthwise_separable_conv_bn, self).__init__()
        self.depthwise = nn.Conv2d(nin, nin, kernel_size=ks, padding=p, groups=nin,bias=False)
        self.bnorm = BatchNorm2d(nin)
        #self.bnorm=LocalResponseNorm(2)
        self.pointwise = nn.Conv2d(nin, nout, kernel_size=1)
    def forward(self, x):
        out = self.depthwise(x)
        out = self.bnorm(out)
        out = self.pointwise(out)
        return out

    
#We want to be able to change the dimension!
class attention_block(nn.Module):
    def __init__(self, input_channels=8,output_channels=8,input_height=32):
        super(attention_block, self).__init__()

        reduce_dim=int(output_channels/2)
        self.reduce_dim=reduce_dim
        if input_channels==output_channels:
            self.residual=False
        else:
            self.reduce_residual=nn.Conv2d(input_channels, output_channels, kernel_size=1)
            self.residual=True
            
        self.reduce1 = nn.Conv2d(input_channels, reduce_dim, kernel_size=1)
        self.conv1 = depthwise_separable_conv_bn(reduce_dim,reduce_dim)

        self.el=nn.ELU()
        self.reduce2 = nn.Conv2d(reduce_dim, reduce_dim*3, kernel_size=1)
        self.tanh1=nn.Tanh()
        self.tanh2=nn.Tanh()
        self.sig1=nn.Sigmoid()

        self.reduce3 = nn.Conv2d(reduce_dim,output_channels, kernel_size=1)
        self.conv2=depthwise_separable_conv_bn(output_channels,output_channels)
        self.el2=nn.ELU()
        
        w=reduce_dim
        self.ln_1=LayerNorm(w,elementwise_affine=True)#.cuda()
        self.ln_2=LayerNorm(w,elementwise_affine=False)#.cuda()
        self.ln_3=LayerNorm(w,elementwise_affine=False)#.cuda()
        self.ln_4=LayerNorm(w,elementwise_affine=False)#.cuda()
        self.ln_5=LayerNorm(w,elementwise_affine=False)#.cuda()   

    def forward(self, inpu):

        x=self.reduce1(inpu)
        x=self.conv1(x)
        x=self.el(x)
        
        x=self.ln_1(x.permute(chan_norm)).permute(chan_norm)

        x=self.reduce2(x)
        k,q,v=torch.split(x,self.reduce_dim,dim=1)

        k,v,q=self.tanh1(k),self.tanh2(v),self.sig1(q)
        
        k,v,q=self.ln_2(k.permute(chan_norm)).permute(chan_norm),self.ln_3(v.permute(chan_norm)).permute(chan_norm),self.ln_4(q.permute(chan_norm)).permute(chan_norm)
                
        
        atn=(k-v)*q
        
        x=self.ln_5(atn.permute(chan_norm)).permute(chan_norm)
        x=self.reduce3(x)
        x=self.conv2(x)
        x=self.el2(x)
        
        #If we want to use it to change the dimension
        if self.residual ==True:
            inpu=self.reduce_residual(inpu)
        
        
        x=x+inpu
        return x
    
#to do a 1x1 at the beginning also we want to change later too.
  
class cnn_attention_ocr(nn.Module):
    def __init__(self, n_layers, nclasses=5,model_dim=128,input_dim=3):
        super(cnn_attention_ocr, self).__init__()
        
        self.classes=nclasses+1
        self.input_dim=input_dim
        self.n_layers=n_layers
        self.atn_blocks_0=attention_block(19,model_dim)
        #what we can do then is whenever we reduce size we are allowed to increase dimension
        self.atn_blocks_1=attention_block(model_dim,model_dim)
        self.atn_blocks_2=attention_block(model_dim,model_dim)
        self.mp1=MaxPool2d((2,2))
        self.atn_blocks_3=attention_block(model_dim,model_dim*4,input_height=16)
        self.mp2=MaxPool2d((2,1))
        
        #For now we do 8 layers only 
        if n_layers>4: 
            
            self.atn_blocks_4=attention_block(model_dim*4,model_dim*8,input_height=16)
            
            atn_blocks = nn.ModuleList([attention_block(model_dim*8,model_dim*8,input_height=8) for i in range(n_layers-5)])
            self.layers=nn.Sequential(*atn_blocks)

        self.conv1=depthwise_separable_conv_bn(16,16,13,6)

        self.reduce1 = nn.Conv2d(3, 16, kernel_size=1)
        self.reduce2 = nn.Conv2d(model_dim*8, self.classes, kernel_size=1)
                
        self.drop1=nn.Dropout2d(0.5)
        self.drop2=nn.Dropout2d(0.2)
 
        self.ln_3=LayerNorm(self.classes)        
        
        self.ln_1=LayerNorm(3).cuda()
        self.ln_4=LayerNorm(16).cuda()
        
    def forward(self, x):
        
        
        #It makes the most sense to either normalize over the channel or the width dim, cause those are the ones that are fixed
        #But does normalizing voer with really make sense ? I think over the channels makes more sense 
        #x=self.ln_1(x.permute(chan_norm)).permute(chan_norm)
        x_res=x
        x=self.reduce1(x)
        x=torch.softmax(x,1)
        x=self.conv1(x)
        x=self.drop1(x)
        x=self.ln_4(x.permute(chan_norm)).permute(chan_norm)
        x=torch.cat([x,x_res],dim=1)

        x=self.atn_blocks_0(x)
        x=self.atn_blocks_1(x)
        x=self.atn_blocks_2(x)
        x=self.mp1(x)
        x=self.atn_blocks_3(x)
        
        if self.n_layers>4: 
            x=self.atn_blocks_4(x)
            x=self.mp2(x)
            x=self.layers(x)

        x=self.drop2(x)
        x=self.reduce2(x)
        x=torch.mean(x,dim=2)
        #x=LayerNorm((self.classes,input_width)).cuda()(x)
        x=self.ln_3(x.permute((0,2,1))).permute((0,2,1))
        x=nn.LogSoftmax(dim=1)(x)
        
        return x