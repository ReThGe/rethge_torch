## this file contains some different Model Class build with Pytorch
# Author: rethge
# 2023/07/02


## imports

import torch
from torch import nn
from rethge_components import RTG_depthwise_separable_conv, RTG_res_block, RTG_res_block_expand, RTG_inverted_res, RTG_channel_spatial_attention


class LinetV0(nn.Module): # baseline
    def __init__(self,
                 res: int = 640, # res should match with dataloader transformation size
                 output_shape: int = 3,  # len(class_name)
                 input_shape: int = 3,  # 640*640*3 = 1228800
                 hidden_units: int = 8):
        super().__init__()
        '''
        |           O           O
        | -> BN ->  O -> BN ->  O -> result
        |           O           O
        
        **(in)      8(hidden)   3(out)

        Forward/backward pass size (MB): 157.29
        '''

        self.Linear_stack = nn.Sequential(
            nn.Flatten(),
            # nn.BatchNorm1d(num_features=input_shape*res**2), # BN to avoid convergence failure
            nn.Linear(in_features=input_shape*res**2, out_features=hidden_units),
            nn.ReLU(),
            # nn.BatchNorm1d(hidden_units),
            nn.Linear(in_features=hidden_units, out_features=output_shape),
            # nn.ReLU()
        )
    
    def forward(self, x: torch.Tensor):
        return self.Linear_stack(x)
    

class LinetV0_5(nn.Module):
    """Forward/backward pass size (MB): 0.00"""
    def __init__(self, res):
        super().__init__()

        self.downsamp = res//2

        self.stack = nn.Sequential(
            nn.AdaptiveAvgPool2d((self.downsamp, self.downsamp)),
            nn.Flatten(),
            nn.BatchNorm1d(num_features=3*self.downsamp**2),  # using BN leading to an increase on Forward/backward pass size 39.32 MB
            # without BN, fail to converge
            nn.Linear(in_features=3*self.downsamp**2, out_features=8),
            nn.ReLU(),
            nn.BatchNorm1d(8),
            nn.Linear(in_features=8, out_features=16),
            nn.ReLU(),
            nn.BatchNorm1d(16),
            nn.Linear(in_features=16, out_features=3)
        )

    def forward(self, x):
        return self.stack(x)


class LinetV1(nn.Module): # convnet, tiny vgg-like
    def __init__(self,
                 output_shape: int = 3, # len(class_name)
                 init_channels: int = 32,
                 input_shape: int = 3,
                 clf_shape: int = 3) -> None:
        super().__init__()
        """

        Forward/backward pass size (MB): 5452.60 on [8,3,480,480]

        """
        
        self.mul_1 = 64 # 4-64
        self.mul_2 = 128 # 4-128
        self.mul_3 = 256 # 4-256


        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape, 
                      out_channels=init_channels,
                      kernel_size=3,
                      stride=1,
                      padding='same', bias=False),
            nn.BatchNorm2d(num_features=init_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=init_channels, 
                      out_channels=self.mul_1,
                      kernel_size=3,
                      stride=1,
                      padding='same', bias=False),
            nn.BatchNorm2d(num_features=self.mul_1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        

        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(in_channels=self.mul_1, 
                      out_channels=self.mul_2,
                      kernel_size=3,
                      stride=1,
                      padding='same', bias=False),
            nn.BatchNorm2d(num_features=self.mul_2),
            nn.ReLU(),
            nn.Conv2d(in_channels=self.mul_2, 
                      out_channels=self.mul_2,
                      kernel_size=3,
                      stride=1,
                      padding='same', bias=False),
            nn.BatchNorm2d(num_features=self.mul_2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.conv_block_3 = nn.Sequential(
            nn.Conv2d(in_channels=self.mul_2, 
                      out_channels=self.mul_3,
                      kernel_size=3,
                      stride=1,
                      padding='same', bias=False),
            nn.BatchNorm2d(num_features=self.mul_3),
            nn.ReLU(),
            nn.Conv2d(in_channels=self.mul_3, 
                      out_channels=self.mul_3,
                      kernel_size=3,
                      stride=1,
                      padding='same', bias=True),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3)
        )


        self.classifier = nn.Sequential(
            nn.BatchNorm2d(self.mul_3),
            nn.AdaptiveAvgPool2d((clf_shape,clf_shape)),
            nn.Flatten(),
            nn.Linear(in_features=self.mul_3*clf_shape*clf_shape,
                      out_features=output_shape)
        )
    
    def forward(self, x: torch.Tensor):
        # x = self.conv_block_1(x) # if input res is 480:
        # print(x.shape) # torch.Size([b, 64, 240, 240])
        # x = self.conv_block_2(x)
        # print(x.shape) # torch.Size([b, 128, 120, 120])
        # x = self.conv_block_3(x)
        # print(x.shape) # torch.Size([b, 256, 40, 40])
        # x = self.classifier(x)
        # print(x.shape) # torch.Size([b, 3])
        # return x
        return self.classifier(self.conv_block_3(self.conv_block_2(self.conv_block_1(x)))) # benefit from operate fusion
    

class LinetV1_DSC(nn.Module):
    def __init__(self,
                 output_shape: int = 3, # len(class_name)
                 init_channels: int = 32,
                 input_shape: int = 3,
                 clf_shape: int = 3) -> None:
        super().__init__()
        """

        Forward/backward pass size (MB): 5452.60 on [8,3,480,480]

        """
        
        self.mul_1 = 64 # 4-64
        self.mul_2 = 128 # 4-128
        self.mul_3 = 256 # 4-256


        self.conv_block_1 = nn.Sequential(
            RTG_depthwise_separable_conv(input_size=input_shape, 
                      output_size=init_channels,
                      kernel_size=3,
                      stride=1,
                      padding='same', bias=False),
            nn.BatchNorm2d(num_features=init_channels),
            nn.ReLU(),
            RTG_depthwise_separable_conv(input_size=init_channels, 
                      output_size=self.mul_1,
                      kernel_size=3,
                      stride=1,
                      padding='same', bias=False),
            nn.BatchNorm2d(num_features=self.mul_1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        

        self.conv_block_2 = nn.Sequential(
            RTG_depthwise_separable_conv(input_size=self.mul_1, 
                      output_size=self.mul_2,
                      kernel_size=3,
                      stride=1,
                      padding='same', bias=False),
            nn.BatchNorm2d(num_features=self.mul_2),
            nn.ReLU(),
            RTG_depthwise_separable_conv(input_size=self.mul_2, 
                      output_size=self.mul_2,
                      kernel_size=3,
                      stride=1,
                      padding='same', bias=False),
            nn.BatchNorm2d(num_features=self.mul_2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.conv_block_3 = nn.Sequential(
            RTG_depthwise_separable_conv(input_size=self.mul_2, 
                      output_size=self.mul_3,
                      kernel_size=3,
                      stride=1,
                      padding='same', bias=False),
            nn.BatchNorm2d(num_features=self.mul_3),
            nn.ReLU(),
            RTG_depthwise_separable_conv(input_size=self.mul_3, 
                      output_size=self.mul_3,
                      kernel_size=3,
                      stride=1,
                      padding='same', bias=True),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3)
        )


        self.classifier = nn.Sequential(
            nn.BatchNorm2d(self.mul_3),
            nn.AdaptiveAvgPool2d((clf_shape,clf_shape)),
            nn.Flatten(),
            nn.Linear(in_features=self.mul_3*clf_shape*clf_shape,
                      out_features=output_shape)
        )
    
    def forward(self, x: torch.Tensor):
        return self.classifier(self.conv_block_3(self.conv_block_2(self.conv_block_1(x))))


class LinetV1_5(nn.Module):
    def __init__(self, nin, nout, expand=32):
        super().__init__()

        """without res connect, we can not going deep"""

        self.stack = nn.Sequential(
            RTG_depthwise_separable_conv(input_size=nin, output_size=expand, kernel_size=3, stride=1, padding="same", bias=False),
            nn.BatchNorm2d(expand),
            RTG_depthwise_separable_conv(input_size=expand, output_size=expand, kernel_size=3, stride=1, padding="same", bias=True),
            nn.SELU(),
            nn.MaxPool2d(kernel_size=2),

            RTG_depthwise_separable_conv(input_size=expand, output_size=expand*4, kernel_size=3, stride=1, padding=0, bias=True),
            # nn.BatchNorm2d(expand*4),
            nn.SELU(),
            RTG_depthwise_separable_conv(input_size=expand*4, output_size=expand*4, kernel_size=3, stride=1, padding=0, bias=True),
            nn.SELU(),
            nn.MaxPool2d(kernel_size=2),

            RTG_depthwise_separable_conv(input_size=expand*4, output_size=expand*8, kernel_size=3, stride=1, padding="same", bias=False),
            nn.BatchNorm2d(expand*8),
            RTG_depthwise_separable_conv(input_size=expand*8, output_size=expand*8, kernel_size=3, stride=1, padding="same", bias=True),
            nn.SELU(),
            nn.MaxPool2d(kernel_size=2),

        )

        self.clf = nn.Sequential(
            nn.AdaptiveAvgPool2d((3,3)),
            nn.Flatten(),
            nn.Linear(in_features=3*3*expand*8, out_features=nout)
        )

    def forward(self, x):
        return self.clf(self.stack(x))


class LinetV2(nn.Module):
    def __init__(self, nin, nout, expand=32):
        super().__init__()

        self.head = nn.Sequential(
            RTG_depthwise_separable_conv(input_size=nin, output_size=expand,
                                         kernel_size=5, stride=1, padding="same", bias=False),
            nn.BatchNorm2d(expand),
            RTG_depthwise_separable_conv(input_size=expand, output_size=expand*2, 
                                         kernel_size=3, stride=1, padding=1, bias=True),
            nn.SELU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.res_body = nn.Sequential(
            RTG_res_block(nin=expand*2), # output is selu(...)
            RTG_res_block(nin=expand*2),
            RTG_res_block_expand(nin=expand*2, nout=expand*4), # 128
            nn.BatchNorm2d(expand*4),
            nn.MaxPool2d(kernel_size=2),
            RTG_res_block(nin=expand*4),
            RTG_res_block(nin=expand*4),
            RTG_res_block_expand(nin=expand*4, nout=expand*8), # 256
            nn.MaxPool2d(kernel_size=2)
            )
        
        self.tail = nn.Sequential(
            RTG_depthwise_separable_conv(input_size=expand*8, output_size=expand*16, # 512
                                         kernel_size=3, stride=1, padding=1, bias=True),
            # nn.BatchNorm2d(expand*16),
            nn.SELU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.clf = nn.Sequential(
            nn.AdaptiveAvgPool2d((2,2)),
            nn.Flatten(),
            nn.Linear(in_features= 2*2*expand*16, out_features=256), # 2*2*512 = 2048
            nn.SELU(),
            nn.Dropout(0.2),
            nn.Linear(in_features=256, out_features=nout),
        )

    def forward(self, x):
        return self.clf(self.tail(self.res_body(self.head(x))))


class LinetV2_no_pool_WIP(nn.Module): # without pooling layer, it will fail to learn
    def __init__(self, nin, nout, expand=32):
        super().__init__()

        self.head = nn.Sequential(
            RTG_depthwise_separable_conv(input_size=nin, output_size=expand,
                                         kernel_size=5, stride=1, padding="same", bias=False),
            nn.BatchNorm2d(expand),
            RTG_depthwise_separable_conv(input_size=expand, output_size=expand*2, 
                                         kernel_size=3, stride=1, padding=1, bias=True),
            nn.SELU(),
            RTG_depthwise_separable_conv(input_size=expand*2, output_size=expand*2,
                                         kernel_size=3, stride=2, padding=1) # 224 -> 112
        )

        self.res_body = nn.Sequential(
            RTG_res_block(nin=expand*2), # output is selu(...)
            RTG_res_block(nin=expand*2),
            RTG_res_block_expand(nin=expand*2, nout=expand*4),
            RTG_depthwise_separable_conv(input_size=expand*4, output_size=expand*4,
                                         kernel_size=3, stride=2, padding=1), # 112 -> 56
            nn.BatchNorm2d(expand*4),
            RTG_res_block(nin=expand*4),
            RTG_res_block(nin=expand*4),
            RTG_res_block_expand(nin=expand*4, nout=expand*8), # 256
            RTG_depthwise_separable_conv(input_size=expand*8, output_size=expand*8,
                                         kernel_size=3, stride=2, padding=1), # 56 -> 28
            nn.BatchNorm2d(expand*8),
            )
        
        self.tail = nn.Sequential(
            RTG_depthwise_separable_conv(input_size=expand*8, output_size=expand*16, # 512
                                         kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(expand*16),
            RTG_depthwise_separable_conv(input_size=expand*16, output_size=expand*16,
                                         kernel_size=3, stride=2, padding=1), # 28 -> 14
            nn.BatchNorm2d(expand*16),
        )

        self.clf = nn.Sequential(
            nn.AdaptiveAvgPool2d((2,2)),
            nn.Flatten(),
            nn.Linear(in_features=2*2*expand*16, out_features=128), # 2*2*512 = 2048
            nn.SELU(),
            nn.Dropout(0.2),
            nn.Linear(in_features=128, out_features=nout),
        )

    def forward(self, x):
        return self.clf(self.tail(self.res_body(self.head(x))))


class LinetV2_fast(nn.Module): # 3, batch, 224, 224
    def __init__(self, nin, nout, expand=16, exp_size=64):
        super().__init__()
        self.nin = nin
        self.nout = nout
        self.expand = expand
        self.exp_size = exp_size

        self.head = nn.Sequential(
            RTG_depthwise_separable_conv(input_size=self.nin, output_size=self.expand, kernel_size=3, stride=1, padding="same"), # if you BN, you can set bias to false
            nn.BatchNorm2d(self.expand), 
            nn.ReLU6(),
            RTG_depthwise_separable_conv(input_size=self.expand, output_size=self.expand*2, kernel_size=3, stride=1, padding="same", bias=False), # 32 channel
            nn.MaxPool2d(kernel_size=2) # 112
        )

        self.res_body = nn.Sequential(
            RTG_inverted_res(nin=self.expand*2, exp_size=round(self.exp_size/2), kernel_size=3),
            nn.ReLU6(),
            RTG_inverted_res(nin=self.expand*2, exp_size=round(self.exp_size/2), kernel_size=3),
            RTG_inverted_res(nin=self.expand*2, exp_size=round(self.exp_size/2), kernel_size=3),
            nn.ReLU6(),
            RTG_inverted_res(nin=self.expand*2, exp_size=round(self.exp_size/2), kernel_size=3),
            nn.BatchNorm2d(self.expand*2),
            RTG_res_block_expand(nin=self.expand*2, nout=self.expand*4), # 64
            nn.MaxPool2d(kernel_size=2), # 56

            RTG_inverted_res(nin=self.expand*4, exp_size=self.exp_size, kernel_size=3),
            nn.SELU(),
            RTG_inverted_res(nin=self.expand*4, exp_size=self.exp_size, kernel_size=3),
            nn.ReLU6(),
            RTG_inverted_res(nin=self.expand*4, exp_size=self.exp_size, kernel_size=3),
            nn.ReLU6(),
            RTG_inverted_res(nin=self.expand*4, exp_size=self.exp_size, kernel_size=3),
            nn.BatchNorm2d(self.expand*4),
            RTG_inverted_res(nin=self.expand*4, exp_size=self.exp_size*2, kernel_size=3),
            nn.ReLU6(),
            RTG_inverted_res(nin=self.expand*4, exp_size=self.exp_size*2, kernel_size=3),
            nn.BatchNorm2d(self.expand*4),
            # RTG_res_block_expand(nin=self.expand*4, nout=self.expand*8), # 128
            nn.MaxPool2d(kernel_size=2), # 28
        )

        self.clf = nn.Sequential(
            nn.AdaptiveAvgPool2d((3, 3)), # 64, batch, 28, 28  # maybe this affects the performance
            nn.Flatten(), # 3x3x128
            nn.Linear(in_features=3*3*self.expand*4, out_features=3), 
        )

    def forward(self, x):
        return self.clf(self.res_body(self.head(x)))
    


class LinetV2_faster(nn.Module): # 3, batch, 224, 224
    def __init__(self, nin=3, nout=3):
        super().__init__()

        self.start_block = nn.Sequential(
            RTG_depthwise_separable_conv(input_size=nin, output_size=16, kernel_size=5, stride=1, padding="same"), # 224 -> 112
            nn.BatchNorm2d(16),
            nn.SiLU(),
            nn.AvgPool2d(2),
        )

        self.stage_1 = nn.Sequential(
            RTG_inverted_res(nin=16, exp_size=64, kernel_size=5),
            RTG_inverted_res(nin=16, exp_size=64, kernel_size=5),
            RTG_channel_spatial_attention(nin=16, reduce_to=8),
            nn.BatchNorm2d(16),
            RTG_inverted_res(nin=16, exp_size=64, kernel_size=3),
            RTG_inverted_res(nin=16, exp_size=64, kernel_size=3),
            RTG_channel_spatial_attention(nin=16, reduce_to=8),
            nn.BatchNorm2d(16),
            RTG_res_block_expand(nin=16, nout=32),
            nn.MaxPool2d(2), # 112-> 56
        )

        self.stage_2 = nn.Sequential(
            RTG_inverted_res(nin=32, exp_size=64, kernel_size=3),
            RTG_inverted_res(nin=32, exp_size=64, kernel_size=3),
            RTG_channel_spatial_attention(nin=32, reduce_to=8),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2), # 56 -> 28

            RTG_res_block_expand(nin=32, nout=64),
            RTG_inverted_res(nin=64, exp_size=128, kernel_size=3),
            RTG_inverted_res(nin=64, exp_size=128, kernel_size=3),
            RTG_channel_spatial_attention(nin=64, reduce_to=8),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2), # 28 -> 14

            RTG_res_block_expand(nin=64, nout=128),
            RTG_res_block(nin=128),
            RTG_channel_spatial_attention(nin=128, reduce_to=32),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2), # 14 -> 7
        )

        self.clf = nn.Sequential(
            nn.AdaptiveAvgPool2d((3,3)), # 8, 128, 3, 3
            nn.Flatten(),
            nn.Linear(in_features=128*3*3, out_features=nout),
        )

    def forward(self, x):
        return self.clf(self.stage_2(self.stage_1(self.start_block(x))))