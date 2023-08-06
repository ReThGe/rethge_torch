## this file contains self-coded components classes, for build deepleanring model in pytorch

# author: rethge
# created data: 2023/07/20

import torch
from torch import nn

import torch.nn.functional as F


# Depthwise separeble conv———————————————————————————————————————————————

class RTG_depthwise_separable_conv(nn.Module):
    def __init__(self, input_size, output_size, kernel_size,
                 stride, padding, bias: bool = True):
        super().__init__()

        self.dsc = nn.Sequential(
            nn.Conv2d(in_channels=input_size, 
                                    out_channels=input_size,
                                    kernel_size=kernel_size,
                                    stride=stride,
                                    padding=padding,
                                    groups=input_size,
                                    bias=bias),

            nn.Conv2d(in_channels=input_size,
                                        out_channels=output_size,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0,
                                        bias=bias)
        )

        
    def forward(self, x):
        return self.dsc(x)


class RTG_res_block(nn.Module):
    def __init__(self, nin):
        super().__init__()

        self.inp = nin
        self.conv1 = RTG_depthwise_separable_conv(nin, nin, kernel_size=3,
                                                  stride=1,padding=1)
        self.conv2 = RTG_depthwise_separable_conv(nin, nin, kernel_size=3,
                                                  stride=1,padding=1)
        
    def forward(self, x):
        return F.selu(self.conv2(F.selu(self.conv1(x))) + x)


class RTG_res_block_expand(nn.Module):
    def __init__(self, nin, nout):
        super().__init__()

        self.inp = nin
        self.oup = nout
        self.conv1 = RTG_depthwise_separable_conv(nin, nout, kernel_size=3,
                                                  stride=1, padding=1)
        self.conv2 = RTG_depthwise_separable_conv(nout, nout, kernel_size=3,
                                                  stride=1, padding=1)
        
        self.identity = nn.Conv2d(nin, nout, kernel_size=1,
                                  stride=1, padding=0)
        
    def forward(self, x):
        a = self.conv2(F.selu(self.conv1(x)))
        identity = self.identity(x)

        return F.selu(a + identity)

# ViT————————————————————————————————————————————————————————————————————

# a torch class for patch layer -- vision transformer
class RTG_PatchEmbedding(nn.Module):
    """Turns a 2D input image into a 1D sequence learnable embedding vector.
    
    Args:
        in_channels (int): Number of color channels for the input images. Defaults to 3.
        patch_size (int): Size of patches to convert input image into. Defaults to 16.
        embedding_dim (int): Size of embedding to turn image into. Defaults to 768.
    """ 
    def __init__(self,
                 input_channels: int = 3,
                 patch_size: int = 16,
                 embedding_size: int = 768):
        super().__init__()

        self.patch_size = patch_size

        self.patcher = nn.Conv2d(in_channels=input_channels,
                       out_channels=embedding_size,
                       stride=patch_size,
                       kernel_size=patch_size,
                       padding=0)
        
        self.flatten = nn.Flatten(start_dim=2, end_dim=3)

    def forward(self, x):
        img_size = x.shape[-1]
        assert img_size % self.patch_size == 0, f"Input image size must be divisble by patch size, image shape: {img_size}, patch size: {self.patch_size}"

        return self.flatten(self.patcher(x)).permute(0,2,1)


class RTG_MultiheadSelf_attention_block(nn.Module):
    def __init__(self,
                 embedding_dim: int = 768,
                 num_head: int = 12,
                 attention_dropout: float = 0):
        super().__init__()

        self.LayerNorm = nn.LayerNorm(normalized_shape=embedding_dim)

        self.multihead_attention = nn.MultiheadAttention(embed_dim=embedding_dim,
                                                         num_heads=num_head,
                                                         dropout=attention_dropout,
                                                         batch_first=True)
        
    def forward(self, x):
        x = self.LayerNorm(x)  # x with shape [1, 197, 768]
        attention_out, _ = self.multihead_attention(query=x,
                                                    key=x,
                                                    value=x,
                                                    need_weights=False)
        
        return attention_out


class RTG_MLPBlock(nn.Module):
    def __init__(self,
                 embedding_dim: int = 768,
                 mlp_size: int = 3072,
                 dropout: float = 0.1):
        super().__init__()

        self.LayerNorm = nn.LayerNorm(normalized_shape=embedding_dim)

        self.mlp = nn.Sequential(
            nn.Linear(in_features=embedding_dim,
                      out_features=mlp_size),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=mlp_size,
                      out_features=embedding_dim),
            nn.Dropout(p=dropout)
        )


    def forward(self, x):
        return self.mlp(self.LayerNorm(x))


class RTG_TransformerEncoderBlock(nn.Module):
    def __init__(self,
                 embedding_dim: int = 768,
                 num_head: int = 12,
                 mlp_size: int = 3072,
                 mlp_dropout: float = 0.1,
                 attention_dropout: float = 0):
        super().__init__()

        self.msa_block = RTG_MultiheadSelf_attention_block(embedding_dim=embedding_dim,
                                                           num_head=num_head,
                                                           attention_dropout=attention_dropout)
        
        self.mlp_block = RTG_MLPBlock(embedding_dim=embedding_dim,
                                      mlp_size=mlp_size,
                                      dropout=mlp_dropout)
        

    def forward(self, x):
        x = self.msa_block(x) + x
        x = self.mlp_block(x) + x

        return x


class RTG_ViT(nn.Module):
    def __init__(self,
                 img_size: int = 224,
                 input_channels: int = 3,
                 patch_size: int = 16,
                 num_transformer_layers: int = 12,
                 # embedding_dim: int = 768,
                 num_head: int = 12,
                 mlp_size: int = 3072,
                 embedding_dropout: float = 0.1, # Dropout for patch and position embeddings
                 mlp_dropout: float = 0.1, # Dropout for dense/MLP layers 
                 attention_dropout: float = 0, # Dropout for attention projection
                 num_classes: int = 1000): # Default for ImageNet but can customize this
        super().__init__()

        assert img_size % patch_size == 0, f"Image size must be divisible by patch size, image size: {img_size}, patch size: {patch_size}."

        # self.patch_size = patch_size
        self.num_patches = (img_size//patch_size)**2
        self.embedding_dim = input_channels*patch_size**2

        self.class_embedding = nn.Parameter(data=torch.randn(1, 1, self.embedding_dim),
                                            requires_grad=True) # 1x1x768
        
        self.position_embedding = nn.Parameter(data=torch.randn(1, self.num_patches+1, self.embedding_dim),
                                            requires_grad=True) # 1x197x768
        
        self.embedding_dropout = nn.Dropout(p=embedding_dropout)

        self.img_patch_embedding = RTG_PatchEmbedding(input_channels=input_channels,
                                                      patch_size=patch_size,
                                                      embedding_size=self.embedding_dim)
        
        # Note: The "*" means "all"
        self.transformer_encoder = nn.Sequential( # stack 12 times of encoder block
            *[RTG_TransformerEncoderBlock(embedding_dim=self.embedding_dim,
                                          num_head=num_head,
                                          mlp_size=mlp_size,
                                          mlp_dropout=mlp_dropout) for _ in range(num_transformer_layers)])
        

        self.classifier = nn.Sequential(
            nn.LayerNorm(normalized_shape=self.embedding_dim),
            nn.Linear(in_features=self.embedding_dim,
                      out_features=num_classes)
        )


    def forward(self, x):

        # Get batch size
        # batch_size = x.shape[0]
        
        # Create class token embedding and expand it to match the batch size (equation 1)
        class_token = self.class_embedding.expand(x.shape[0], -1, -1) # "-1" means to infer the dimension (try this line on its own)

        # Create patch embedding (equation 1)
        # x = self.img_patch_embedding(x)

        # Concat class embedding and patch embedding (equation 1)
        x = torch.cat((class_token, self.img_patch_embedding(x)), dim=1) + self.position_embedding

        # Add position embedding to patch embedding (equation 1) 
        # x = self.position_embedding + x

        # Run embedding dropout (Appendix B.1)
        # x = self.embedding_dropout(x)

        # Pass patch, position and class embedding through transformer encoder layers (equations 2 & 3)
        x = self.transformer_encoder(self.embedding_dropout(x))

        # Put 0 index logit through classifier (equation 4)
        x = self.classifier(x[:, 0]) # run on each sample in a batch at 0 index -> class embeddings at the very top of tensor

        return x



# Resnet——————————————————————————————————————————————————————————————————————————————————————————————————————
class Block(nn.Module):
    def __init__(self,
                 in_c,
                 out_c,
                 identity_downsample=None, # conv
                 stride=1):
        super().__init__()

        self.expansion = 4

        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(out_c)

        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(out_c)

        self.conv3 = nn.Conv2d(out_c, out_c*self.expansion, kernel_size=1, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(out_c*self.expansion)

        self.relu = nn.ReLU()
        self.identity_downsample = identity_downsample

    def forward(self, x):
        identity = x
        
        x = self.relu(self.bn1(self.conv1(x)))

        x = self.relu(self.bn2(self.conv2(x)))

        x = self.bn3(self.conv3(x))

        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)

        x += identity
        x = self.relu(x)

        return x
    

class RTG_Resnet(nn.Module):
    def __init__(self, block, layers, img_channels, num_classes):
        super().__init__()

        self.in_c = 64
        self.conv1 = nn.Conv2d(img_channels, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # res layers
        self.layer1 = self._make_layer(block, layers[0],
                                       out_channels=64, 
                                       stride=1)
        
        self.layer2 = self._make_layer(block, layers[1],
                                       out_channels=128, 
                                       stride=2)
        
        self.layer3 = self._make_layer(block, layers[2],
                                       out_channels=256, 
                                       stride=2)
        
        self.layer4 = self._make_layer(block, layers[3],
                                       out_channels=512, 
                                       stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1,1)) # GAP
        self.fc = nn.Linear(512*4, num_classes)


    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))

        x = self.avgpool(self.layer4(self.layer3(self.layer2(self.layer1(x)))))
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)

        return x

    def _make_layer(self, block, num_res_blocks, out_channels, stride):
        
        identity_dowansample = None
        layers = []

        if stride != 1 or self.in_c != out_channels*4:
            identity_dowansample = nn.Sequential(
                nn.Conv2d(self.in_c, out_channels*4, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels*4)
            )

        layers.append(block(self.in_c, out_channels, identity_dowansample, stride))

        self.in_c = out_channels*4

        for _ in range(num_res_blocks - 1):
            layers.append(block(self.in_c, out_channels))

        return nn.Sequential(*layers)
    
def RTG_Resnet50(img_c=3, num_class=3):
    return RTG_Resnet(Block, [3,4,6,3], img_c, num_class)