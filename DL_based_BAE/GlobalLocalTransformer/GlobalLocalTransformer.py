"""
This is the code for global-local transformer for brain age estimation

@email: heshengxgd@gmail.com

"""

import torch
import torch.nn as nn

import copy
import math
import vgg as vnet

class GlobalAttention(nn.Module):
    def __init__(self, 
                 transformer_num_heads=8,
                 hidden_size=512,
                 transformer_dropout_rate=0.0):
        super().__init__()
        
        self.num_attention_heads = transformer_num_heads
        # The attention head size is the hidden size divided by the number of attention heads
        self.attention_head_size = int(hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        
        # create the query, key, and value projection layers
        # with n (hidden_size) inputs and m (all_head_size) output
        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)
        
        # create the output layer
        self.out = nn.Linear(hidden_size, hidden_size)
        self.attn_dropout = nn.Dropout(transformer_dropout_rate)
        self.proj_dropout = nn.Dropout(transformer_dropout_rate)
        
        self.softmax = nn.Softmax(dim=-1)
    
    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)
    
    def forward(self,locx,glox):
        locx_query_mix = self.query(locx)
        glox_key_mix = self.key(glox)
        glox_value_mix = self.value(glox)
        
        query_layer = self.transpose_for_scores(locx_query_mix)
        key_layer = self.transpose_for_scores(glox_key_mix)
        value_layer = self.transpose_for_scores(glox_value_mix)
        
        # calculate the attention scores
        # softmax(Q*K.T/sqrt(head_size))*V
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        
        # calculate the attention probabilities using attention scores
        attention_probs = self.softmax(attention_scores)
        attention_probs = self.attn_dropout(attention_probs)

        # contextual layers
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        
        return attention_output

class convBlock(nn.Module):
    def __init__(self,inplace,outplace,kernel_size=3,padding=1):
        super().__init__()
        
        # create a convolutional block with batch normalization and ReLU activation functions
        self.relu = nn.ReLU(inplace=False)
        self.conv1 = nn.Conv2d(inplace,outplace,kernel_size=kernel_size,padding=padding,bias=False)
        self.bn1 = nn.BatchNorm2d(outplace)

    def forward(self,x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        return x
    
class Feedforward(nn.Module):
    def __init__(self,inplace,outplace):
        super().__init__()
        
        self.conv1 = convBlock(inplace,outplace,kernel_size=1,padding=0)
        self.conv2 = convBlock(outplace,outplace,kernel_size=1,padding=0)
    def forward(self,x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class GlobalLocalBrainAge(nn.Module):
    def __init__(self,inplace,
                 patch_size=64,
                 step=-1,
                 nblock=6,
                 drop_rate=0.5,
                 backbone='vgg8'):
        """
        Parameter:
            @patch_size: the patch size of the local pathway
            @step: the step size of the sliding window of the local patches
            @nblock: the number of blocks for the Global-Local Transformer
            @Drop_rate: dropout rate
            @backbone: the backbone of extract the features
        """
        
        super().__init__()
        
        self.patch_size = patch_size
        self.step = step
        self.nblock = nblock

        # Calculate the number of patches from the image size and patch size
        # self.num_patches = (self.image_size // self.patch_size) ** 2
        
        if self.step <= 0:
            self.step = int(patch_size//2)

        # extract the global and local feature maps using the backbone     
        if backbone == 'vgg8':
            self.global_feat = vnet.VGG8(inplace)
            self.local_feat = vnet.VGG8(inplace)
            hidden_size = 512
        elif backbone == 'vgg16':
            self.global_feat = vnet.VGG16(inplace)
            self.local_feat = vnet.VGG16(inplace)
            hidden_size = 512
        else:
            raise ValueError('% model does not supported!'%backbone)

        ########### Transformer encoder block ###############
        # initialize pytorch models (containing sub-modules)
        # because each block has a self attention and feed forward network
        self.attnlist = nn.ModuleList()
        self.fftlist = nn.ModuleList()
        
        # create the multi-head self attention and feed forward network
        # each block has a self attention and feed forward network
        for n in range(nblock):
            # returns the attention output with attention probabilities
            atten = GlobalAttention(
                    transformer_num_heads=8,
                    hidden_size=hidden_size,
                    transformer_dropout_rate=drop_rate)
            self.attnlist.append(atten)
            
            # project the patches/tokens onto lower-dimensional space by convolution
            # create two convolutional blocks 
            fft = Feedforward(inplace=hidden_size*2,
                              outplace=hidden_size)
            self.fftlist.append(fft)

        # average pooling layer    
        self.avg = nn.AdaptiveAvgPool2d(1)
        out_hidden_size = hidden_size
        
        # fully connected local and global output layers
        self.gloout = nn.Linear(out_hidden_size,1)
        self.locout = nn.Linear(out_hidden_size,1)
        
    # forward block here
    def forward(self,xinput):

        # H represents the height and W represents the width of the 2D slice
        _,_,H,W=xinput.size()

        # what is the purpose of outlist?
        outlist = []
        
        # get the global features using the backbone (VGG)
        xglo = self.global_feat(xinput)
        xgfeat = torch.flatten(self.avg(xglo),1)
            
        glo = self.gloout(xgfeat)
        outlist=[glo]
        
        # H1 and W1 represent the dimensions of local features
        # H2 and W2 represent the dimensions of global features
        B2,C2,H2,W2 = xglo.size()
        xglot = xglo.view(B2,C2,H2*W2)
        # change the shape and order of the given tensor without changing its data
        xglot = xglot.permute(0,2,1)
        

        for y in range(0,H-self.patch_size,self.step):
            for x in range(0,W-self.patch_size,self.step):

                # get the patch-wise local features 
                locx = xinput[:,:,y:y+self.patch_size,x:x+self.patch_size]
                xloc = self.local_feat(locx)
                
                # n identical transformer blocks
                for n in range(self.nblock):
                    B1,C1,H1,W1 = xloc.size()
                    xloct = xloc.view(B1,C1,H1*W1)
                    xloct = xloct.permute(0,2,1)
                    
                    tmp = self.attnlist[n](xloct,xglot)
                    tmp = tmp.permute(0,2,1)
                    tmp = tmp.view(B1,C1,H1,W1)
                    tmp = torch.cat([tmp,xloc],1)
                    
                    # add location indexing to the embeddings
                    tmp = self.fftlist[n](tmp)
                    xloc = xloc + tmp
                    
                xloc = torch.flatten(self.avg(xloc),1)
                    
                # get the output of the fully connected local output layer
                # out contains multiple local outputs depending on the height, width, and patch size
                out = self.locout(xloc)
                outlist.append(out)
      
        # returns encoded tensors concatenating global output and local output
        # outlist = [gloout, locout]
        # the number of output tensors [n,1] depend on the height, width, and patch size
        # for an input H=128, W=128, patchsize=128, we get one output tensor
        # n represents the number of samples
        return outlist
    

class ViTForClassfication(nn.Module):
    """
    The ViT model for classification.
    """

    def __init__(self, zlist, hidden_size, num_classes):
        super().__init__()
        self.config = config
        # self.image_size = config["image_size"]
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        # self.zlist = zlist
        # Create the embedding module
        # self.embedding = Embeddings(config)
        # Create the transformer encoder module
        # self.encoder = Encoder(config)
        # Create a linear layer to project the encoder's output to the number of classes
        self.classifier = nn.Linear(self.hidden_size, 1)
        # Initialize the weights
        # self.apply(self._init_weights)

    def forward(self, output_attentions=False):
        # Calculate the embedding output
        # embedding_output = self.embedding(x)
        encoder_output = self.zlist
        # Calculate the encoder's output
        # encoder_output, all_attentions = self.encoder(embedding_output, output_attentions=output_attentions)
        # Calculate the logits, take the [CLS] token's output as features for classification
        logits = self.classifier(encoder_output[:, 0, :])
        # Return the logits and the attention probabilities (optional)
        if not output_attentions:
            return (logits, None)
        else:
            return (logits, all_attentions)
    
    # def _init_weights(self, module):
    #     if isinstance(module, (nn.Linear, nn.Conv2d)):
    #         torch.nn.init.normal_(module.weight, mean=0.0, std=self.config["initializer_range"])
    #         if module.bias is not None:
    #             torch.nn.init.zeros_(module.bias)
    #     elif isinstance(module, nn.LayerNorm):
    #         module.bias.data.zero_()
    #         module.weight.data.fill_(1.0)
    #     elif isinstance(module, Embeddings):
    #         module.position_embeddings.data = nn.init.trunc_normal_(
    #             module.position_embeddings.data.to(torch.float32),
    #             mean=0.0,
    #             std=self.config["initializer_range"],
    #         ).to(module.position_embeddings.dtype)

    #         module.cls_token.data = nn.init.trunc_normal_(
    #             module.cls_token.data.to(torch.float32),
    #             mean=0.0,
    #             std=self.config["initializer_range"],
    #         ).to(module.cls_token.dtype)


if __name__ == '__main__':
    x1 = torch.rand(1,5,130,170)
    
    mod = GlobalLocalBrainAge(5,
                        patch_size=64,
                        step=32,
                        nblock=6,
                        backbone='vgg8')
    zlist = mod(x1)
    for z in zlist:
        print(z.shape)
    print('number is:',len(zlist))
   
        
