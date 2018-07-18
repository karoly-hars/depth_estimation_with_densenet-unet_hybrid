import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from collections import OrderedDict
import os


class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size *
                        growth_rate, kernel_size=1, stride=1, bias=False)),
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                        kernel_size=3, stride=1, padding=1, bias=False)),
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate)
            self.add_module('denselayer%d' % (i + 1), layer)


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))



"""
def get_incoming_shape(incoming):
    size = incoming.size()
    # returns the incoming data shape as a list
    return [size[0], size[1], size[2], size[3]]


def interleave(tensors, axis):
    # change the first element (batch_size to -1)
    old_shape = get_incoming_shape(tensors[0])[1:]
    new_shape = [-1] + old_shape
    
    # double 1 dimension
    new_shape[axis] *= len(tensors)
    
    # pack the tensors on top of each other 
    stacked = torch.stack(tensors, axis+1)
    
    # reshape and return
    reshaped = stacked.view(new_shape)
    return reshaped

    
class UnPool_as_Conv(nn.Module):       
    def __init__(self, inplanes, planes):
        super(UnPool_as_Conv, self).__init__()
        
        # interleaving convolutions
        self.conv_A = nn.Conv2d(in_channels = inplanes, out_channels = planes, kernel_size = (3,3), stride = 1, padding = 1)
        self.conv_B = nn.Conv2d(in_channels = inplanes, out_channels = planes, kernel_size = (2,3), stride = 1, padding = 0)
        self.conv_C = nn.Conv2d(in_channels = inplanes, out_channels = planes, kernel_size = (3,2), stride = 1, padding = 0)
        self.conv_D = nn.Conv2d(in_channels = inplanes, out_channels = planes, kernel_size = (2,2), stride = 1, padding = 0)
        
        
    def forward(self,x):
        output_A = self.conv_A(x)
        
        padded_B = nn.functional.pad(x, (1,1,0,1))
        output_B = self.conv_B(padded_B)
        
        padded_C = nn.functional.pad(x, (0,1,1,1))
        output_C = self.conv_C(padded_C)
        
        padded_D = nn.functional.pad(x, (0,1,0,1))
        output_D = self.conv_D(padded_D)  
        
        left = interleave([output_A, output_B], axis = 2)
        right = interleave([output_C, output_D], axis = 2)
        Y = interleave([left, right], axis = 3)
        
        return Y


class UpProj_v0(nn.Module):  
    def __init__(self, inplanes, planes):
        super(UpProj, self).__init__()        
        
        self.unpool_main = UnPool_as_Conv(inplanes, planes)
        self.unpool_res = UnPool_as_Conv(inplanes, planes)
             
        self.main_branch = nn.Sequential(
            self.unpool_main,
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=False),
            nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(planes)
        )
        
        self.residual_branch = nn.Sequential(
            self.unpool_res,
            nn.BatchNorm2d(planes),
        )

        self.relu = nn.ReLU(inplace=False)       
        
        
    def forward(self, input_data):
        
        x = self.main_branch(input_data)
        res = self.residual_branch(input_data)             
        x += res  
        x = self.relu(x)
        
        return x
"""    

class UpProj(nn.Sequential):

    def __init__(self, num_input_features, num_output_features):
        super(UpProj, self).__init__()
        
        self.conv1 = nn.Conv2d(num_input_features, num_output_features, kernel_size=5, stride=1, padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(num_output_features)
        self.relu = nn.ReLU(inplace=True)
        self.conv1_2 = nn.Conv2d(num_output_features, num_output_features, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1_2 = nn.BatchNorm2d(num_output_features)

        self.conv2 = nn.Conv2d(num_input_features, num_output_features, kernel_size=5, stride=1, padding=2, bias=False)
        self.bn2 = nn.BatchNorm2d(num_output_features)

    def forward(self, x, new_size):
        #print('==================\nbup:', x.size())
        x = F.upsample(x, size=new_size, mode='bilinear', align_corners=False)
        #print('aup:', x.size())
        branch1 = self.relu(self.bn1(self.conv1(x)))
        branch1 = self.bn1_2(self.conv1_2(branch1))
        
        branch2 = self.bn2(self.conv2(x))

        out = self.relu(branch1 + branch2)

        return out


class ConConv(nn.Module):
    def __init__(self, inplanes_x1, inplanes_x2, planes):
        super(ConConv, self).__init__()        
        self.conv = nn.Conv2d(inplanes_x1 + inplanes_x2, planes, kernel_size=1, bias=True)
        #self.bn = nn.BatchNorm2d(planes)
    
    def forward(self, x1, x2):
        #print(x1.size(), x2.size())
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        #x = self.bn(x)
        return x



class DenseUnetHyb(nn.Module):
 
    """Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
    """

    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16),
                 num_init_features=64, bn_size=4, drop_rate=0):

        super(DenseUnetHyb, self).__init__()

        # First convolution
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2

        # Final batch norm
        self.features.add_module('norm5', nn.BatchNorm2d(num_features))
        
        
        # Conv2
        self.conv2 = nn.Conv2d(1664, 832, 1, bias=True)
        self.bn2 = nn.BatchNorm2d(832)
        
        self.up1 = UpProj(832, 416)
        self.up2 = UpProj(416, 208)
        self.up3 = UpProj(208, 104)
        self.up4 = UpProj(104, 52)
                
        ''' padding + concat for unet stuff '''
        self.con_conv1 = ConConv(640, 416, 416)
        self.con_conv2 = ConConv(256, 208, 208) 
        self.con_conv3 = ConConv(128, 104, 104) 
        self.con_conv4 = ConConv(64, 52, 52) 
        
        
        # Final layers
        self.drop = nn.Dropout(0.8, False)
        self.conv3 = nn.Conv2d(52, 1, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        
        
        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

        
    def forward(self, x):
        
        x01 = self.features.conv0(x)
        x = self.features.norm0(x01)
        x = self.features.relu0(x)
        #print('after 1st relu:', x.size())
    
        # pool1, block1
        x = self.features.pool0(x)
        #print('after pool0:', x.size())
        x = self.features.denseblock1(x)
        x = self.features.transition1.norm(x)
        x = self.features.transition1.relu(x)
        block1 = self.features.transition1.conv(x)
        #print('b1:', block1.size())
        
        # pool2, block2
        x = self.features.transition1.pool(block1)
        x = self.features.denseblock2(x)
        x = self.features.transition2.norm(x)
        x = self.features.transition2.relu(x)
        block2 = self.features.transition2.conv(x)
        #print('b2:', block2.size())
        
        # poo3, block3
        x = self.features.transition2.pool(block2)
        x = self.features.denseblock3(x)
        x = self.features.transition3.norm(x)
        x = self.features.transition3.relu(x)
        block3 = self.features.transition3.conv(x)
        #print('b3:', block3.size())

        # pool4, block4
        x = self.features.transition3.pool(block3)
        x = self.features.denseblock4(x)
        block4 = self.features.norm5(x)
        #print('b4:', block4.size())
        
        # conv2
        x = self.conv2(block4)
        x = self.bn2(x)
        #print('conv2_out:', x.size())
        
        ''' up project part'''
        x = self.up1(x, [block3.size(2), block3.size(3)])
        #print('up1_out:', x.size())
        x = self.con_conv1(x, block3)
        #print('cconv1_out:', x.size())
        
        x = self.up2(x, [block2.size(2), block2.size(3)])
        #print('up2_out:', x.size())
        x = self.con_conv2(x, block2)
        #print('cconv1_out:', x.size())
        
        x = self.up3(x, [block1.size(2), block1.size(3)])
        #print('up3_out:', x.size())
        x = self.con_conv3(x, block1)
        #print('cconv1_out:', x.size())
        
        x = self.up4(x, [x01.size(2), x01.size(3)])
        #print('up4_out:', x.size())
        x = self.con_conv4(x, x01)
        #print('cconv1_out:', x.size())        
        
        # final layers
        x = self.drop(x)
        x = self.conv3(x)
        x = self.relu(x)
        
        return x




model_urls = {'densenet169': 'https://download.pytorch.org/models/densenet169-b2777c0a.pth'}

def densenet_unet_hyb(load_path='DE_densenet.model', **kwargs):

    model = DenseUnetHyb(num_init_features=64, growth_rate=32,
                         block_config=(6, 12, 32, 32), drop_rate=0, **kwargs)
    
    
    # download the weight in case they are not present
    if not os.path.exists(load_path):
        print('Downloading model weights...')
        os.system("wget https://www.dropbox.com/s/jf4elm14ts1da1n/DE_densenet.model")
        print('Done.')
        
    model.load_state_dict(torch.load(load_path))

    return model


"""
''' TESTING '''
model_conv = densenet_unet_hyb(pretrained=1)
model_conv.eval()


import torchvision
from torchvision import datasets, models, transforms

data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((228,304)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.1, 0.1, 0.1, 0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
    'test': transforms.Compose([
        transforms.Resize((228,304)),
        #transforms.CenterCrop((228,304)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
}
    

from PIL import Image
import numpy as np
rgb = Image.open('tests/21_rgb.png')
rgb = data_transforms['test'](rgb)
rgb = np.expand_dims(rgb, 0)

from torch.autograd import Variable
rgb = Variable(torch.Tensor(rgb))
print('model input size:', rgb.size())
x = model_conv(rgb)
print('model outputs size:', x.size())



#print(x.size())
#print(up8_in.size())

import matplotlib.pyplot as plt
x = x[0,0,:,:].detach().numpy()
plt.imshow(x)
plt.show()
"""