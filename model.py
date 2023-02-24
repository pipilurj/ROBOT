import torch
import torch.nn as nn
import torch.nn.functional as functional
import torch.nn.init as init
import torch.nn.functional as F
import numpy as np

def _weights_init(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            functional.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = functional.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = functional.relu(out)
        return out


class ResNet32(nn.Module):
    def __init__(self, num_classes=10, block=BasicBlock, num_blocks=[5, 5, 5]):
        super(ResNet32, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = nn.Linear(64, num_classes)

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = functional.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = functional.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


class HiddenLayer(nn.Module):
    def __init__(self, input_size, output_size):
        super(HiddenLayer, self).__init__()
        self.fc = nn.Linear(input_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.fc(x))


class MLP(nn.Module):
    def __init__(self, hidden_size=100, num_layers=1, activation_layer="sigmoid", input_dim = 1):
        super(MLP, self).__init__()
        self.activation_layer = activation_layer
        self.first_hidden_layer = HiddenLayer(input_dim, hidden_size)
        self.rest_hidden_layers = nn.Sequential(*[HiddenLayer(hidden_size, hidden_size) for _ in range(num_layers - 1)])
        self.output_layer = nn.Linear(hidden_size, 1)

    def forward(self, x, temp=1):
        x = self.first_hidden_layer(x)
        x = self.rest_hidden_layers(x)
        x = self.output_layer(x)
        if self.activation_layer == "sigmoid":
            return torch.sigmoid(x*temp)
        else:
            return torch.clamp(x, 0,1)

class MLP2(nn.Module):
    def __init__(self, hidden_size=100, num_layers=1, activation_layer="sigmoid",embed_dim_x=5, embed_dim_y=5):
        super(MLP2, self).__init__()
        self.activation_layer = activation_layer
        self.first_hidden_layer = HiddenLayer(1, embed_dim_x)
        self.second_hidden_layer = HiddenLayer(embed_dim_x+embed_dim_y, hidden_size)
        self.y_embeddings = torch.nn.Embedding(10, embed_dim_y)
        self.rest_hidden_layers = nn.Sequential(*[HiddenLayer(hidden_size, hidden_size) for _ in range(num_layers - 2)])
        self.output_layer = nn.Linear(hidden_size, 1)

    def forward(self, x, y, temp=1):
        loss_embed = self.first_hidden_layer(x)
        y_embed = self.y_embeddings(y)
        out = self.second_hidden_layer(torch.cat([loss_embed, y_embed], dim=1))
        out = self.rest_hidden_layers(out)
        out = self.output_layer(out)
        if self.activation_layer == "sigmoid":
            return torch.sigmoid(out*temp)
        else:
            return torch.clamp(out, 0,1)


class LabelProxy(nn.Module):
    def __init__(self, hidden_size=100, num_layers=1, num_classes = 10):
        super(LabelProxy, self).__init__()
        self.first_hidden_layer = HiddenLayer(2*num_classes, hidden_size)
        self.rest_hidden_layers = nn.Sequential(*[HiddenLayer(hidden_size, hidden_size) for _ in range(num_layers - 1)])
        self.output_layer = nn.Linear(hidden_size, num_classes)

    def forward(self, y_pred, y_hat, if_softmax=True):
        if if_softmax:
            y_pred = torch.softmax(y_pred, dim=1)
        concat = torch.cat([y_pred, y_hat], dim=1)
        out = self.first_hidden_layer(concat)
        out = self.rest_hidden_layers(out)
        out = self.output_layer(out)
        return out

class ConvNet(nn.Module):
    def __init__(self, output_dim, maxpool=True, base_hid=32):
        super(ConvNet, self).__init__()
        self.base_hid = base_hid
        self.conv1 = nn.Conv2d(1, base_hid, 5, 1)
        self.conv2 = nn.Conv2d(base_hid, base_hid*2, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * base_hid*2, base_hid*4)
        self.fc2 = nn.Linear(base_hid*4, output_dim)
        self.maxpool = maxpool

    def forward(self, x, return_feat=False):
        x = self.embed(x)
        out = self.fc2(x)
        if return_feat:
            return out, x.detach()
        return out

    def embed(self, x):
        x = F.relu(self.conv1(x))
        if self.maxpool:
            x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        if self.maxpool:
            x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 2*self.base_hid)
        x = F.relu(self.fc1(x))
        return x


class Lenet(nn.Module):

    def __init__(self, num_classes = 10):
        super(Lenet, self).__init__()
        self.conv1 = nn.Conv2d(1,6,5,stride=1,padding=2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1   = nn.Linear(400, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, num_classes)


    def forward(self, x):

        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)

        # clean = F.softmax(out, 1)


        return out

class sig_t(nn.Module):
    def __init__(self, device, num_classes, init=2):
        super(sig_t, self).__init__()

        self.register_parameter(name='w', param=nn.parameter.Parameter(-init*torch.ones(num_classes, num_classes)))

        self.w.to(device)

        co = torch.ones(num_classes, num_classes)
        ind = np.diag_indices(co.shape[0])
        co[ind[0], ind[1]] = torch.zeros(co.shape[0])
        self.co = co.to(device)
        self.identity = torch.eye(num_classes).to(device)


    def forward(self):
        sig = torch.sigmoid(self.w)
        T = self.identity.detach() + sig*self.co.detach()
        T = F.normalize(T, p=1, dim=1)

        return T