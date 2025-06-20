from torchvision.models import ResNet
from torchvision.models.resnet import Bottleneck, BasicBlock
import torch.utils.model_zoo as model_zoo
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


class ResNet_cam(ResNet):
    arch_settings = {
        18: (BasicBlock, (2, 2, 2, 2)),
        34: (BasicBlock, (3, 4, 6, 3)),
        50: (Bottleneck, (3, 4, 6, 3)),
        101: (Bottleneck, (3, 4, 23, 3)),
        152: (Bottleneck, (3, 8, 36, 3))
    }

    def __init__(self, num_classes, depth=101, cutmix=None):
        self.block, self.layers = self.arch_settings[depth]
        self.depth = depth
        super(ResNet_cam, self).__init__(self.block, self.layers)
        self.init_weights(pretrained=True, cutmix=cutmix)
        self.loss_func = F.binary_cross_entropy_with_logits
        self.fc_all = nn.Linear(512 * 4, num_classes)
        self.num_labels = num_classes

        for class_c in range(num_classes):
            selector = nn.Sequential(
                nn.Linear(512 * 4, 512 * 4),
                nn.BatchNorm1d(512 * 4),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Linear(512 * 4, 1)
            )
            self.__setattr__('contrastive_class_selector_' + str(class_c), selector)

    def backbone(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        return x

    def forward_train(self, x):
        x = self.backbone(x)

        feat = x
        N, C, H, W = feat.shape
        # global
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        logit = self.fc_all(x)
        params = list(self.parameters())
        #fc_weights = params[-2].data
        fc_weights = params[312].data
        fc_weights = fc_weights.view(1, self.num_labels, C, 1, 1)
        fc_weights = Variable(fc_weights, requires_grad=False)
        feat = feat.unsqueeze(1)  # N * 1 * C * H * W
        hm = feat * fc_weights
        hm = hm.sum(2)  # N * self.num_labels * H * W
        heatmap = hm

        return logit, feat.squeeze(1), heatmap

    def forward_test(self, x):
        x = self.backbone(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc_all(x)
        return x

    def forward(self, x, cam=False):
        if cam is not False:
            return self.forward_train(x)
        if cam is False:
            return self.forward_test(x)

    def init_weights(self, pretrained=True, cutmix=None):
        if cutmix is not None:
            print("backbone params inited by CutMix pretrained model")
            state_dict = torch.load(cutmix)
        elif pretrained:
            print("backbone params inited by Pytorch official model")
            model_url = model_urls["resnet{}".format(self.depth)]
            state_dict = model_zoo.load_url(model_url)

        model_dict = self.state_dict()
        try:
            pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict}
            self.load_state_dict(pretrained_dict)
        except:
            logger = logging.getLogger()
            logger.info(
                "the keys in pretrained model is not equal to the keys in the ResNet you choose, trying to fix...")
            state_dict = self._keysFix(model_dict, state_dict)
            self.load_state_dict(state_dict)

        # remove the original 1000-class fc
        self.fc = nn.Sequential() 