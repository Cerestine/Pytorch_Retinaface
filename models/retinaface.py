"""RetinaFace Network from deepinsight
https://github.com/deepinsight/insightface/tree/master/detection/retinaface
https://arxiv.org/abs/1905.00641"""
from collections import OrderedDict
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models.detection import backbone_utils
from torchvision.models import _utils
from models.net import MobileNetV1
from models.net import FPN
from models.net import SSH


class ClassHead(nn.Module):
    """Class head"""
    def __init__(self,inchannels=512,num_anchors=3):
        super(ClassHead,self).__init__()
        self.num_anchors = num_anchors
        self.conv1x1 = nn.Conv2d(inchannels,self.num_anchors*2,kernel_size=(1,1),stride=1,padding=0)

    def forward(self,x):
        """Class forward"""
        out = self.conv1x1(x)
        out = out.permute(0,2,3,1).contiguous()
        return out.view(out.shape[0], -1, 2)

class BboxHead(nn.Module):
    """Bbox head"""
    def __init__(self,inchannels=512,num_anchors=3):
        super(BboxHead,self).__init__()
        self.conv1x1 = nn.Conv2d(inchannels,num_anchors*4,kernel_size=(1,1),stride=1,padding=0)

    def forward(self,x):
        """Bbox forward"""
        out = self.conv1x1(x)
        out = out.permute(0,2,3,1).contiguous()

        return out.view(out.shape[0], -1, 4)

class LandmarkHead(nn.Module):
    """Landmark head"""
    def __init__(self,inchannels=512,num_anchors=3):
        super(LandmarkHead,self).__init__()
        self.conv1x1 = nn.Conv2d(inchannels,num_anchors*10,kernel_size=(1,1),stride=1,padding=0)

    def forward(self,x):
        """Landmark forward"""
        out = self.conv1x1(x)
        out = out.permute(0,2,3,1).contiguous()

        return out.view(out.shape[0], -1, 10)

class RetinaFace(nn.Module):
    """RetinaFace class"""
    def __init__(self, cfg = None, phase = 'train'):
        """
        :param cfg:  Network related settings.
        :param phase: train or test.
        """
        super(RetinaFace, self).__init__()
        self.phase = phase
        self.cfg = cfg

        self._make_body()
        self._make_head()
        self._make_channels()

    def _load_backbone(self, cfg):
        if cfg['name'] == 'mobilenet0.25':
            backbone = MobileNetV1()
            if cfg['pretrain']:
                checkpoint = torch.load(
                    "./weights/mobilenetV1X0.25_pretrain.tar", 
                    map_location=torch.device('cpu'))
                new_state_dict = OrderedDict()
                for k, v in checkpoint['state_dict'].items():
                    name = k[7:]  # remove module.
                    new_state_dict[name] = v
                # load params
                backbone.load_state_dict(new_state_dict)
        elif cfg['name'] == 'Resnet50':
            backbone = models.resnet50(pretrained=cfg['pretrain'])
        else:
            backbone = None
        return backbone

    def _make_class_head(self, fpn_num=3, inchannels=64, anchor_num=2):
        classhead = nn.ModuleList()
        for i in range(fpn_num):
            classhead.append(ClassHead(inchannels, anchor_num))
        return classhead

    def _make_bbox_head(self, fpn_num=3, inchannels=64, anchor_num=2):
        bboxhead = nn.ModuleList()
        for i in range(fpn_num):
            bboxhead.append(BboxHead(inchannels, anchor_num))
        return bboxhead

    def _make_landmark_head(self, fpn_num=3, inchannels=64, anchor_num=2):
        landmarkhead = nn.ModuleList()
        for i in range(fpn_num):
            landmarkhead.append(LandmarkHead(inchannels, anchor_num))
        return landmarkhead

    def _make_body(self):
        """Load backbone and make body"""
        backbone = self._load_backbone(self.cfg)
        self.body = _utils.IntermediateLayerGetter(backbone, self.cfg['return_layers'])

    def _make_head(self):
        """Make head layer"""
        self.ClassHead = self._make_class_head(fpn_num=3, inchannels=self.cfg['out_channel'])
        self.BboxHead = self._make_bbox_head(fpn_num=3, inchannels=self.cfg['out_channel'])
        self.LandmarkHead = self._make_landmark_head(fpn_num=3, inchannels=self.cfg['out_channel'])

    def _make_channels(self):
        in_channels_stage2 = self.cfg['in_channel']
        in_channels_list = [
            in_channels_stage2 * 2,
            in_channels_stage2 * 4,
            in_channels_stage2 * 8,
        ]
        out_channels = self.cfg['out_channel']
        self.fpn = FPN(in_channels_list, out_channels)
        self.ssh1 = SSH(out_channels, out_channels)
        self.ssh2 = SSH(out_channels, out_channels)
        self.ssh3 = SSH(out_channels, out_channels)

    # def _concat_tensors(self, head, features):
    #     for i, feature in enumerate(features):
    #         head[i](feature)
    #     return torch.cat(head, dim=1)
    
    def _concat_tensors(self, head, features):
        return torch.cat([head[i](feature) for i, feature in enumerate(features)], dim=1)

    def forward(self, inputs):
        """Make output layer"""
        out = self.body(inputs)

        # FPN
        fpn = self.fpn(out)

        # SSH
        feature1 = self.ssh1(fpn[0])
        feature2 = self.ssh2(fpn[1])
        feature3 = self.ssh3(fpn[2])
        features = [feature1, feature2, feature3]

        bbox_regressions = self._concat_tensors(self.BboxHead, features)
        classifications = self._concat_tensors(self.ClassHead, features)
        ldm_regressions = self._concat_tensors(self.LandmarkHead, features)

        if self.phase == 'train':
            output = (bbox_regressions, classifications, ldm_regressions)
        else:
            output = (bbox_regressions, F.softmax(classifications, dim=-1), ldm_regressions)
        return output
