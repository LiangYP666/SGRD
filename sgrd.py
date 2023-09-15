from typing import Tuple, Union
import torch.nn as nn
import torch.nn.functional as F
import torch
import copy
from mmcv.cnn import constant_init, kaiming_init
from ..builder import DISTILL_LOSSES
from torch.nn import Softmax

def INF(B, H, W):
    return -torch.diag(torch.tensor(float("inf")).cuda().repeat(H), 0).unsqueeze(0).repeat(B * W, 1, 1)

@DISTILL_LOSSES.register_module()
class AXRFeatureLoss(nn.Module):
    """
    PyTorch version of 'Semantic and Global Relational Distillation for Object Detection'
    Args:
        student_channels(int): Number of channels in the student's feature map.
        teacher_channels(int): Number of channels in the teacher's feature map.
        name (str): the loss name of the layer.
        num_classes(int): The number of categories in the data set, COCO 80, SODA10M 6, PASCAL VOC 20.
        causal_attention_loss_weight(float): Hyperparameters of category-specific semantic knowledge distillation.
        ccnet_loss_weight(float): Hyperparameters of global context distillation
    """

    def __init__(self,
                 student_channels,
                 teacher_channels,
                 name,
                 num_classes=6,
                 causal_attention_loss_weight=0.0005,
                 ccnet_loss_weight=0.00001
                 ):
        super(AXRFeatureLoss, self).__init__()
        self.num_classes = num_classes
        self.name = name

        self.causal_attention_loss_weight = causal_attention_loss_weight
        self.ccnet_loss_weight = ccnet_loss_weight

        if student_channels != teacher_channels:
            self.align = nn.Conv2d(student_channels, teacher_channels, kernel_size=1, stride=1, padding=0)
        else:
            self.align = None

        self.classify = nn.Conv2d(256, num_classes, 1, 1, 0, bias=False)
        self.softmax = nn.Softmax(dim=2)

        # CCNET
        in_dim = teacher_channels  # 256
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.softmax1 = Softmax(dim=3)
        self.INF = INF
        self.gamma1 = nn.Parameter(torch.zeros(1))

        self.reset_parameters()

    def reset_parameters(self):
        kaiming_init(self.query_conv, mode='fan_in')
        kaiming_init(self.key_conv, mode='fan_in')
        kaiming_init(self.value_conv, mode='fan_in')
        self.query_conv.inited = True
        self.key_conv.inited = True
        self.value_conv.inited = True

    def norm(self, feat: torch.Tensor) -> torch.Tensor:
        """
        Normalize the feature maps to have zero mean and unit variances.
        Args:
            feat (torch.Tensor): The original feature map with shape
                (N, C, H, W).
        """
        assert len(feat.shape) == 4
        N, C, H, W = feat.shape
        feat = feat.permute(1, 0, 2, 3).reshape(C, -1)
        mean = feat.mean(dim=-1, keepdim=True)
        std = feat.std(dim=-1, keepdim=True)
        feat = (feat - mean) / (std + 1e-6)
        return feat.reshape(C, N, H, W).permute(1, 0, 2, 3)


    def forward(self,
                preds_S,
                preds_T,
                gt_bboxes,
                img_metas
                ):
        """Forward function.
        Args:
            preds_S(Tensor): Bs*C*H*W, student's feature map
            preds_T(Tensor): Bs*C*H*W, teacher's feature map
            gt_bboxes(tuple): Bs*[nt*4], pixel decimal: (tl_x, tl_y, br_x, br_y)
            img_metas (list[dict]): Meta information of each image, e.g.,
            image size, scaling factor, etc.
        """
        assert preds_S.shape[-2:] == preds_T.shape[-2:], 'the output dim of teacher and student differ'

        if self.align is not None:
            preds_S = self.align(preds_S)

        preds_S, preds_T = self.norm(preds_S), self.norm(preds_T)

        T_causal_attention = self.get_causal_attention(preds_T)
        S_causal_attention = self.get_causal_attention(preds_S)
        causal_attention_loss = self.get_attention_loss(T_causal_attention, S_causal_attention)

        ccnet_T = self.Ccnet(preds_T)
        ccnet_S = self.Ccnet(preds_S)
        ccnet_loss = self.get_ccnet_loss(ccnet_T, ccnet_S)

        loss = causal_attention_loss * self.causal_attention_loss_weight + ccnet_loss * self.ccnet_loss_weight

        return loss

    def get_causal_attention(self, x):
        """
         x: Bs*C*H*W

        """
        N, C, H, W = x.shape  # B 256 64 64
        device = x.device

        M = self.classify(x)  # B 20 64 64
        score = M.flatten(2)  # B 20 4096
        score_soft = self.softmax(score)  # B 20 4096

        flat_x = x.flatten(2)  # B 256 4096
        flat_x = flat_x.transpose(1, 2)  # B 4096 256
        class_feat = torch.matmul(score_soft, flat_x)  # B 20 256

        # out_proj = nn.Linear(256, 256).to(device)
        # attn_output = out_proj(class_feat)
        return class_feat

    def get_attention_loss(self, t_causal_attention, s_causal_attention):
        loss_mse = nn.MSELoss(reduction='sum')

        causal_attention_loss = loss_mse(t_causal_attention, s_causal_attention) / len(t_causal_attention)
        return causal_attention_loss

    def Ccnet(self, x):
        m_batchsize, _, height, width = x.size()
        proj_query = self.query_conv(x)
        proj_query_H = proj_query.permute(0, 3, 1, 2).contiguous().view(m_batchsize * width, -1, height).permute(0, 2, 1)
        proj_query_W = proj_query.permute(0, 2, 1, 3).contiguous().view(m_batchsize * height, -1, width).permute(0, 2, 1)
        proj_key = self.key_conv(x)
        proj_key_H = proj_key.permute(0, 3, 1, 2).contiguous().view(m_batchsize * width, -1, height)
        proj_key_W = proj_key.permute(0, 2, 1, 3).contiguous().view(m_batchsize * height, -1, width)
        proj_value = self.value_conv(x)
        proj_value_H = proj_value.permute(0, 3, 1, 2).contiguous().view(m_batchsize * width, -1, height)
        proj_value_W = proj_value.permute(0, 2, 1, 3).contiguous().view(m_batchsize * height, -1, width)
        energy_H = (torch.bmm(proj_query_H, proj_key_H) + self.INF(m_batchsize, height, width)).view(m_batchsize, width, height, height).permute(0, 2, 1, 3)

        energy_W = torch.bmm(proj_query_W, proj_key_W).view(m_batchsize, height, width, width)
        concate = self.softmax1(torch.cat([energy_H, energy_W], 3))

        att_H = concate[:, :, :, 0:height].permute(0, 2, 1, 3).contiguous().view(m_batchsize * width, height, height)
        # print(concate)
        # print(att_H)
        att_W = concate[:, :, :, height:height + width].contiguous().view(m_batchsize * height, width, width)
        out_H = torch.bmm(proj_value_H, att_H.permute(0, 2, 1)).view(m_batchsize, width, -1, height).permute(0, 2, 3, 1)
        out_W = torch.bmm(proj_value_W, att_W.permute(0, 2, 1)).view(m_batchsize, height, -1, width).permute(0, 2, 1, 3)
        # print(out_H.size(),out_W.size())

        return self.gamma1 * (out_H + out_W) + x

    def get_ccnet_loss(self,  ccnet_T, ccnet_S):
        loss_mse = nn.MSELoss(reduction='sum')
        ccnet_loss = loss_mse(ccnet_T, ccnet_S) / len(ccnet_T)

        return ccnet_loss
