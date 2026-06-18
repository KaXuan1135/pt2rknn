import os
import torch
import numpy as np
import torch.nn.functional as F

from torch import nn, Tensor
from typing import Optional, List
from types import SimpleNamespace

DEVICE = 'cpu'

class AnchorPoints(nn.Module):
    # Copyright (C) 2021 THL A29 Limited, a Tencent company.  All rights reserved. 
    # Single Layer Version
    def __init__(self, pyramid_levels=None, stride=None, row=3, line=3):
        super(AnchorPoints, self).__init__()
        self.pyramid_level = pyramid_levels  # default: 3
        if stride is None:
            self.stride = 2 ** self.pyramid_level # default: 8
        else:
            self.stride = stride  # default 8
        self.row = row
        self.line = line

    def forward(self, image):
        image_shape = image.shape[2:] # image shape: h,w
        image_shape = np.array(image_shape)
        image_shapes = (image_shape + self.stride - 1) // self.stride # get downsample scale
        # get reference points for each level
        # each anchor block expand the number (row * line) of anchors
        anchor_points = self._generate_anchor_points(self.stride, row=self.row, line=self.line) # control the shift in the anchor block
        # anchor_map
        shifted_anchor_points = self._shift(image_shapes, self.stride, anchor_points)
        # get final anchor_map
        all_anchor_points = np.zeros((0, 2)).astype(np.float32)
        all_anchor_points = np.append(all_anchor_points, shifted_anchor_points, axis=0)
        all_anchor_points = np.expand_dims(all_anchor_points, axis=0)
        # send reference points to device
        return torch.from_numpy(all_anchor_points.astype(np.float32)).to(DEVICE)

    def _generate_anchor_points(self, stride=16, row=3, line=3):
        # generate the reference points in grid layout
        row_step = stride / row
        line_step = stride / line

        shift_x = (np.arange(1, line + 1) - 0.5) * line_step - stride / 2
        shift_y = (np.arange(1, row + 1) - 0.5) * row_step - stride / 2

        shift_x, shift_y = np.meshgrid(shift_x, shift_y)

        anchor_points = np.vstack((
            shift_x.ravel(), shift_y.ravel()
        )).transpose()
        return anchor_points
    
    def _shift(self, shape, stride, anchor_points):  # shape is feature map shape
        # shift the meta-anchor to get an acnhor points
        shift_x = (np.arange(0, shape[1]) + 0.5) * stride
        shift_y = (np.arange(0, shape[0]) + 0.5) * stride

        shift_x, shift_y = np.meshgrid(shift_x, shift_y)

        shifts = np.vstack((
            shift_x.ravel(), shift_y.ravel()
        )).transpose()

        A = anchor_points.shape[0]  # num_of_points
        K = shifts.shape[0]  # num_of_pixel 
        all_anchor_points = (anchor_points.reshape((1, A, 2)) + shifts.reshape((1, K, 2)).transpose((1, 0, 2)))
        all_anchor_points = all_anchor_points.reshape((K * A, 2))
        return all_anchor_points

class SpatialEncoding(nn.Module):
    def __init__(self, 
                 in_dim, 
                 out_dim, 
                 sigma = 6,
                 cat_input=True,
                 require_grad=False,):

        super().__init__()
        assert out_dim % (2*in_dim) == 0, "dimension must be dividable"

        n = out_dim // 2 // in_dim
        m = 2**np.linspace(0, sigma, n)
        m = np.stack([m] + [np.zeros_like(m)]*(in_dim-1), axis=-1)
        m = np.concatenate([np.roll(m, i, axis=-1) for i in range(in_dim)], axis=0)
        self.emb = torch.FloatTensor(m)
        if require_grad:
            self.emb = nn.Parameter(self.emb, requires_grad=True)    
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.sigma = sigma
        self.cat_input = cat_input
        self.require_grad = require_grad

    def forward(self, x):
        assert self.require_grad
        assert self.cat_input
        self.emb = self.emb.to(x.device)
        y = torch.matmul(x, self.emb.T)
        return torch.cat([x, torch.sin(y), torch.cos(y)], dim=-1)

def make_coord(shape, ranges=None, flatten=True):
    """ Make coordinates at grid centers.
    """
    coord_seqs = []
    for i, n in enumerate(shape): # shape: h,w
        if ranges is None:
            v0, v1 = -1, 1
        else:
            v0, v1 = ranges[i]
        r = (v1 - v0) / (2 * n)
        seq = v0 + r + (2 * r) * torch.arange(n).float()
        coord_seqs.append(seq)
    ret = torch.stack(torch.meshgrid(*coord_seqs), dim=-1)
    if flatten:
        ret = ret.view(-1, ret.shape[-1])
    # print("Coord:", ret.size())
    return ret

def ifi_feat_original(res, size, stride=1, local=False):
    '''
    res is input feature map, size is target scale (h, w)
    rel_coord is target mapping with feature coords.
    ex. target size 64*64 will find the nearest feature coords.
    '''
    assert local
    # local is define local patch: 3*3 mapping near by center point.
    bs, hh, ww = res.shape[0], res.shape[-2], res.shape[-1]
    h , w = size
    coords = (make_coord((h,w)).to(DEVICE).flip(-1) + 1) / 2
    coords = coords.unsqueeze(0).expand(bs, *coords.shape)
    coords = (coords*2-1).flip(-1)

    feat_coords = make_coord((hh,ww), flatten=False).to(DEVICE).permute(2, 0, 1) .unsqueeze(0).expand(res.shape[0], 2, *(hh,ww))

    vx_list = [-1, 1]
    vy_list = [-1, 1]
    eps_shift = 1e-6
    rel_coord_list = []
    q_feat_list = []
    area_list = []

    rx = stride / h 
    ry = stride / w 
    
    for vx in vx_list:
        for vy in vy_list:
            coords_ = coords.clone()
            coords_[:,:,0] += vx * rx + eps_shift
            coords_[:,:,1] += vy * ry + eps_shift
            coords_.clamp_(-1+1e-6, 1-1e-6)
            q_feat = F.grid_sample(res, coords_.flip(-1).unsqueeze(1),mode='nearest',align_corners=False)
            q_coord = F.grid_sample(feat_coords, coords_.flip(-1).unsqueeze(1),mode='nearest',align_corners=False)
            
            q_feat = q_feat[:,:,0,:].permute(0,2,1)
            q_coord = q_coord[:,:,0,:].permute(0,2,1)
            
            rel_coord = coords - q_coord
            rel_coord[:,:,0] *= hh#res.shape[-2]
            rel_coord[:,:,1] *= ww#res.shape[-1]

            rel_coord_list.append(rel_coord)
            q_feat_list.append(q_feat)
            area = torch.abs(rel_coord[:, :, 0] * rel_coord[:, :, 1])
            area_list.append(area+1e-9)

    return rel_coord_list, q_feat_list, area_list

def rknn_compatible_sample_nearest(source, grid, align_corners=False):
    """
    Memory-efficient replacement for F.grid_sample(mode='nearest').
    Optimized for RKNN/NPU to prevent OOM during compilation.
    
    Args:
        source: [N, C, H, W]
        grid: [N, 1, M, 2] (Standard grid_sample format: x, y)
        align_corners: bool
    """
    N, C, H, W = source.shape
    _, _, M, _ = grid.shape

    # 1. Separate X and Y coordinates (grid is [..., 0] for X, [..., 1] for Y)
    x = grid[..., 0] 
    y = grid[..., 1]

    # 2. Map coordinates to pixel indices based on align_corners
    if align_corners:
        # Range [-1, 1] -> [0, W-1]
        x_phys = (x + 1) * (W - 1) / 2
        y_phys = (y + 1) * (H - 1) / 2
    else:
        # Range [-1, 1] -> [-0.5, W-0.5]
        x_phys = ((x + 1) * W - 1) / 2
        y_phys = ((y + 1) * H - 1) / 2

    # 3. Mode='nearest' logic: Round and Clamp
    # We use .long() which acts like floor, so we add 0.5 to simulate rounding
    x_idx = torch.clamp((x_phys + 0.5).floor(), 0, W - 1).long()
    y_idx = torch.clamp((y_phys + 0.5).floor(), 0, H - 1).long()

    # 4. Memory-Efficient Indexing (Avoids GatherElements/Einsum)
    # We use a batch-loop for the export. While it looks slower in Python,
    # it generates a much cleaner ONNX graph for RKNN.
    output_batches = []
    for i in range(N):
        # Indexing [C, H, W] with [1, M] indices
        # This usually exports to 'Gather' or 'GatherND' ops
        sampled = source[i, :, y_idx[i, 0], x_idx[i, 0]] # Result: [C, M]
        output_batches.append(sampled.unsqueeze(0))

    # 5. Restore to [N, C, 1, M] to match grid_sample output
    return torch.cat(output_batches, dim=0).unsqueeze(2)

def ifi_feat(res, size, stride=1, local=False):
    '''
    res is input feature map, size is target scale (h, w)
    rel_coord is target mapping with feature coords.
    ex. target size 64*64 will find the nearest feature coords.
    '''
    assert local
    # local is define local patch: 3*3 mapping near by center point.
    bs, hh, ww = res.shape[0], res.shape[-2], res.shape[-1]
    h , w = size
    coords = (make_coord((h,w)).to(DEVICE).flip(-1) + 1) / 2
    coords = coords.unsqueeze(0).expand(bs, *coords.shape)
    coords = (coords*2-1).flip(-1)

    feat_coords = make_coord((hh,ww), flatten=False).to(DEVICE).permute(2, 0, 1) .unsqueeze(0).expand(res.shape[0], 2, *(hh,ww))

    vx_list = [-1, 1]
    vy_list = [-1, 1]
    eps_shift = 1e-6
    rel_coord_list = []
    q_feat_list = []
    area_list = []

    rx = stride / h 
    ry = stride / w 
    
    for vx in vx_list:
        for vy in vy_list:
            coords_ = coords.clone()
            coords_[:,:,0] += vx * rx + eps_shift
            coords_[:,:,1] += vy * ry + eps_shift
            coords_.clamp_(-1+1e-6, 1-1e-6)

            q_feat = rknn_compatible_sample_nearest(res, coords_.flip(-1).unsqueeze(1), align_corners=False)
            q_coord = rknn_compatible_sample_nearest(feat_coords, coords_.flip(-1).unsqueeze(1), align_corners=False)

            q_feat = q_feat[:,:,0,:].permute(0,2,1)
            q_coord = q_coord[:,:,0,:].permute(0,2,1)
            
            rel_coord = coords - q_coord
            rel_coord[:,:,0] *= hh#res.shape[-2]
            rel_coord[:,:,1] *= ww#res.shape[-1]

            rel_coord_list.append(rel_coord)
            q_feat_list.append(q_feat)
            area = torch.abs(rel_coord[:, :, 0] * rel_coord[:, :, 1])
            area_list.append(area+1e-9)

    return rel_coord_list, q_feat_list, area_list

class ASPP(nn.Module):
    """
    Reference:
        Chen, Liang-Chieh, et al. *"Rethinking Atrous Convolution for Semantic Image Segmentation."*
    """
    def __init__(self, in_planes, inner_planes=256, sync_bn=False, bn=False, dilations=(12, 24, 36)):
        super(ASPP, self).__init__()

        norm_layer = nn.SyncBatchNorm if sync_bn else nn.BatchNorm2d
        if bn == False:
            self.conv1 = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                    nn.Conv2d(in_planes, inner_planes, kernel_size=1, padding=0, dilation=1, bias=False),
                                    nn.ReLU(inplace=True))
            self.conv2 = nn.Sequential(nn.Conv2d(in_planes, inner_planes, kernel_size=1, padding=0, dilation=1, bias=False),
                                    nn.ReLU(inplace=True))
            self.conv3 = nn.Sequential(nn.Conv2d(in_planes, inner_planes, kernel_size=3,
                                    padding=dilations[0], dilation=dilations[0], bias=False),
                                    nn.ReLU(inplace=True))
            self.conv4 = nn.Sequential(nn.Conv2d(in_planes, inner_planes, kernel_size=3,
                                    padding=dilations[1], dilation=dilations[1], bias=False),
                                    nn.ReLU(inplace=True))
            self.conv5 = nn.Sequential(nn.Conv2d(in_planes, inner_planes, kernel_size=3,
                                    padding=dilations[2], dilation=dilations[2], bias=False),
                                    nn.ReLU(inplace=True))        
        else:
            self.conv1 = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                    nn.Conv2d(in_planes, inner_planes, kernel_size=1, padding=0, dilation=1, bias=False),
                                    norm_layer(inner_planes),
                                    nn.ReLU(inplace=True))
            self.conv2 = nn.Sequential(nn.Conv2d(in_planes, inner_planes, kernel_size=1, padding=0, dilation=1, bias=False),
                                    norm_layer(inner_planes),
                                    nn.ReLU(inplace=True))
            self.conv3 = nn.Sequential(nn.Conv2d(in_planes, inner_planes, kernel_size=3,
                                    padding=dilations[0], dilation=dilations[0], bias=False),
                                    norm_layer(inner_planes),
                                    nn.ReLU(inplace=True))
            self.conv4 = nn.Sequential(nn.Conv2d(in_planes, inner_planes, kernel_size=3,
                                    padding=dilations[1], dilation=dilations[1], bias=False),
                                    norm_layer(inner_planes),
                                    nn.ReLU(inplace=True))
            self.conv5 = nn.Sequential(nn.Conv2d(in_planes, inner_planes, kernel_size=3,
                                    padding=dilations[2], dilation=dilations[2], bias=False),
                                    norm_layer(inner_planes),
                                    nn.ReLU(inplace=True))
        self.out_planes = (len(dilations) + 2) * inner_planes

    def get_outplanes(self):
        return self.out_planes

    def forward(self, x):
        _, _, h, w = x.size()
        feat1 = F.upsample(self.conv1(x), size=(h, w), mode='bilinear', align_corners=True)
        feat2 = self.conv2(x)
        feat3 = self.conv3(x)
        feat4 = self.conv4(x)
        feat5 = self.conv5(x)
        aspp_out = torch.cat((feat1, feat2, feat3, feat4, feat5), 1)
        return aspp_out

class ifi_simfpn(nn.Module):
    def __init__(self, ultra_pe=False, pos_dim=40, sync_bn=False, num_anchor_points=4, num_classes=2, local=False, unfold=False, 
                 stride=1, learn_pe=False, require_grad=False, head_layers=[512,256,256], feat_num=4, feat_dim=256):  
        # feat_num: # of encoder feats; num_anchor_points: line*row, num of queries each pixel;
        # num_classes: 2 {'regression':2, 'classifier':confidence},
        # ultra_pe/learn_pe: additional position encoding, pos_dim: additional position encoding dimension
        # local/unflod: impose the local feature information, head_layers: head layers setting, defaults=[512,256,256]
        # feat_dim: input_feats_dims
        super(ifi_simfpn, self).__init__()
        self.pos_dim = pos_dim
        self.ultra_pe = ultra_pe
        self.local = local
        self.unfold = unfold
        self.stride = stride
        self.learn_pe = learn_pe
        self.feat_num = feat_num
        self.feat_dim = feat_dim
        self.num_anchor_points = num_anchor_points
        self.regression_dims = 2  # fixed, coords
        self.num_classes = num_classes  # default: 2, confidence
        self.head_layers = head_layers
        norm_layer = nn.SyncBatchNorm if sync_bn else nn.BatchNorm1d # nn.BatchNorm1d / nn.InstanceNorm2d

        for level in range(self.feat_num):
            self._update_property('pos'+str(level+1), SpatialEncoding(2, self.pos_dim, require_grad=require_grad))
        self.pos_dim += 2

        # Predict Heads
        in_dim = self.feat_num*(self.feat_dim + self.pos_dim)
        if unfold:
            in_dim = self.feat_num*(self.feat_dim*9 + self.pos_dim)
        self.in_dim = in_dim

        confidence_head_list = []
        offset_head_list = []

        for ct, hidden_feature in enumerate(head_layers):
            if ct == 0:
                src_dim = in_dim
            else:
                src_dim = head_layers[ct-1]
            confidence_head_list.append([nn.Conv1d(src_dim, hidden_feature, 1), norm_layer(hidden_feature), nn.ReLU()])
            offset_head_list.append([nn.Conv1d(src_dim, hidden_feature, 1), norm_layer(hidden_feature), nn.ReLU()])
        
        confidence_head_list.append([nn.Conv1d(head_layers[-1], self.num_anchor_points*self.num_classes, 1), nn.ReLU()])
        offset_head_list.append([nn.Conv1d(head_layers[-1], self.num_anchor_points*2, 1)])

        # build heads
        confidence_head_list = [item for sublist in confidence_head_list for item in sublist]
        offset_head_list = [item for sublist in offset_head_list for item in sublist]

        self.confidence_head = nn.Sequential(*confidence_head_list)
        self.offset_head = nn.Sequential(*offset_head_list)

    def forward(self, x, size, level=0, after_cat=False):
        h, w = size
        if not after_cat:
            assert self.local
            assert not self.unfold
            rel_coord_list, q_feat_list, area_list = ifi_feat(x, [h, w],  local=True, stride=self.stride)
            total_area = torch.stack(area_list).sum(dim=0)
            context_list = []
            for rel_coord, q_feat, area in zip(rel_coord_list, q_feat_list, area_list):
                rel_coord = eval('self.pos'+str(level))(rel_coord)
                dynamic_zero = x.detach().mean() * 0 # this is a must, to prevent rknn-conversion from folding it
                rel_coord = rel_coord + dynamic_zero
                context_list.append(torch.cat([rel_coord, q_feat], dim=-1))
            ret = 0
            t = area_list[0]; area_list[0] = area_list[3]; area_list[3] = t
            t = area_list[1]; area_list[1] = area_list[2]; area_list[2] = t
            for conte, area in zip(context_list, area_list):
                x = ret + conte *  ((area / total_area).unsqueeze(-1))          
            return x
        else:  # make output
            ############### for Conv1d ##############
            # offset regression
            offset = self.offset_head(x).view(x.shape[0], -1, h, w)
            offset = offset.permute(0, 2, 3, 1) # # b, h, w, line*cow*2
            offset = offset.contiguous().view(x.shape[0], -1, 2)  # b, num_queries, 2     

            # classifier
            confidence = self.confidence_head(x).view(x.shape[0], -1, h, w)
            confidence = confidence.permute(0, 2, 3, 1) # # b, h, w, line*cow*num_classes
            confidence = confidence.contiguous().view(x.shape[0], -1, self.num_classes)  # b, num_queries, self.num_classes
            return offset, confidence

    def _update_property(self, property, value):
        setattr(self, property, value)

class IFI_Decoder_Model(nn.Module):
    def __init__(self, in_planes, feat_layers=[3,4], num_classes=2, num_anchor_points=4,line=2, row=2, anchor_stride=None, inner_planes=256, 
                 sync_bn=False, dilations=(2, 4, 8), require_grad=False, head_layers=[512,256,256], out_type='Normal',
                 pos_dim=24, ultra_pe=False, learn_pe=False, unfold=False, local=False, no_aspp=False, stride=1, **kwargs):
        """
        in_planes: encoder_outplanes; feat_layers: use_encoder_feats; inner_planes: trans_ens_outplanes;
        dilations: only for ASPP; out_type: type of output layer.
        num_anchor_points = line*row
        num_classes = num of classes

        -- For IfI modules:
        # feat_num: # of encoder feats; num_anchor_points: line*row, num of queries each pixel;
        # num_classes: 2 {'regression':2(momentum), 'classifier':confidence},
        # ultra_pe/learn_pe: additional position encoding, pos_dim: additional position encoding dimension
        # local/unfold: impose the local feature information, head_layers: head layers setting, defaults=[512,256,256]
        # feat_dim: input_feats_dims=inner_planes

        -- forward: {'pred_logits', 'pred_points', 'offset'}
        """
        super().__init__()
        norm_layer = nn.SyncBatchNorm if sync_bn else nn.BatchNorm2d
        self.in_planes = in_planes          # feat1,2,3,4's out_dims; default: VGG16, [128, 256, 512, 512]
        self.inner_planes = inner_planes    # default: 256
        self.num_anchor_points = num_anchor_points
        self.num_classes = num_classes      # default: 2

        self.feat_num = len(feat_layers)    # change the encoder feature num.
        self.feat_layers = feat_layers      # control the number of decoder features.
        self.no_aspp = no_aspp
        self.unfold = unfold
        self.out_type = out_type
        self.num_anchor_points = num_anchor_points
        self.num_classes = num_classes
        
        # build modules
        # Embedding Decoder.
        if 1 in self.feat_layers:
            self.enc1 = nn.Sequential(nn.Conv2d(self.in_planes[0], inner_planes, kernel_size=1), norm_layer(inner_planes), nn.ReLU(inplace=True))
        if 2 in self.feat_layers:
            self.enc2 = nn.Sequential(nn.Conv2d(self.in_planes[1], inner_planes, kernel_size=1), norm_layer(inner_planes), nn.ReLU(inplace=True))
        if 3 in self.feat_layers:
            self.enc3 = nn.Sequential(nn.Conv2d(self.in_planes[2], inner_planes, kernel_size=1), norm_layer(inner_planes), nn.ReLU(inplace=True))
        if 4 in self.feat_layers:
            if self.no_aspp:
                self.head = nn.Sequential(nn.Conv2d(self.in_planes[-1], inner_planes, kernel_size=1), norm_layer(inner_planes), nn.ReLU(inplace=True))
            else:
                self.aspp = ASPP(self.in_planes[-1], inner_planes=inner_planes, sync_bn=sync_bn, dilations=dilations)
                self.head = nn.Sequential(
                    nn.Conv2d(self.aspp.get_outplanes(), inner_planes, kernel_size=3, padding=1, dilation=1, bias=False),
                    norm_layer(inner_planes),
                    nn.ReLU(inplace=True),
                    nn.Dropout2d(0.1))   
        # IFI Module: position_encoding + regression + classifier
        self.ifi = ifi_simfpn(ultra_pe=ultra_pe, pos_dim=pos_dim, sync_bn=sync_bn, 
                              num_anchor_points=self.num_anchor_points, num_classes=self.num_classes, 
                              local=local, unfold=unfold, stride=stride, learn_pe=learn_pe, 
                              require_grad=require_grad, head_layers=head_layers, feat_num=self.feat_num, feat_dim=inner_planes)
        # Output Decoder.
        if self.out_type == 'Conv':        
            raise NotImplemented
        elif self.out_type == "Deconv":      
            raise NotImplemented

        # Align to real coords
        self.anchor_stride, self.row, self.line = anchor_stride, row, line
        self.anchor_points = AnchorPoints(pyramid_levels=self.feat_layers[0], stride=anchor_stride, row=row, line=line)
        
        # Auxiliary Anchors
        self.aux_en = kwargs['AUX_EN']
        self.aux_number = kwargs['AUX_NUMBER']
        self.aux_range = kwargs['AUX_RANGE']
        self.aux_kwargs = kwargs['AUX_kwargs']

    def forward(self, samples, features):
        # feats is [feat1, feat2, feat3, feat4]
        # align to max_shape of input_feat by implicit function
        ht, wt = features[self.feat_layers[0]-1].shape[-2], features[self.feat_layers[0]-1].shape[-1]
        batch_size = features[0].shape[0]
        
        # Embedding encoding.
        target_feat = []
        if 1 in self.feat_layers:
            x1 = self.enc1(features[0])
            target_feat.append(x1)
        if 2 in self.feat_layers:
            x2 = self.enc2(features[1])
            target_feat.append(x2)
        if 3 in self.feat_layers:
            x3 = self.enc3(features[2])
            target_feat.append(x3)
        if 4 in self.feat_layers:
            if self.no_aspp:
                aspp_out = self.head(features[-1])
            else:
                aspp_out = self.aspp(features[-1])
                aspp_out = self.head(aspp_out)
            target_feat.append(aspp_out)

        # IFI module forward: position_encoding + offset_head + confidence_head
        context = []
        for i, feat in enumerate(target_feat):
            context.append(self.ifi(feat, size=[ht, wt], level=i+1))
        context = torch.cat(context, dim=-1).permute(0,2,1)
        offset, confidence = self.ifi(context, size=[ht, wt], after_cat=True)

        # output decoder.
        if self.out_type == 'Conv':        
            raise KeyError('{} is not finished'.format(self.out_type))
        elif self.out_type == 'Deconv':
            raise KeyError('{} is not finished'.format(self.out_type))

        # Transform output
        offset *= 100   
        anchor_points = self.anchor_points(samples).repeat(batch_size, 1, 1)   # get sample point map
        output_coord = offset + anchor_points   # [b, h/d*w/d*line*row, 2(x,y)]  # transfer feature coordinate moving to image coordinate. (correspond to head center)
        output_confid = confidence              # [b, h/d*w/d*line*row, 2(confidence)]
        out = {'pred_logits': output_confid, 'pred_points': output_coord, 'offset': offset} 

        if not self.aux_en or not self.training:
            return out
        else:
            raise NotImplemented                # still refinement, will be announced ASAP
            out['aux'] = None
            return out

class VGG(nn.Module):
    def __init__(self, features, num_classes=1000, init_weights=True):
        super(VGG, self).__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

cfgs = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

def make_layers(cfg, batch_norm=False, sync=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                if sync:
                    print('use sync backbone')
                    layers += [conv2d, nn.SyncBatchNorm(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

def _vgg(arch, cfg, batch_norm, pretrained, progress, sync=False, **kwargs):
    kwargs['init_weights'] = False
    model = VGG(make_layers(cfgs[cfg], batch_norm=batch_norm, sync=sync), **kwargs)
    state_dict = torch.load(os.environ['BACKBONE_CKPT'])
    model.load_state_dict(state_dict)
    return model

def vgg16_bn(pretrained=False, progress=True, sync=False, **kwargs):
    r"""VGG 16-layer model (configuration "D") with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg16_bn', 'D', True, pretrained, progress, sync=sync, **kwargs)

class Base_VGG(nn.Module):
    def __init__(self, name: str, last_pool=False , num_channels=256, **kwargs):
        super().__init__()
        # loading backbone features
        if name == 'vgg16_bn':
            backbone = vgg16_bn(pretrained=True)
        elif name == 'vgg16':
            assert 0
            # backbone = VGG.vgg16(pretrained=True)
        
        features = list(backbone.features.children())

        # setting base module.
        if name == 'vgg16_bn':
            self.body1 = nn.Sequential(*features[:13])
            self.body2 = nn.Sequential(*features[13:23])
            self.body3 = nn.Sequential(*features[23:33])
            if last_pool:
                self.body4 = nn.Sequential(*features[33:44])  # 32x down-sample
            else:
                self.body4 = nn.Sequential(*features[33:43])  # 16x down-sample
        else:
            self.body1 = nn.Sequential(*features[:9])
            self.body2 = nn.Sequential(*features[9:16])
            self.body3 = nn.Sequential(*features[16:23])
            if last_pool:
                self.body4 = nn.Sequential(*features[23:31])  # 32x down-sample
            else:
                self.body4 = nn.Sequential(*features[23:30])  # 16x down-sample
        self.num_channels = num_channels
        self.last_pool = last_pool
        
    def get_outplanes(self):
        outplanes = []
        for i in range(4):
            last_dims = 0
            for param_tensor in self.__getattr__('body'+str(i+1)).state_dict():
                if 'weight' in param_tensor:
                    last_dims = list(self.__getattr__('body'+str(i+1)).state_dict()[param_tensor].size())[0]
            outplanes.append(last_dims)
        return outplanes   # get the last layer params of all modules, and trans to the size.

    def forward(self, tensor_list):
        out = []
        xs = tensor_list
        for _, layer in enumerate([self.body1, self.body2, self.body3, self.body4]):
            xs = layer(xs)
            out.append(xs)
        return out

class NestedTensor(object):
    def __init__(self, tensors, mask: Optional[Tensor]):
        self.tensors = tensors
        self.mask = mask

    def to(self, device):
        # type: (Device) -> NestedTensor # noqa
        cast_tensor = self.tensors.to(device)
        mask = self.mask
        if mask is not None:
            assert mask is not None
            cast_mask = mask.to(device)
        else:
            cast_mask = None
        return NestedTensor(cast_tensor, cast_mask)

    def decompose(self):
        return self.tensors, self.mask

    def __repr__(self):
        return str(self.tensors)

class Model_builder(nn.Module):
    def __init__(self, cfg=None):
        super().__init__()

        assert cfg is None
        cfg = SimpleNamespace(
            TAG='APGCC_SHHA',
            SEED=1229,
            GPU_ID=0,
            OUTPUT_DIR='./output/',
            VIS=True,
            MODEL=SimpleNamespace(
                ENCODER='vgg16_bn',
                ENCODER_kwargs={"last_pool":False},
                DECODER='IFI',
                DECODER_kwargs={
                    "num_classes": 2,
                    "inner_planes": 64,
                    "feat_layers": [3, 4],
                    "pos_dim": 32,
                    "ultra_pe": True,
                    "learn_pe": False,
                    "unfold": False,
                    "local": True,
                    "no_aspp": False,
                    "require_grad": True,
                    "out_type": 'Normal',
                    "head_layers": [1024, 512, 256, 256]
                },
                STRIDE=8,
                ROW=2,
                LINE=2,
                FROZEN_WEIGHTS=None,
                POINT_LOSS_COEF=0.0002,
                EOS_COEF=0.5,
                LOSS=['L2'],
                WEIGHT_DICT={'loss_ce': 1, 'loss_points': 0.0002, 'loss_aux': 0.2},
                AUX_EN=True,
                AUX_NUMBER=[2, 2],
                AUX_RANGE=[2, 8],
                AUX_kwargs=SimpleNamespace(
                    pos_coef=1.0,
                    neg_coef=1.0,
                    pos_loc=0.0002,
                    neg_loc=0.0002
                )
            ),
            RESUME=False,
            RESUME_PATH='',
            DATASETS=SimpleNamespace(
                DATASET='SHHA',
                DATA_ROOT='/home/paiworker1/KX/.Workspace/.Projects/crowd_count/APGCC-infer/results/part_A'
            ),
            DATALOADER=SimpleNamespace(
                AUGUMENTATION=['Normalize', 'Crop', 'Flip'],
                CROP_SIZE=128,
                CROP_NUMBER=4,
                UPPER_BOUNDER=-1,
                NUM_WORKERS=0
            ),
            SOLVER=SimpleNamespace(
                BATCH_SIZE=8,
                START_EPOCH=0,
                EPOCHS=3500,
                LR=0.0001,
                LR_BACKBONE=1e-05,
                WEIGHT_DECAY=0.0001,
                LR_DROP=3500,
                CLIP_MAX_NORM=0.1,
                EVAL_FREQ=1,
                LOG_FREQ=1
            ),
            MATCHER=SimpleNamespace(
                SET_COST_CLASS=1.0,
                SET_COST_POINT=0.05
            ),
            TEST=SimpleNamespace(
                THRESHOLD=0.5,
                WEIGHT='./output/SHHA_best.pth'
            ),
            config_file='./configs/SHHA_test.yml',
            test=True
        )

        self.cfg = cfg
        self.num_classes = self.cfg.MODEL.DECODER_kwargs["num_classes"] # default:2 (person/background)
        self.num_anchor_points = cfg.MODEL.ROW * cfg.MODEL.LINE  # default:4
        self.encoder = self._build_encoder()
        self.decoder = self._build_decoder() 

    def _build_encoder(self, ):
        #########################################################################################
        # input: image, output: [feat1(H/2,W/2), feat2(H/4,W/4), feat3(H/8,W/8), feat4(H/16,W/16)]
        #########################################################################################
        if self.cfg.MODEL.ENCODER in ['vgg16', 'vgg16_bn']:
            pass
        elif self.cfg.MODEL.ENCODER in ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']:
            assert 0
        self.cfg.MODEL.ENCODER_kwargs['name'] = self.cfg.MODEL.ENCODER
        encoder = Base_VGG(**self.cfg.MODEL.ENCODER_kwargs)
        return encoder
    
    def _build_decoder(self, ): 
        assert self.cfg.MODEL.DECODER == 'IFI'    
        self.cfg.MODEL.DECODER_kwargs['in_planes'] = self.encoder.get_outplanes()
        self.cfg.MODEL.DECODER_kwargs['line'] = self.cfg.MODEL.LINE
        self.cfg.MODEL.DECODER_kwargs['row'] = self.cfg.MODEL.ROW
        self.cfg.MODEL.DECODER_kwargs['num_anchor_points'] = self.num_anchor_points
        self.cfg.MODEL.DECODER_kwargs['sync_bn'] = False
        self.cfg.MODEL.DECODER_kwargs['AUX_EN'] = self.cfg.MODEL.AUX_EN
        self.cfg.MODEL.DECODER_kwargs['AUX_NUMBER'] = self.cfg.MODEL.AUX_NUMBER
        self.cfg.MODEL.DECODER_kwargs['AUX_RANGE'] = self.cfg.MODEL.AUX_RANGE
        self.cfg.MODEL.DECODER_kwargs['AUX_kwargs'] = self.cfg.MODEL.AUX_kwargs
        decoder = IFI_Decoder_Model(**self.cfg.MODEL.DECODER_kwargs)
        return decoder

    def forward(self, samples: NestedTensor):
        features = self.encoder(samples)
        out = self.decoder(samples, features)       
        return out   # {'pred_logits', 'pred_points', 'offset'}
