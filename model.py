import torch
import torch.nn as nn
from torchvision.models import vgg16_bn, vgg11_bn
from abc import ABCMeta
from anchor import make_center_anchors
from utils import center_to_corner, find_jaccard_overlap, corner_to_center

import matplotlib.pyplot as plt
import matplotlib.patches as patches


class YOLO(nn.Module, metaclass=ABCMeta):
    def __init__(self, num_classes=20):
        self.anchors = [(1.3221, 1.73145), (3.19275, 4.00944), (5.05587, 8.09892), (9.47112, 4.84053),
                        (11.2364, 10.0071)]

        super().__init__()

        self.num_anchors = 5
        self.num_classes = num_classes

        self.extra = nn.Sequential(nn.Conv2d(512, 512, 3, 1, 1),
                                   nn.BatchNorm2d(512),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(512, 512, 3, 1, 1),
                                   nn.BatchNorm2d(512),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(512, 512, 3, 1, 1),
                                   nn.BatchNorm2d(512),
                                   nn.ReLU(inplace=True),
                                   nn.MaxPool2d(2, 2))

        self.skip_module = nn.Sequential(nn.Conv2d(512, 64, 1, stride=1, padding=0),
                                         nn.BatchNorm2d(64),
                                         nn.ReLU(inplace=True))

        self.final = nn.Sequential(nn.Conv2d(768, 1024, 3, padding=1),
                                   nn.BatchNorm2d(1024),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(1024, 256, 3, padding=1),
                                   nn.BatchNorm2d(256),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(256, self.num_anchors * (5 + self.num_classes), 1))  # anchor 5, class 20

        self.stu_feature_adap = nn.Sequential(nn.Conv2d(512, 512, kernel_size=3, padding=1),
                                              nn.ReLU())

        self.init_conv2d()

        # print("num_params : ", self.count_parameters())

    def _get_imitation_mask(self, x, gt_boxes, iou_factor=0.5):
        """
        gt_box: (B, K, 4) [x_min, y_min, x_max, y_max]
        """
        out_size = x.size(2)
        batch_size = x.size(0)

        center_anchors = make_center_anchors(anchors_wh=self.anchors, grid_size=out_size)
        anchors = center_to_corner(center_anchors).view(out_size * out_size * 5, 4)  # (N, 4)
        gt_boxes = gt_boxes

        mask_batch = torch.zeros([batch_size, out_size, out_size])

        for i in range(batch_size):
            num_obj = gt_boxes[i].size(0)
            if not num_obj:
                continue

            IOU_map = find_jaccard_overlap(anchors, gt_boxes[i] * float(out_size), 0).view(out_size, out_size, self.num_anchors, num_obj)
            max_iou, _ = IOU_map.view(-1, num_obj).max(dim=0)
            mask_img = torch.zeros([out_size, out_size], dtype=torch.int64, requires_grad=False).type_as(x)
            threshold: torch.Tensor = max_iou * iou_factor

            for k in range(num_obj):

                mask_per_gt = torch.sum(IOU_map[:, :, :, k] > threshold[k], dim=2)

                mask_img += mask_per_gt

                mask_img += mask_img
            mask_batch[i] = mask_img

        mask_batch = mask_batch.clamp(0, 1)
        return mask_batch  # (B, h, w)

    def init_conv2d(self):
        for c in self.extra.children():
            if isinstance(c, nn.Conv2d):
                nn.init.normal_(c.weight, mean=0.0, std=0.01)
                nn.init.constant_(c.bias, 0.)

        for c in self.skip_module.children():
            if isinstance(c, nn.Conv2d):
                nn.init.normal_(c.weight, mean=0.0, std=0.01)
                nn.init.constant_(c.bias, 0.)

        for c in self.final.children():
            if isinstance(c, nn.Conv2d):
                nn.init.normal_(c.weight, mean=0.0, std=0.01)
                nn.init.constant_(c.bias, 0.)

    def forward(self, x, target=None):
        output_size = x.size(-1)
        output_size /= 32
        o_size = int(output_size)

        x = self.middle_feature(x)  # after conv4_3, maxpooling
        features = x
        skip_x = self.skip_module(x)  # torch.Size([B, 512, 26, 26])--->  torch.Size([B, 64, 26, 26])

        # --------------------- yolo v2 reorg layer ---------------------
        skip_x = skip_x.view(-1, 64, o_size, 2, o_size, 2).contiguous()
        skip_x = skip_x.permute(0, 3, 5, 1, 2, 4).contiguous()
        skip_x = skip_x.view(-1, 256, o_size, o_size)  # torch.Size([B, 256, 13, 13])

        x = self.extra(x)  # torch.Size([B, 1024, 13, 13])
        x = torch.cat([x, skip_x], dim=1)  # torch.Size([B, 1280, 13, 13])
        x = self.final(x)  # torch.Size([B, 125, 13, 13])

        if target is not None:
            # if knowledge distillation is in process
            # https://arxiv.org/pdf/1906.03609.pdf
            return x, features, self._get_imitation_mask(features, target)
        return x

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class YOLO_VGG_16(YOLO):
    def __init__(self, num_classes=20):
        super().__init__(num_classes)
        self.middle_feature = nn.Sequential(*list(vgg16_bn(pretrained=True).features.children())[:-1])


class YOLO_VGG_11(YOLO):
    def __init__(self, num_classes=20):
        super().__init__(num_classes)
        self.middle_feature = nn.Sequential(*list(vgg11_bn(pretrained=True).features.children())[:-1])


if __name__ == '__main__':
    image = torch.randn([2, 3, 416, 416]).cuda()
    anchor = [torch.tensor([[0.3, 0.3, 0.6, 0.6], [0.1, 0.1, 0.4, 0.4]]).cuda(), torch.tensor([[0.1, 0.1, 0.4, 0.4]]).cuda()]
    #
    # model = YOLO_VGG_11().cuda()
    # print("num_params : ", model.count_parameters())
    # print(model(image).shape)

    model = YOLO_VGG_16().cuda()
    # print("num_params : ", model.count_parameters())
    # print(model(image).shape)

    with torch.no_grad():
        x, feature, mask = model(image, anchor)
    print(x.requires_grad)

    print(x.shape)
    print(feature.shape)
    print(mask.shape)

    plt.imshow(mask[1].squeeze().cpu(), cmap='gray')
    ax = plt.gca()
    for corner in anchor[1]:
        center = corner_to_center(corner)
        center *= feature.size(2)
        corner *= feature.size(2)
        rect = patches.Rectangle((corner[0], corner[1]), center[2], center[3], linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
    plt.show()
