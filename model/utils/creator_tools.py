import numpy as np
import torch
from torchvision.ops import nms
from model.utils.bbox_tools import bbox2loc, bbox_iou, loc2bbox


class AnchorTargetCreator(object):
    def __init__(self,
                 n_sample=256,
                 pos_iou_thresh=0.7, neg_iou_thresh=0.3,#征服框的阈值区间
                 pos_ratio=0.5):#0.5是正负样本的比例
        self.n_sample = n_sample
        self.pos_iou_thresh = pos_iou_thresh
        self.neg_iou_thresh = neg_iou_thresh
        self.pos_ratio = pos_ratio

    def __call__(self, bbox, anchor, img_size):
        img_H,img_W=img_size
        n_anchor = len(anchor)
        inside_index = self._get_inside_index(anchor, img_H, img_W)
        anchor = anchor[inside_index]
        argmax_ious, label = self._create_label(
        inside_index, anchor, bbox)

        loc = bbox2loc(anchor, bbox[argmax_ious])

        label = _unmap(label, n_anchor, inside_index, fill=-1)
        loc = _unmap(loc, n_anchor, inside_index, fill=0)

        return loc, label
    



    def _create_label(self,inside_index,anchor,bbox):
        label=np.empty((len(inside_index),),dtype=np.int32)
        label.fill(-1)
        argmax_ious, max_ious, gt_argmax_ious = \
            self._calc_ious(anchor, bbox, inside_index)
        label[max_ious<self.neg_iou_thresh]=0
        label[gt_argmax_ious]=1
        label[max_ious >= self.pos_iou_thresh] = 1
        n_pos=int(self.pos_ratio*self.n_sample)
        pos_index=np.where(label==1)[0]
        if len(pos_index) > n_pos:
            disable_index = np.random.choice(pos_index, size=(len(pos_index) - n_pos), replace=False)
            label[disable_index] = -1
            #只要把超过采样数据的阳性框标签设置为负即可
        return argmax_ious,label


        


    def _calc_ious(self,anchor,bbox,inside_index):
        ious=bbox_iou(anchor,bbox)
        argmax_ious=ious.argmax(axis=1)#axis=1表示按照行选择最大的值，应该就是标记，每一个myanchor和targetanchor的IOU更大 数据表现为 0 1 0 1 1 1 0 表示和哪个anchor更大
        max_ious = ious[np.arange(len(inside_index)), argmax_ious]#相当于是给了行索引，给了列索引，取出来的数据  行 1 3 5 7 9 ，列 0 1 1 0 1 0 1 这是取出来的最大的IOU数值 x.x 
        gt_argmax_ious=ious.argmax(axis=0)
        gt_max_ious = ious[gt_argmax_ious, np.arange(ious.shape[1])]#这里后面使用 ，：也可以，对于groudtruth最大的值，直接全取了
        gt_argmax_ious=np.where(ious==gt_max_ious)[0]#获得最大groundtruth iou的最大索引序号，获得和最大groundtruth IOU的 anchor序号
        return argmax_ious, max_ious, gt_argmax_ious


def _unmap(data, count, index, fill=0):
    # Unmap a subset of item (data) back to the original set of items (of
    # size count)

    if len(data.shape) == 1:
        ret = np.empty((count,), dtype=data.dtype)#(count,)和传入一个int型的一样，可能这样更规范化
        ret.fill(fill)
        ret[index] = data
    else:
        ret = np.empty((count,) + data.shape[1:], dtype=data.dtype)
        ret.fill(fill)
        ret[index, :] = data
    return ret

class ProposalTargetCreator(object):
    def __init__(self,
                 n_sample=128,
                 pos_ratio=0.25, pos_iou_thresh=0.5,
                 neg_iou_thresh_hi=0.5, neg_iou_thresh_lo=0.0
                 ):
        self.n_sample = n_sample
        self.pos_ratio = pos_ratio
        self.pos_iou_thresh = pos_iou_thresh
        self.neg_iou_thresh_hi = neg_iou_thresh_hi
        self.neg_iou_thresh_lo = neg_iou_thresh_lo  # NOTE:default 0.1 in py-faster-rcnn

    def __call__(self, roi, bbox, label,
                 loc_normalize_mean=(0., 0., 0., 0.),
                 loc_normalize_std=(0.1, 0.1, 0.2, 0.2)):


        n_bbox, _ = bbox.shape

        roi = np.concatenate((roi, bbox), axis=0)#这里读到的label是正常的label
        pos_roi_per_image = np.round(self.n_sample * self.pos_ratio)
        iou = bbox_iou(roi, bbox)
        gt_assignment = iou.argmax(axis=1)
        max_iou = iou.max(axis=1)
        # Offset range of classes from [0, n_fg_class - 1] to [1, n_fg_class].
        # The label with value 0 is the background.
        gt_roi_label = label[gt_assignment] + 1

        # Select foreground RoIs as those with >= pos_iou_thresh IoU.
        pos_index = np.where(max_iou >= self.pos_iou_thresh)[0]
        pos_roi_per_this_image = int(min(pos_roi_per_image, pos_index.size))#这里直接调用了size属性，直接返回一个长度
        #如果用shape的话，shape返回的是一个tuple,tuple还得取0，就很麻烦
        if pos_index.size > 0:
            pos_index = np.random.choice(
                pos_index, size=pos_roi_per_this_image, replace=False)

        # Select background RoIs as those within
        # [neg_iou_thresh_lo, neg_iou_thresh_hi).
        neg_index = np.where((max_iou < self.neg_iou_thresh_hi) &
                             (max_iou >= self.neg_iou_thresh_lo))[0]
        neg_roi_per_this_image = self.n_sample - pos_roi_per_this_image
        neg_roi_per_this_image = int(min(neg_roi_per_this_image,
                                         neg_index.size))
        if neg_index.size > 0:
            neg_index = np.random.choice(
                neg_index, size=neg_roi_per_this_image, replace=False)

        # The indices that we're selecting (both positive and negative).
        keep_index = np.append(pos_index, neg_index)
        gt_roi_label = gt_roi_label[keep_index]
        gt_roi_label[pos_roi_per_this_image:] = 0  # negative labels --> 0
        sample_roi = roi[keep_index]#最后要是正样本框不够了，就得牺牲一部分负样本框，尽可能的让正样本框多一点。

        # Compute offsets and scales to match sampled RoIs to the GTs.

        gt_roi_loc = bbox2loc(sample_roi, bbox[gt_assignment[keep_index]])
        gt_roi_loc = ((gt_roi_loc - np.array(loc_normalize_mean, np.float32)
                       ) / np.array(loc_normalize_std, np.float32))

        return sample_roi, gt_roi_loc, gt_roi_label

def _get_inside_index(anchor, H, W):
    # Calc indicies of anchors which are located completely inside of the image
    # whose size is speficied.
    index_inside = np.where(
        (anchor[:, 0] >= 0) &
        (anchor[:, 1] >= 0) &
        (anchor[:, 2] <= H) &
        (anchor[:, 3] <= W)
    )[0]
    return index_inside

#返回没有超出边界的边框
def _get_inside_index(anchor, H, W):
    # Calc indicies of anchors which are located completely inside of the image
    # whose size is speficied.
    index_inside = np.where(
        (anchor[:, 0] >= 0) &
        (anchor[:, 1] >= 0) &
        (anchor[:, 2] <= H) &
        (anchor[:, 3] <= W)
    )[0]
    return index_inside

class ProposalCreator:
    def __init__(self,
                 parent_model,
                 nms_thresh=0.7,
                 n_train_pre_nms=12000,
                 n_train_post_nms=2000,
                 n_test_pre_nms=6000,
                 n_test_post_nms=300,
                 min_size=16
                 ):
        self.parent_model = parent_model
        self.nms_thresh = nms_thresh
        self.n_train_pre_nms = n_train_pre_nms
        self.n_train_post_nms = n_train_post_nms
        self.n_test_pre_nms = n_test_pre_nms
        self.n_test_post_nms = n_test_post_nms
        self.min_size = min_size

        def __call__(self, loc, score,\
                 anchor, img_size, scale=1.):
            #太小的检测框就不要了
            if self.parent_model.training:
                n_pre_nms = self.n_train_pre_nms
                n_post_nms = self.n_train_post_nms
            else:
                n_pre_nms = self.n_test_pre_nms
                n_post_nms = self.n_test_post_nms
            

            roi = loc2bbox(anchor, loc)


            roi[:, slice(0, 4, 2)] = np.clip(
            roi[:, slice(0, 4, 2)], 0, img_size[0])
            roi[:, slice(1, 4, 2)] = np.clip(
            roi[:, slice(1, 4, 2)], 0, img_size[1])#将roi的区域限定在image图片的范围内
            #此处的坐标格式是 y1,y2,x1,x2类型的，clip将坐标狂限定在图的范围内
            min_size = self.min_size * scale
            hs = roi[:, 2] - roi[:, 0]
            ws = roi[:, 3] - roi[:, 1]
            keep = np.where((hs >= min_size) & (ws >= min_size))[0]
            roi = roi[keep, :]#获取大于最小标准框的检测框
            score = score[keep]
            order=score.ravel.argsort()[::-1]#argsort是从小到大排，这里需要反过来

            if n_pre_nms>0:
                order =order[:n_pre_nms]
            roi=roi[order,:]
            score=score[order]
            keep = nms(
            torch.from_numpy(roi).cuda(),
            torch.from_numpy(score).cuda(),
            self.nms_thresh)
            if n_post_nms > 0:
                keep = keep[:n_post_nms]
            roi = roi[keep.cpu().numpy()]
            return roi