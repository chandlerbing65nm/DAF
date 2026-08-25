import copy
import os.path as osp

from torch.utils.data import DataLoader

from mmcv.transforms.loading import LoadImageFromFile
from mmcv.transforms.processing import Resize

from mmengine.registry import TRANSFORMS
from mmengine.registry import DATASETS
import mmengine.fileio as fileio

from mmseg.datasets import BaseSegDataset
from mmseg.datasets.ade import ADE20KDataset as MMADE20KDataset
from mmseg.datasets.loveda import LoveDADataset as MMLoveDADataset
from mmseg.datasets.transforms.loading import LoadAnnotations
from mmseg.datasets.transforms.formatting import PackSegInputs

from utils import mm_transforms
from utils.misc import custom_collate, get_cls_idx


@DATASETS.register_module()
class CityscapesDataset(BaseSegDataset):
    """Cityscapes dataset.

    The ``img_suffix`` is fixed to '_leftImg8bit.png' and ``seg_map_suffix`` is
    fixed to '_gtFine_labelTrainIds.png' for Cityscapes dataset.
    """
    METAINFO = dict(
        classes=('road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
                 'traffic light', 'traffic sign', 'vegetation', 'terrain',
                 'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train',
                 'motorcycle', 'bicycle'),
        palette=[[128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
                 [190, 153, 153], [153, 153, 153], [250, 170,
                                                    30], [220, 220, 0],
                 [107, 142, 35], [152, 251, 152], [70, 130, 180],
                 [220, 20, 60], [255, 0, 0], [0, 0, 142], [0, 0, 70],
                 [0, 60, 100], [0, 80, 100], [0, 0, 230], [119, 11, 32]])
    
    class_extensions, extentions_to_real_class_idx = get_cls_idx("utils/class_extensions/cityscapes.txt")

    def __init__(self,
                 img_suffix='_leftImg8bit.png',
                 seg_map_suffix='_gtFine_labelTrainIds.png',
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix, seg_map_suffix=seg_map_suffix, **kwargs)


@DATASETS.register_module()
class CityscapesFoggyDataset(BaseSegDataset):
    """Cityscapes foggy dataset.

    Uses the same 19 Cityscapes evaluation classes and gtFine masks,
    but loads foggy images from ``leftImg8bit_foggy/val/`` with beta-suffix filenames.
    The ``img_suffix`` defaults to ``_leftImg8bit_foggy_beta_0.005.png`` (light)
    and is overridden by ``prepare_data`` based on the corruption value
    (light/medium/dense).
    """
    METAINFO = CityscapesDataset.METAINFO

    class_extensions, extentions_to_real_class_idx = get_cls_idx("utils/class_extensions/cityscapes.txt")

    def __init__(self,
                 img_suffix='_leftImg8bit_foggy_beta_0.005.png',
                 seg_map_suffix='_gtFine_labelTrainIds.png',
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix, seg_map_suffix=seg_map_suffix, **kwargs)


@DATASETS.register_module()
class BDD100kDataset(BaseSegDataset):
    METAINFO = CityscapesDataset.METAINFO

    class_extensions, extentions_to_real_class_idx = get_cls_idx("utils/class_extensions/cityscapes.txt")

    def __init__(self,
                 img_suffix='.jpg',
                 seg_map_suffix='_train_id.png',
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=False,
            **kwargs)


@DATASETS.register_module()
class ACDCDataset(BaseSegDataset):
    """ACDC dataset for adverse-condition driving segmentation.

    Uses the 19 Cityscapes evaluation classes and palette.
    Images are in ``rgb_anon/<condition>/val/<seq>/<frame>_rgb_anon.png``
    and masks in ``gt/<condition>/val/<seq>/<frame>_gt_labelTrainIds.png``.
    """
    METAINFO = CityscapesDataset.METAINFO

    class_extensions, extentions_to_real_class_idx = get_cls_idx("utils/class_extensions/cityscapes.txt")

    def __init__(self,
                 img_suffix='_rgb_anon.png',
                 seg_map_suffix='_gt_labelTrainIds.png',
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=False,
            **kwargs)

    def load_data_list(self) -> list:
        """Load annotation from ACDC's nested directory structure.

        Unlike the default ``BaseSegDataset.load_data_list``, this handles
        the mismatched basenames (``_rgb_anon`` vs ``_gt_labelTrainIds``)
        and the nested ``<seq>/`` subdirectories.
        """
        data_list = []
        img_dir = self.data_prefix.get('img_path', None)
        ann_dir = self.data_prefix.get('seg_map_path', None)
        _img_suffix_len = len(self.img_suffix)

        for img in fileio.list_dir_or_file(
                dir_path=img_dir,
                list_dir=False,
                suffix=self.img_suffix,
                recursive=True,
                backend_args=self.backend_args):
            # img is a relative path like "val/GOPR0476/GOPR0476_frame_000761_rgb_anon.png"
            # or "train/GOPR0476/..." — keep val only, skip train/test/ref.
            # When data_prefix already includes /val (e.g. rgb_anon/fog/val),
            # paths are "GOPR0476/..." with no split component.
            # When data_prefix is at condition level (e.g. rgb_anon/),
            # paths are "fog/val/GOPR0476/..." so we filter by component.
            parts = img.split('/')
            if any(p in ('train', 'test', 'ref') for p in parts):
                continue
            img_name = img[:-_img_suffix_len]  # "val/GOPR0476/GOPR0476_frame_000761"
            data_info = dict(img_path=osp.join(img_dir, img))
            if ann_dir is not None:
                # Replace _rgb_anon with _gt_labelTrainIds for the seg map
                seg_map = img_name + self.seg_map_suffix
                data_info['seg_map_path'] = osp.join(ann_dir, seg_map)
            data_info['label_map'] = self.label_map
            data_info['reduce_zero_label'] = self.reduce_zero_label
            data_info['seg_fields'] = []
            data_list.append(data_info)

        data_list = sorted(data_list, key=lambda x: x['img_path'])
        return data_list


@DATASETS.register_module()
class CarlaDataset(BaseSegDataset):
    """Carla dataset for driving semantic segmentation.

    Images live under ``<domain>/.../camera/*.png`` and segmentation masks
    under ``<domain>/.../segmentation/*.png`` (same filename).  The label
    information is stored in the red channel of the RGBA mask and remapped
    to 14 training IDs via ``LoadCarlaAnnotations``.
    """
    METAINFO = dict(
        classes=('road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
                 'traffic light', 'traffic sign', 'vegetation', 'terrain',
                 'sky', 'person', 'vehicle', 'road line'),
        palette=[[128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
                 [190, 153, 153], [153, 153, 153], [250, 170, 30],
                 [220, 220, 0], [107, 142, 35], [152, 251, 152],
                 [70, 130, 180], [220, 20, 60], [0, 0, 142], [0, 0, 90]])

    class_extensions, extentions_to_real_class_idx = get_cls_idx("utils/class_extensions/carla.txt")

    def __init__(self,
                 img_suffix='.png',
                 seg_map_suffix='.png',
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=False,
            **kwargs)

    def load_data_list(self) -> list:
        """Load annotation from Carla's nested camera/segmentation structure.

        Recursively scans the ``img_path`` directory for ``.png`` files and
        derives the corresponding seg map by replacing ``/camera/`` with
        ``/segmentation/`` in the path.
        """
        data_list = []
        img_dir = self.data_prefix.get('img_path', None)
        ann_dir = self.data_prefix.get('seg_map_path', None)

        for img in fileio.list_dir_or_file(
                dir_path=img_dir,
                list_dir=False,
                suffix=self.img_suffix,
                recursive=True,
                backend_args=self.backend_args):
            data_info = dict(img_path=osp.join(img_dir, img))
            if ann_dir is not None:
                # Replace /camera/ with /segmentation/ to find the seg map
                seg_rel = img.replace('/camera/', '/segmentation/')
                data_info['seg_map_path'] = osp.join(ann_dir, seg_rel)
            data_info['label_map'] = self.label_map
            data_info['reduce_zero_label'] = self.reduce_zero_label
            data_info['seg_fields'] = []
            data_list.append(data_info)

        data_list = sorted(data_list, key=lambda x: x['img_path'])
        return data_list


@DATASETS.register_module()
class ADE20kDataset(MMADE20KDataset):
    METAINFO = MMADE20KDataset.METAINFO

    class_extensions = None
    extentions_to_real_class_idx = None

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)


@DATASETS.register_module()
class DarkZurichDataset(BaseSegDataset):
    """Dark Zurich dataset for nighttime driving segmentation.

    Uses the 19 Cityscapes evaluation classes and palette.
    Images are in ``val/rgb_anon/val/<condition>/<seq>/<frame>_rgb_anon.png``
    and masks in ``val/gt/val/<condition>/<seq>/<frame>_gt_labelTrainIds.png``.
    """
    METAINFO = CityscapesDataset.METAINFO

    class_extensions, extentions_to_real_class_idx = get_cls_idx("utils/class_extensions/cityscapes.txt")

    def __init__(self,
                 img_suffix='_rgb_anon.png',
                 seg_map_suffix='_gt_labelTrainIds.png',
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=False,
            **kwargs)


@DATASETS.register_module()
class LoveDADataset(MMLoveDADataset):
    METAINFO = MMLoveDADataset.METAINFO

    class_extensions = None
    extentions_to_real_class_idx = None

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)


@DATASETS.register_module()
class COCOStuffDataset(BaseSegDataset):
    """COCO-Stuff dataset.

    In segmentation map annotation for COCO-Stuff, Train-IDs of the 10k version
    are from 1 to 171, where 0 is the ignore index, and Train-ID of COCO Stuff
    164k is from 0 to 170, where 255 is the ignore index. So, they are all 171
    semantic categories. ``reduce_zero_label`` is set to True and False for the
    10k and 164k versions, respectively. The ``img_suffix`` is fixed to '.jpg',
    and ``seg_map_suffix`` is fixed to '.png'.
    """
    METAINFO = dict(
        classes=(
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
            'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
            'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
            'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
            'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
            'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
            'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
            'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
            'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
            'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
            'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
            'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
            'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
            'scissors', 'teddy bear', 'hair drier', 'toothbrush', 'banner',
            'blanket', 'branch', 'bridge', 'building-other', 'bush', 'cabinet',
            'cage', 'cardboard', 'carpet', 'ceiling-other', 'ceiling-tile',
            'cloth', 'clothes', 'clouds', 'counter', 'cupboard', 'curtain',
            'desk-stuff', 'dirt', 'door-stuff', 'fence', 'floor-marble',
            'floor-other', 'floor-stone', 'floor-tile', 'floor-wood', 'flower',
            'fog', 'food-other', 'fruit', 'furniture-other', 'grass', 'gravel',
            'ground-other', 'hill', 'house', 'leaves', 'light', 'mat', 'metal',
            'mirror-stuff', 'moss', 'mountain', 'mud', 'napkin', 'net',
            'paper', 'pavement', 'pillow', 'plant-other', 'plastic',
            'platform', 'playingfield', 'railing', 'railroad', 'river', 'road',
            'rock', 'roof', 'rug', 'salad', 'sand', 'sea', 'shelf',
            'sky-other', 'skyscraper', 'snow', 'solid-other', 'stairs',
            'stone', 'straw', 'structural-other', 'table', 'tent',
            'textile-other', 'towel', 'tree', 'vegetable', 'wall-brick',
            'wall-concrete', 'wall-other', 'wall-panel', 'wall-stone',
            'wall-tile', 'wall-wood', 'water-other', 'waterdrops',
            'window-blind', 'window-other', 'wood'),
        palette=[[0, 192, 64], [0, 192, 64], [0, 64, 96], [128, 192, 192],
                 [0, 64, 64], [0, 192, 224], [0, 192, 192], [128, 192, 64],
                 [0, 192, 96], [128, 192, 64], [128, 32, 192], [0, 0, 224],
                 [0, 0, 64], [0, 160, 192], [128, 0, 96], [128, 0, 192],
                 [0, 32, 192], [128, 128, 224], [0, 0, 192], [128, 160, 192],
                 [128, 128, 0], [128, 0, 32], [128, 32, 0], [128, 0, 128],
                 [64, 128, 32], [0, 160, 0], [0, 0, 0], [192, 128, 160],
                 [0, 32, 0], [0, 128, 128], [64, 128, 160], [128, 160, 0],
                 [0, 128, 0], [192, 128, 32], [128, 96, 128], [0, 0, 128],
                 [64, 0, 32], [0, 224, 128], [128, 0, 0], [192, 0, 160],
                 [0, 96, 128], [128, 128, 128], [64, 0, 160], [128, 224, 128],
                 [128, 128, 64], [192, 0, 32], [128, 96, 0], [128, 0, 192],
                 [0, 128, 32], [64, 224, 0], [0, 0, 64], [128, 128, 160],
                 [64, 96, 0], [0, 128, 192], [0, 128, 160], [192, 224, 0],
                 [0, 128, 64], [128, 128, 32], [192, 32, 128], [0, 64, 192],
                 [0, 0, 32], [64, 160, 128], [128, 64, 64], [128, 0, 160],
                 [64, 32, 128], [128, 192, 192], [0, 0, 160], [192, 160, 128],
                 [128, 192, 0], [128, 0, 96], [192, 32, 0], [128, 64, 128],
                 [64, 128, 96], [64, 160, 0], [0, 64, 0], [192, 128, 224],
                 [64, 32, 0], [0, 192, 128], [64, 128, 224], [192, 160, 0],
                 [0, 192, 0], [192, 128, 96], [192, 96, 128], [0, 64, 128],
                 [64, 0, 96], [64, 224, 128], [128, 64, 0], [192, 0, 224],
                 [64, 96, 128], [128, 192, 128], [64, 0, 224], [192, 224, 128],
                 [128, 192, 64], [192, 0, 96], [192, 96, 0], [128, 64, 192],
                 [0, 128, 96], [0, 224, 0], [64, 64, 64], [128, 128, 224],
                 [0, 96, 0], [64, 192, 192], [0, 128, 224], [128, 224, 0],
                 [64, 192, 64], [128, 128, 96], [128, 32, 128], [64, 0, 192],
                 [0, 64, 96], [0, 160, 128], [192, 0, 64], [128, 64, 224],
                 [0, 32, 128], [192, 128, 192], [0, 64, 224], [128, 160, 128],
                 [192, 128, 0], [128, 64, 32], [128, 32, 64], [192, 0, 128],
                 [64, 192, 32], [0, 160, 64], [64, 0, 0], [192, 192, 160],
                 [0, 32, 64], [64, 128, 128], [64, 192, 160], [128, 160, 64],
                 [64, 128, 0], [192, 192, 32], [128, 96, 192], [64, 0, 128],
                 [64, 64, 32], [0, 224, 192], [192, 0, 0], [192, 64, 160],
                 [0, 96, 192], [192, 128, 128], [64, 64, 160], [128, 224, 192],
                 [192, 128, 64], [192, 64, 32], [128, 96, 64], [192, 0, 192],
                 [0, 192, 32], [64, 224, 64], [64, 0, 64], [128, 192, 160],
                 [64, 96, 64], [64, 128, 192], [0, 192, 160], [192, 224, 64],
                 [64, 128, 64], [128, 192, 32], [192, 32, 192], [64, 64, 192],
                 [0, 64, 32], [64, 160, 192], [192, 64, 64], [128, 64, 160],
                 [64, 32, 192], [192, 192, 192], [0, 64, 160], [192, 160, 192],
                 [192, 192, 0], [128, 64, 96], [192, 32, 64], [192, 64, 128],
                 [64, 192, 96], [64, 160, 64], [64, 64, 0]])
    
    class_extensions, extentions_to_real_class_idx = get_cls_idx("utils/class_extensions/coco_stuff.txt")


    def __init__(self,
                 img_suffix='.jpg',
                 seg_map_suffix='_labelTrainIds.png',
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix, seg_map_suffix=seg_map_suffix, **kwargs)


@DATASETS.register_module()
class COCOStuff10kDataset(COCOStuffDataset):
    """COCO-Stuff10k dataset using the test split file list."""

    METAINFO = COCOStuffDataset.METAINFO
    class_extensions = COCOStuffDataset.class_extensions
    extentions_to_real_class_idx = COCOStuffDataset.extentions_to_real_class_idx

    def __init__(self,
                 ann_file,
                 img_suffix='.jpg',
                 seg_map_suffix='_labelTrainIds.png',
                 reduce_zero_label=True,
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            ann_file=ann_file,
            reduce_zero_label=reduce_zero_label,
            **kwargs)
        assert fileio.exists(self.data_prefix['img_path'],
                             self.backend_args) and osp.isfile(self.ann_file)


@DATASETS.register_module()
class PascalVOC21Dataset(BaseSegDataset):
    """Pascal VOC dataset.

    Args:
        split (str): Split txt file for Pascal VOC.
    """
    METAINFO = dict(
        classes=('background', 'aeroplane', 'bicycle', 'bird', 'boat',
                 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
                 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep',
                 'sofa', 'train', 'tvmonitor'),
        palette=[[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                 [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                 [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
                 [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
                 [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
                 [0, 64, 128]])

    class_extensions, extentions_to_real_class_idx = get_cls_idx("utils/class_extensions/voc21.txt")

    def __init__(self,
                 ann_file,
                 img_suffix='.jpg',
                 seg_map_suffix='.png',
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            ann_file=ann_file,
            **kwargs)
        assert fileio.exists(self.data_prefix['img_path'],
                             self.backend_args) and osp.isfile(self.ann_file)


@DATASETS.register_module()
class COCOObjectDataset(BaseSegDataset):
    """
    Implementation borrowed from TCL (https://github.com/kakaobrain/tcl) and GroupViT (https://github.com/NVlabs/GroupViT)
    COCO-Object dataset.
    1 bg class + first 80 classes from the COCO-Stuff dataset.
    """

    METAINFO = dict(
        classes=('background', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
                 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
                 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
                 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
                 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
                 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut',
                 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
                 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book',
                 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'),
        palette=[[0, 0, 0], [0, 192, 64], [0, 192, 64], [0, 64, 96], [128, 192, 192], [0, 64, 64], [0, 192, 224],
                 [0, 192, 192], [128, 192, 64], [0, 192, 96], [128, 192, 64], [128, 32, 192], [0, 0, 224], [0, 0, 64],
                 [0, 160, 192], [128, 0, 96], [128, 0, 192], [0, 32, 192], [128, 128, 224], [0, 0, 192],
                 [128, 160, 192],
                 [128, 128, 0], [128, 0, 32], [128, 32, 0], [128, 0, 128], [64, 128, 32], [0, 160, 0], [0, 0, 0],
                 [192, 128, 160], [0, 32, 0], [0, 128, 128], [64, 128, 160], [128, 160, 0], [0, 128, 0], [192, 128, 32],
                 [128, 96, 128], [0, 0, 128], [64, 0, 32], [0, 224, 128], [128, 0, 0], [192, 0, 160], [0, 96, 128],
                 [128, 128, 128], [64, 0, 160], [128, 224, 128], [128, 128, 64], [192, 0, 32],
                 [128, 96, 0], [128, 0, 192], [0, 128, 32], [64, 224, 0], [0, 0, 64], [128, 128, 160], [64, 96, 0],
                 [0, 128, 192], [0, 128, 160], [192, 224, 0], [0, 128, 64], [128, 128, 32], [192, 32, 128],
                 [0, 64, 192],
                 [0, 0, 32], [64, 160, 128], [128, 64, 64], [128, 0, 160], [64, 32, 128], [128, 192, 192], [0, 0, 160],
                 [192, 160, 128], [128, 192, 0], [128, 0, 96], [192, 32, 0], [128, 64, 128], [64, 128, 96],
                 [64, 160, 0],
                 [0, 64, 0], [192, 128, 224], [64, 32, 0], [0, 192, 128], [64, 128, 224], [192, 160, 0]])

    class_extensions, extentions_to_real_class_idx = get_cls_idx("utils/class_extensions/coco_object.txt")


    def __init__(self, **kwargs):
        super(COCOObjectDataset, self).__init__(img_suffix='.jpg', seg_map_suffix='_instanceTrainIds.png', **kwargs)


@DATASETS.register_module()
class PascalVOC20Dataset(BaseSegDataset):
    """Pascal VOC dataset.

    Args:
        split (str): Split txt file for Pascal VOC.
    """
    METAINFO = dict(
        classes=('aeroplane', 'bicycle', 'bird', 'boat',
                 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
                 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep',
                 'sofa', 'train', 'tvmonitor'),
        palette=[[128, 0, 0], [0, 128, 0], [0, 0, 192],
                 [128, 128, 0], [128, 0, 128], [0, 128, 128], [192, 128, 64],
                 [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
                 [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
                 [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
                 [0, 64, 128]])

    class_extensions, extentions_to_real_class_idx = get_cls_idx("utils/class_extensions/voc20.txt")


    def __init__(self,
                 ann_file,
                 img_suffix='.jpg',
                 seg_map_suffix='.png',
                 reduce_zero_label=True,
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=reduce_zero_label,
            ann_file=ann_file,
            **kwargs)
        assert fileio.exists(self.data_prefix['img_path'],
                             self.backend_args) and osp.isfile(self.ann_file)


@DATASETS.register_module()
class PascalContext60Dataset(BaseSegDataset):
    METAINFO = dict(
        classes=('background', 'aeroplane', 'bag', 'bed', 'bedclothes',
                 'bench', 'bicycle', 'bird', 'boat', 'book', 'bottle',
                 'building', 'bus', 'cabinet', 'car', 'cat', 'ceiling',
                 'chair', 'cloth', 'computer', 'cow', 'cup', 'curtain', 'dog',
                 'door', 'fence', 'floor', 'flower', 'food', 'grass', 'ground',
                 'horse', 'keyboard', 'light', 'motorbike', 'mountain',
                 'mouse', 'person', 'plate', 'platform', 'pottedplant', 'road',
                 'rock', 'sheep', 'shelves', 'sidewalk', 'sign', 'sky', 'snow',
                 'sofa', 'table', 'track', 'train', 'tree', 'truck',
                 'tvmonitor', 'wall', 'water', 'window', 'wood'),
        palette=[[120, 120, 120], [180, 120, 120], [6, 230, 230], [80, 50, 50],
                 [4, 200, 3], [120, 120, 80], [140, 140, 140], [204, 5, 255],
                 [230, 230, 230], [4, 250, 7], [224, 5, 255], [235, 255, 7],
                 [150, 5, 61], [120, 120, 70], [8, 255, 51], [255, 6, 82],
                 [143, 255, 140], [204, 255, 4], [255, 51, 7], [204, 70, 3],
                 [0, 102, 200], [61, 230, 250], [255, 6, 51], [11, 102, 255],
                 [255, 7, 71], [255, 9, 224], [9, 7, 230], [220, 220, 220],
                 [255, 9, 92], [112, 9, 255], [8, 255, 214], [7, 255, 224],
                 [255, 184, 6], [10, 255, 71], [255, 41, 10], [7, 255, 255],
                 [224, 255, 8], [102, 8, 255], [255, 61, 6], [255, 194, 7],
                 [255, 122, 8], [0, 255, 20], [255, 8, 41], [255, 5, 153],
                 [6, 51, 255], [235, 12, 255], [160, 150, 20], [0, 163, 255],
                 [140, 140, 140], [250, 10, 15], [20, 255, 0], [31, 255, 0],
                 [255, 31, 0], [255, 224, 0], [153, 255, 0], [0, 0, 255],
                 [255, 71, 0], [0, 235, 255], [0, 173, 255], [31, 0, 255]])

    class_extensions, extentions_to_real_class_idx = get_cls_idx("utils/class_extensions/context60.txt")

    def __init__(self,
                 ann_file: str,
                 img_suffix='.jpg',
                 seg_map_suffix='.png',
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            ann_file=ann_file,
            reduce_zero_label=False,
            **kwargs)


@DATASETS.register_module()
class PascalContext59Dataset(BaseSegDataset):
    METAINFO = dict(
        classes=('aeroplane', 'bag', 'bed', 'bedclothes', 'bench', 'bicycle',
                 'bird', 'boat', 'book', 'bottle', 'building', 'bus',
                 'cabinet', 'car', 'cat', 'ceiling', 'chair', 'cloth',
                 'computer', 'cow', 'cup', 'curtain', 'dog', 'door', 'fence',
                 'floor', 'flower', 'food', 'grass', 'ground', 'horse',
                 'keyboard', 'light', 'motorbike', 'mountain', 'mouse',
                 'person', 'plate', 'platform', 'pottedplant', 'road', 'rock',
                 'sheep', 'shelves', 'sidewalk', 'sign', 'sky', 'snow', 'sofa',
                 'table', 'track', 'train', 'tree', 'truck', 'tvmonitor',
                 'wall', 'water', 'window', 'wood'),
        palette=[[180, 120, 120], [6, 230, 230], [80, 50, 50], [4, 200, 3],
                 [120, 120, 80], [140, 140, 140], [204, 5, 255],
                 [230, 230, 230], [4, 250, 7], [224, 5, 255], [235, 255, 7],
                 [150, 5, 61], [120, 120, 70], [8, 255, 51], [255, 6, 82],
                 [143, 255, 140], [204, 255, 4], [255, 51, 7], [204, 70, 3],
                 [0, 102, 200], [61, 230, 250], [255, 6, 51], [11, 102, 255],
                 [255, 7, 71], [255, 9, 224], [9, 7, 230], [220, 220, 220],
                 [255, 9, 92], [112, 9, 255], [8, 255, 214], [7, 255, 224],
                 [255, 184, 6], [10, 255, 71], [255, 41, 10], [7, 255, 255],
                 [224, 255, 8], [102, 8, 255], [255, 61, 6], [255, 194, 7],
                 [255, 122, 8], [0, 255, 20], [255, 8, 41], [255, 5, 153],
                 [6, 51, 255], [235, 12, 255], [160, 150, 20], [0, 163, 255],
                 [140, 140, 140], [250, 10, 15], [20, 255, 0], [31, 255, 0],
                 [255, 31, 0], [255, 224, 0], [153, 255, 0], [0, 0, 255],
                 [255, 71, 0], [0, 235, 255], [0, 173, 255], [31, 0, 255]])

    class_extensions, extentions_to_real_class_idx = get_cls_idx("utils/class_extensions/context59.txt")

    def __init__(self,
                 ann_file: str,
                 img_suffix='.jpg',
                 seg_map_suffix='.png',
                 reduce_zero_label=True,
                 **kwargs):
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            ann_file=ann_file,
            reduce_zero_label=reduce_zero_label,
            **kwargs)


@DATASETS.register_module()
class SUIM6Dataset(BaseSegDataset):
    METAINFO = dict(
        classes=(
            'background',
            'human_divers',
            'wrecks_ruins',
            'robots_instruments',
            'reefs_invertebrates',
            'fish_vertebrates',
        ),
        palette=[[0, 0, 0], [0, 0, 255], [0, 255, 255], [255, 0, 0], [255, 0, 255], [255, 255, 0]],
    )

    class_extensions = None
    extentions_to_real_class_idx = None

    def __init__(self,
                 img_suffix='.jpg',
                 seg_map_suffix='.bmp',
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=False,
            **kwargs)


@DATASETS.register_module()
class DUTUSEG5Dataset(BaseSegDataset):
    METAINFO = dict(
        classes=(
            'background',
            'sea_cucumber',
            'sea_urchin',
            'scallop',
            'starfish',
        ),
        palette=[[0, 0, 0], [255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0]],
    )

    class_extensions = None
    extentions_to_real_class_idx = None

    def __init__(self,
                 ann_file,
                 img_suffix='.jpg',
                 seg_map_suffix='.png',
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            ann_file=ann_file,
            reduce_zero_label=False,
            **kwargs)


@DATASETS.register_module()
class DUTUSEG4Dataset(BaseSegDataset):
    METAINFO = dict(
        classes=(
            'sea_cucumber',
            'sea_urchin',
            'scallop',
            'starfish',
        ),
        palette=[[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0]],
    )

    class_extensions = None
    extentions_to_real_class_idx = None

    def __init__(self,
                 ann_file,
                 img_suffix='.jpg',
                 seg_map_suffix='.png',
                 reduce_zero_label=True,
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            ann_file=ann_file,
            reduce_zero_label=reduce_zero_label,
            **kwargs)


@DATASETS.register_module()
class SUIM5Dataset(BaseSegDataset):
    METAINFO = dict(
        classes=(
            'human_divers',
            'wrecks_ruins',
            'robots_instruments',
            'reefs_invertebrates',
            'fish_vertebrates',
        ),
        palette=[[0, 0, 255], [0, 255, 255], [255, 0, 0], [255, 0, 255], [255, 255, 0]],
    )

    class_extensions = None
    extentions_to_real_class_idx = None

    def __init__(self,
                 img_suffix='.jpg',
                 seg_map_suffix='.bmp',
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=False,
            **kwargs)



CLIP_MEAN = [122.7709, 116.7460, 104.0937]
CLIP_STD  = [68.5005, 66.6322, 70.3232]


# Register the modules with TRANSFORMS
TRANSFORMS.register_module(module=LoadImageFromFile, force=True)
TRANSFORMS.register_module(module=Resize, force=True)
TRANSFORMS.register_module(module=LoadAnnotations, force=True)
TRANSFORMS.register_module(module=PackSegInputs, force=True)


### make them args in the main.py and pass them to the file
data_dir  = ''
batch_size = 2 # number of loaded images
resize = (224, 224) # the size of the image after resizing => it can be vertical or horizontal depending on the image so => (560, 448) or (448, 560)
patch_size = (224, 224) # the size of the patch that will be extracted from the resized image
patch_stride = 112 # the stride of the patch extraction



mm_cocostuff_cfg =  { 
    'type': 'COCOStuffDataset', 
    'data_root': data_dir,  
    'data_prefix': {'img_path': 'images/val2017', 'seg_map_path': 'annotations/val2017'}, 
    'pipeline': [{'type': 'LoadImageFromFile'},
                {'type': 'LoadAnnotations'},
                {'type': 'ResizeAndPatchify', 'resize': resize, 'patch_size': patch_size, 'patch_stride': patch_stride},
                {'type': 'ToTensorAndNormalize', 'mean': CLIP_MEAN, 'std': CLIP_STD},
                ]
    }

mm_cocostuff10k_cfg =  {
    'type': 'COCOStuff10kDataset',
    'data_root': data_dir,
    'data_prefix': {'img_path': 'images/test2014', 'seg_map_path': 'annotations/test2014'},
    'ann_file': 'imageLists/test.txt',
    'pipeline': [{'type': 'LoadImageFromFile'},
                {'type': 'LoadAnnotations', 'reduce_zero_label': True},
                {'type': 'ResizeAndPatchify', 'resize': resize, 'patch_size': patch_size, 'patch_stride': patch_stride},
                {'type': 'ToTensorAndNormalize', 'mean': CLIP_MEAN, 'std': CLIP_STD},
                ]
}

mm_cocoobject_cfg =  { 
    'type': 'COCOObjectDataset', 
    'data_root': data_dir,  
    'data_prefix': {'img_path': 'images/val2017', 'seg_map_path': 'annotations/val2017'}, 
    'pipeline': [{'type': 'LoadImageFromFile'},
                {'type': 'LoadAnnotations'},
                {'type': 'ResizeAndPatchify', 'resize': resize, 'patch_size': patch_size, 'patch_stride': patch_stride},
                {'type': 'ToTensorAndNormalize', 'mean': CLIP_MEAN, 'std': CLIP_STD},
                ]
    }

mm_cityscapes_cfg =  {
    'type': 'CityscapesDataset',
    'data_root': data_dir,
    'data_prefix': {'img_path': 'leftImg8bit/val', 'seg_map_path': 'gtFine/val'},
    'pipeline': [{'type': 'LoadImageFromFile'},
                {'type': 'LoadAnnotations'},
                {'type': 'ResizeAndPatchify', 'resize': resize, 'patch_size': patch_size, 'patch_stride': patch_stride},
                {'type': 'ToTensorAndNormalize', 'mean': CLIP_MEAN, 'std': CLIP_STD},
                ]
    }

mm_cityscapes_foggy_cfg =  {
    'type': 'CityscapesFoggyDataset',
    'data_root': data_dir,
    'data_prefix': {'img_path': 'leftImg8bit_foggy/val', 'seg_map_path': 'gtFine/val'},
    'pipeline': [{'type': 'LoadImageFromFile'},
                {'type': 'LoadAnnotations'},
                {'type': 'ResizeAndPatchify', 'resize': resize, 'patch_size': patch_size, 'patch_stride': patch_stride},
                {'type': 'ToTensorAndNormalize', 'mean': CLIP_MEAN, 'std': CLIP_STD},
                ]
    }

mm_bdd100k_cfg =  {
    'type': 'BDD100kDataset',
    'data_root': data_dir,
    'data_prefix': {'img_path': 'images/10k/val', 'seg_map_path': 'labels/sem_seg/masks/val'},
    'pipeline': [{'type': 'LoadImageFromFile'},
                {'type': 'LoadAnnotations'},
                {'type': 'ResizeAndPatchify', 'resize': resize, 'patch_size': patch_size, 'patch_stride': patch_stride},
                {'type': 'ToTensorAndNormalize', 'mean': CLIP_MEAN, 'std': CLIP_STD},
                ]
    }

mm_acdc_cfg = {
    'type': 'ACDCDataset',
    'data_root': data_dir,
    'data_prefix': {'img_path': 'rgb_anon/fog/val', 'seg_map_path': 'gt/fog/val'},
    'pipeline': [{'type': 'LoadImageFromFile'},
                {'type': 'LoadAnnotations'},
                {'type': 'ResizeAndPatchify', 'resize': resize, 'patch_size': patch_size, 'patch_stride': patch_stride},
                {'type': 'ToTensorAndNormalize', 'mean': CLIP_MEAN, 'std': CLIP_STD},
                ]
    }

mm_dark_zurich_cfg = {
    'type': 'DarkZurichDataset',
    'data_root': data_dir,
    'data_prefix': {'img_path': 'val/rgb_anon/val', 'seg_map_path': 'val/gt/val'},
    'pipeline': [{'type': 'LoadImageFromFile'},
                {'type': 'LoadAnnotations'},
                {'type': 'ResizeAndPatchify', 'resize': resize, 'patch_size': patch_size, 'patch_stride': patch_stride},
                {'type': 'ToTensorAndNormalize', 'mean': CLIP_MEAN, 'std': CLIP_STD},
                ]
    }

mm_carla_cfg = {
    'type': 'CarlaDataset',
    'data_root': data_dir,
    'data_prefix': {'img_path': 'clear_fog_1200', 'seg_map_path': 'clear_fog_1200'},
    'pipeline': [{'type': 'LoadImageFromFile'},
                {'type': 'LoadCarlaAnnotations'},
                {'type': 'ResizeAndPatchify', 'resize': resize, 'patch_size': patch_size, 'patch_stride': patch_stride},
                {'type': 'ToTensorAndNormalize', 'mean': CLIP_MEAN, 'std': CLIP_STD},
                ]
    }

mm_ade20k_cfg = {
    'type': 'ADE20kDataset',
    'data_root': data_dir,
    'data_prefix': {'img_path': 'images/validation', 'seg_map_path': 'annotations/validation'},
    'pipeline': [{'type': 'LoadImageFromFile'},
                {'type': 'LoadAnnotations'},
                {'type': 'ResizeAndPatchify', 'resize': resize, 'patch_size': patch_size, 'patch_stride': patch_stride},
                {'type': 'ToTensorAndNormalize', 'mean': CLIP_MEAN, 'std': CLIP_STD},
                ]
}

mm_loveda_cfg = {
    'type': 'LoveDADataset',
    'data_root': data_dir,
    'data_prefix': {'img_path': 'img_dir/val', 'seg_map_path': 'ann_dir/val'},
    'pipeline': [{'type': 'LoadImageFromFile'},
                {'type': 'LoadAnnotations', 'reduce_zero_label': True},
                {'type': 'ResizeAndPatchify', 'resize': resize, 'patch_size': patch_size, 'patch_stride': patch_stride},
                {'type': 'ToTensorAndNormalize', 'mean': CLIP_MEAN, 'std': CLIP_STD},
                ]
}

mm_pascalvoc20_cfg = {
    'type': 'PascalVOC20Dataset',
    'data_root': data_dir,
    'data_prefix': {'img_path': 'JPEGImages', 'seg_map_path': 'SegmentationClass'},
    'ann_file': 'ImageSets/Segmentation/val.txt',
    'pipeline': [{'type': 'LoadImageFromFile'},
                {'type': 'LoadAnnotations'},
                {'type': 'ResizeAndPatchify', 'resize': resize, 'patch_size': patch_size, 'patch_stride': patch_stride},
                {'type': 'ToTensorAndNormalize', 'mean': CLIP_MEAN, 'std': CLIP_STD},
                ]
}

mm_pascalvoc21_cfg = {
    'type': 'PascalVOC21Dataset',
    'data_root': data_dir,
    'data_prefix': {'img_path': 'JPEGImages', 'seg_map_path': 'SegmentationClass'},
    'ann_file': 'ImageSets/Segmentation/val.txt',
    'pipeline': [{'type': 'LoadImageFromFile'},
                {'type': 'LoadAnnotations'},
                {'type': 'ResizeAndPatchify', 'resize': resize, 'patch_size': patch_size, 'patch_stride': patch_stride},
                {'type': 'ToTensorAndNormalize', 'mean': CLIP_MEAN, 'std': CLIP_STD},
                ]
}    


mm_pascalcontect59_cfg = {
    'type': 'PascalContext59Dataset',
    'data_root': data_dir,
    'data_prefix': {'img_path': 'JPEGImages', 'seg_map_path': 'SegmentationClassContext'},
    'ann_file': 'ImageSets/SegmentationContext/val.txt',
    'pipeline': [{'type': 'LoadImageFromFile'},
                {'type': 'LoadAnnotations', 'reduce_zero_label':True},
                {'type': 'ResizeAndPatchify', 'resize': resize, 'patch_size': patch_size, 'patch_stride': patch_stride},
                {'type': 'ToTensorAndNormalize', 'mean': CLIP_MEAN, 'std': CLIP_STD},
                ]
}    


mm_pascalcontect60_cfg = {
    'type': 'PascalContext60Dataset',
    'data_root': data_dir,
    'data_prefix': {'img_path': 'JPEGImages', 'seg_map_path': 'SegmentationClassContext'},
    'ann_file': 'ImageSets/SegmentationContext/val.txt',
    'pipeline': [{'type': 'LoadImageFromFile'},
                {'type': 'LoadAnnotations'},
                {'type': 'ResizeAndPatchify', 'resize': resize, 'patch_size': patch_size, 'patch_stride': patch_stride},
                {'type': 'ToTensorAndNormalize', 'mean': CLIP_MEAN, 'std': CLIP_STD},
                ]
}    


mm_suim6_cfg = {
    'type': 'SUIM6Dataset',
    'data_root': data_dir,
    'data_prefix': {'img_path': 'test/images', 'seg_map_path': 'test/masks'},
    'pipeline': [{'type': 'LoadImageFromFile'},
                {'type': 'LoadSUIMAnnotations'},
                {'type': 'ResizeAndPatchify', 'resize': resize, 'patch_size': patch_size, 'patch_stride': patch_stride},
                {'type': 'ToTensorAndNormalize', 'mean': CLIP_MEAN, 'std': CLIP_STD},
                ]
}


mm_suim5_cfg = {
    'type': 'SUIM5Dataset',
    'data_root': data_dir,
    'data_prefix': {'img_path': 'test/images', 'seg_map_path': 'test/masks'},
    'pipeline': [{'type': 'LoadImageFromFile'},
                {'type': 'LoadSUIMAnnotations', 'drop_background': True},
                {'type': 'ResizeAndPatchify', 'resize': resize, 'patch_size': patch_size, 'patch_stride': patch_stride},
                {'type': 'ToTensorAndNormalize', 'mean': CLIP_MEAN, 'std': CLIP_STD},
                ]
}


mm_dutuseg5_cfg = {
    'type': 'DUTUSEG5Dataset',
    'data_root': data_dir,
    'data_prefix': {'img_path': 'JPEGImages', 'seg_map_path': 'SegmentationClass'},
    'ann_file': 'ImageSets/val.txt',
    'pipeline': [{'type': 'LoadImageFromFile'},
                {'type': 'LoadAnnotations'},
                {'type': 'ResizeAndPatchify', 'resize': resize, 'patch_size': patch_size, 'patch_stride': patch_stride},
                {'type': 'ToTensorAndNormalize', 'mean': CLIP_MEAN, 'std': CLIP_STD},
                ]
}


mm_dutuseg4_cfg = {
    'type': 'DUTUSEG4Dataset',
    'data_root': data_dir,
    'data_prefix': {'img_path': 'JPEGImages', 'seg_map_path': 'SegmentationClass'},
    'ann_file': 'ImageSets/val.txt',
    'pipeline': [{'type': 'LoadImageFromFile'},
                {'type': 'LoadAnnotations', 'reduce_zero_label': True},
                {'type': 'ResizeAndPatchify', 'resize': resize, 'patch_size': patch_size, 'patch_stride': patch_stride},
                {'type': 'ToTensorAndNormalize', 'mean': CLIP_MEAN, 'std': CLIP_STD},
                ]
}




def prepare_data(dataset, data_dir, init_resize, patch_size, patch_stride, corruption="original", batch_size=128, num_workers=1, shuffle=True):
    
    # # print everything
    # print("\n+++++++ Data Preparation +++++++")
    # print(f"Dataset:           {dataset}")
    # print(f"Data directory:    {data_dir}")
    # print(f"Initial resize:    {init_resize}")
    # print(f"Patch size:        {patch_size}")
    # print(f"Patch stride:      {patch_stride}")
    # print(f"Corruption:        {corruption}")
    # print(f"Batch size:        {batch_size}")
    # print(f"Number of workers: {num_workers}")
    # print("----------------------------------------")


    if init_resize is None:
        assert batch_size == 1, "Batch size must be 1 if init_resize is None"

    if dataset == "COCOStuffDataset":
        mm_config = copy.deepcopy(mm_cocostuff_cfg)
    elif dataset == "COCOStuff10kDataset":
        mm_config = copy.deepcopy(mm_cocostuff10k_cfg)
    elif dataset == "COCOObjectDataset":
        mm_config = copy.deepcopy(mm_cocoobject_cfg)
    elif dataset == "CityscapesDataset":
        mm_config = copy.deepcopy(mm_cityscapes_cfg)
    elif dataset == "CityscapesFoggyDataset":
        mm_config = copy.deepcopy(mm_cityscapes_foggy_cfg)
        foggy_suffix_map = {
            'light': '_leftImg8bit_foggy_beta_0.005.png',
            'medium': '_leftImg8bit_foggy_beta_0.01.png',
            'dense': '_leftImg8bit_foggy_beta_0.02.png',
        }
        if corruption in foggy_suffix_map:
            mm_config['img_suffix'] = foggy_suffix_map[corruption]
    elif dataset == "BDD100kDataset":
        mm_config = copy.deepcopy(mm_bdd100k_cfg)
    elif dataset == "ADE20kDataset":
        mm_config = copy.deepcopy(mm_ade20k_cfg)
    elif dataset == "LoveDADataset":
        mm_config = copy.deepcopy(mm_loveda_cfg)
        if corruption in ('rural', 'urban'):
            loc = corruption.capitalize()
            mm_config['data_prefix'] = {
                'img_path': f'Val/{loc}/images_png',
                'seg_map_path': f'Val/{loc}/masks_png',
            }
    elif dataset == "ACDCDataset":
        mm_config = copy.deepcopy(mm_acdc_cfg)
        if corruption == 'original':
            raise ValueError(
                "ACDCDataset does not support 'original' — "
                "use real_fog/real_night/real_rain/real_snow or synthetic corruptions"
            )
        if corruption and corruption.startswith('real_'):
            condition = corruption[5:]  # e.g. 'fog', 'night', 'rain', 'snow'
            mm_config['data_prefix'] = {
                'img_path': f'rgb_anon/{condition}/val',
                'seg_map_path': f'gt/{condition}/val',
            }
    elif dataset == "CarlaDataset":
        mm_config = copy.deepcopy(mm_carla_cfg)
        carla_corruption_to_folder = {
            'clear2fog': 'clear_fog_1200',
            'clear2highway': 'clear_highway',
            'clear2rain': 'clear_rain_1200',
            'day2night': 'day_night_1200',
        }
        if corruption == 'original':
            raise ValueError(
                "CarlaDataset does not support 'original' — "
                "use clear2fog/clear2highway/clear2rain/day2night"
            )
        if corruption and corruption in carla_corruption_to_folder:
            folder = carla_corruption_to_folder[corruption]
            mm_config['data_prefix'] = {
                'img_path': folder,
                'seg_map_path': folder,
            }
    elif dataset == "DarkZurichDataset":
        mm_config = copy.deepcopy(mm_dark_zurich_cfg)
    elif dataset == "DrivingDataset":
        driving_sub_cfgs = {
            'cityscapes': ('CityscapesDataset',
                           {'img_path': 'cityscapes/leftImg8bit/val',
                            'seg_map_path': 'cityscapes/gtFine/val'}),
            'acdc': ('ACDCDataset',
                     {'img_path': 'acdc/rgb_anon',
                      'seg_map_path': 'acdc/gt'}),
            'dark_zurich': ('DarkZurichDataset',
                            {'img_path': 'dark_zurich/val/rgb_anon/val',
                             'seg_map_path': 'dark_zurich/val/gt/val'}),
        }
        if corruption not in driving_sub_cfgs:
            raise ValueError(
                f"DrivingDataset corruption must be one of "
                f"{list(driving_sub_cfgs.keys())}, got '{corruption}'"
            )
        sub_type, sub_prefix = driving_sub_cfgs[corruption]
        mm_config = {
            'type': sub_type,
            'data_root': data_dir,
            'data_prefix': sub_prefix,
            'pipeline': [{'type': 'LoadImageFromFile'},
                        {'type': 'LoadAnnotations'},
                        {'type': 'ResizeAndPatchify', 'resize': resize, 'patch_size': patch_size, 'patch_stride': patch_stride},
                        {'type': 'ToTensorAndNormalize', 'mean': CLIP_MEAN, 'std': CLIP_STD},
                        ]
        }
    elif dataset == "PascalVOC20Dataset":
        mm_config = copy.deepcopy(mm_pascalvoc20_cfg)
    elif dataset == "PascalVOC21Dataset":
        mm_config = copy.deepcopy(mm_pascalvoc21_cfg)
    elif dataset == "PascalContext59Dataset":
        mm_config = copy.deepcopy(mm_pascalcontect59_cfg)
    elif dataset == "PascalContext60Dataset":
        mm_config = copy.deepcopy(mm_pascalcontect60_cfg)
    elif dataset == "SUIM6Dataset":
        mm_config = copy.deepcopy(mm_suim6_cfg)
    elif dataset == "SUIM5Dataset":
        mm_config = copy.deepcopy(mm_suim5_cfg)
    elif dataset == "DUTUSEG5Dataset":
        mm_config = copy.deepcopy(mm_dutuseg5_cfg)
    elif dataset == "DUTUSEG4Dataset":
        mm_config = copy.deepcopy(mm_dutuseg4_cfg)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    
    ### add specified configs
    
    mm_config['data_root'] = data_dir
    mm_config['pipeline'][2]['resize'] = init_resize
    mm_config['pipeline'][2]['patch_size'] = patch_size
    mm_config['pipeline'][2]['patch_stride'] = patch_stride


    ### add corruption to the pipline
    # Find the index of 'LoadImageFromFile' in the pipeline
    if corruption == "original" or corruption in ('rural', 'urban') or (corruption and corruption.startswith('real_')) or corruption in ('clear2fog', 'clear2highway', 'clear2rain', 'day2night') or corruption in ('cityscapes', 'acdc', 'dark_zurich') or corruption in ('light', 'medium', 'dense'):
        print(f"No synthetic corruption added to the pipeline (corruption={corruption})")
    else:
        load_image_index = next(
            (i for i, transform in enumerate(mm_config['pipeline']) if transform['type'] == 'LoadImageFromFile'),
            None
        )  
        # Insert the new transform right after 'LoadImageFromFile'
        if load_image_index is not None:
            corrupt_transform = {
                'type': 'CorruptTransform',
                'corruption_severity': 5,
                'corruption_name': corruption
            }
            mm_config['pipeline'].insert(load_image_index + 1, corrupt_transform)

            print(f"+ Corruption '{corruption}' added to the pipeline")
        else:
            raise ValueError("LoadImageFromFile not found in the pipeline")

    ### bulid the dataset from the config using mmseg registry
    dataset = DATASETS.build(mm_config)

    ### bulid the dataloader
    # if num_workers == 0:
    #     persistent_workers = False
    # else:
    #     persistent_workers = True
    
    persistent_workers = False

    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, 
                            collate_fn=custom_collate, persistent_workers=persistent_workers, pin_memory=True,
                            shuffle=shuffle)

    classes = dataset.METAINFO['classes']

    return dataloader, classes

    
