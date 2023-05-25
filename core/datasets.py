# Data loading based on https://github.com/NVIDIA/flownet2-pytorch

import numpy as np
import torch
import torch.utils.data as data

import os
import random
from glob import glob
import os.path as osp

from core.utils import frame_utils
from core.utils.augmentor import FlowAugmentor, SparseFlowAugmentor


class FlowDataset(data.Dataset):
    def __init__(self, aug_params=None, sparse=False, seq_len=2, overlap=True):
        self.augmentor = None
        self.sparse = sparse
        if aug_params is not None:
            if sparse:
                self.augmentor = SparseFlowAugmentor(**aug_params)
            else:
                self.augmentor = FlowAugmentor(**aug_params)

        self.is_test = False
        self.init_seed = False
        self.flow_list = []
        self.image_list = []
        self.extra_info = []
        self.seq_len = seq_len

    def fix_colors(self, imgs):
        if len(imgs[0].shape) == 2:
            imgs = [np.tile(img[..., None], (1, 1, 3)) for img in imgs]
        else:
            imgs = [img[..., :3] for img in imgs]
        return imgs

    def __getitem__(self, index):

        if self.is_test:
            imgs = [frame_utils.read_gen(self.image_list[index][i]) for i in range(self.seq_len)]
            imgs = [np.array(img).astype(np.uint8)[..., :3] for img in imgs]
            imgs = [torch.from_numpy(img).permute(2, 0, 1).float() for img in imgs]
            imgs = torch.stack(imgs).float()
            return imgs, self.extra_info[index]

        if not self.init_seed:
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                torch.manual_seed(worker_info.id)
                np.random.seed(worker_info.id)
                random.seed(worker_info.id)
                self.init_seed = True

        index = index % len(self.image_list)

        flows = []
        valids = []

        for i in range(self.seq_len):
            valid = None
            flow_path = self.flow_list[index][i]
            # print(flow_path, self.sparse)
            if self.sparse:
                flow, valid = frame_utils.readFlowKITTI(flow_path)
            else:
                flow = frame_utils.read_gen(flow_path)
            flows.append(flow)
            valids.append(valid)

        imgs = [frame_utils.read_gen(self.image_list[index][i]) for i in range(self.seq_len)]
        imgs = [np.array(img).astype(np.uint8) for img in imgs]
        flows = [np.array(flow).astype(np.float32) for flow in flows]

        imgs = self.fix_colors(imgs)

        if self.augmentor is not None:
            if self.sparse:
                imgs, flows, valids = self.augmentor(imgs, flows, valids)
            else:
                imgs, flows = self.augmentor(imgs, flows)

        imgs = [torch.from_numpy(img).permute(2, 0, 1).float() for img in imgs]
        flows = [torch.from_numpy(flow).permute(2, 0, 1).float() for flow in flows]

        for i in range(self.seq_len):
            if valids[i] is not None:
                valids[i] = torch.from_numpy(valids[i])
                continue
            valids[i] = (flows[i][0].abs() < 1000) & (flows[i][1].abs() < 1000)

        flows = torch.stack(flows).float()
        imgs = torch.stack(imgs).float()
        valids = torch.stack(valids).float()

        return imgs, flows, valids

    def __rmul__(self, v):
        self.flow_list = v * self.flow_list
        self.image_list = v * self.image_list
        return self

    def __len__(self):
        return len(self.image_list) - self.seq_len + 1


class MpiSintel(FlowDataset):
    def __init__(
            self,
            aug_params=None,
            split='training',
            root='datasets/Sintel',
            dstype='clean',
            seq_len=2,
            overlap=True,
    ):
        super(MpiSintel, self).__init__(
            aug_params,
            seq_len=seq_len,
            overlap=overlap
        )
        flow_root = osp.join(root, split, 'flow')
        image_root = osp.join(root, split, dstype)

        if split == 'test':
            self.is_test = True

        for scene in os.listdir(image_root):
            image_list = sorted(glob(osp.join(image_root, scene, '*.png')))
            flow_list = []
            if not self.is_test:
                flow_list += sorted(
                    glob(osp.join(flow_root, scene, '*.flo'))
                )
            for i in range(len(image_list)-seq_len+1):
                self.image_list += [[image_list[i+k] for k in range(seq_len)]]
                self.extra_info += [(scene, i)]  # scene and frame_id
                # self.image_list = self.image_list[:8]
                # self.extra_info = self.extra_info[:8]

                if not self.is_test:
                    inner_flow_list = []
                    for k in range(seq_len-1):
                        flow = flow_list[i+k]
                        inner_flow_list.append(flow)
                    inner_flow_list.append(flow)
                    self.flow_list += [inner_flow_list]
                # self.flow_list = self.flow_list[:8]


class FlyingChairs(FlowDataset):
    def __init__(
            self,
            aug_params=None,
            split='train',
            root='datasets/FlyingChairs_release/data'
    ):
        super(FlyingChairs, self).__init__(aug_params)

        images = sorted(glob(osp.join(root, '*.ppm')))
        flows = sorted(glob(osp.join(root, '*.flo')))
        assert (len(images)//2 == len(flows))

        split_list = np.loadtxt('datasets/chairs_split.txt', dtype=np.int32)
        for i in range(len(flows)):
            xid = split_list[i]
            if (split=='training' and xid==1) or (split=='validation' and xid==2):
                self.flow_list += [flows[i]]
                self.image_list += [[images[2*i], images[2*i+1]]]


class FlyingThingsSubset(FlowDataset):
    def __init__(self, aug_params=None, root='datasets/FlyingThingsSubset'):
        super(FlyingThingsSubset, self).__init__(aug_params)

        for view in ['left', 'right']:
            image_root = osp.join(root, 'train/image_clean', view)
            flow_root_forward = osp.join(root, 'train/flow', view, 'into_future')
            flow_root_backward = osp.join(root, 'train/flow/', view, 'into_past')

            image_list = sorted(os.listdir(image_root))
            flow_forward = set(os.listdir(flow_root_forward))
            flow_backward = set(os.listdir(flow_root_backward))

            for i in range(len(image_list)-1):
                img1 = image_list[i]
                img2 = image_list[i+1]

                image_path1 = osp.join(image_root, img1)
                image_path2 = osp.join(image_root, img2)

                if img1.replace('.png', '.flo') in flow_forward:
                    self.image_list += [[image_path1, image_path2]]
                    self.flow_list += [
                        osp.join(
                            flow_root_forward,
                            img1.replace('.png', '.flo')
                        )
                    ]

                if img2.replace('.png', '.flo') in flow_backward:
                    self.image_list += [[image_path2, image_path1]]
                    self.flow_list += [
                        osp.join(
                            flow_root_backward,
                            img2.replace('.png', '.flo')
                        )
                    ]


class FlyingThings3D(FlowDataset):
    def __init__(self, aug_params=None, root='datasets/FlyingThings3D', split='train', dstype='frames_cleanpass'):
        super(FlyingThings3D, self).__init__(aug_params)

        if split == 'train':
            dir_prefix = 'TRAIN'
        elif split == 'test':
            dir_prefix = 'TEST'
        else:
            raise ValueError('Unknown split for FlyingThings3D.')

        for cam in ['left']:
            for direction in ['into_future', 'into_past']:
                image_dirs = sorted(glob(osp.join(root, dstype, f'{dir_prefix}/*/*')))
                image_dirs = sorted([osp.join(f, cam) for f in image_dirs])

                flow_dirs = sorted(glob(osp.join(root, f'optical_flow/{dir_prefix}/*/*')))
                flow_dirs = sorted([osp.join(f, direction, cam) for f in flow_dirs])

                for idir, fdir in zip(image_dirs, flow_dirs):
                    images = sorted(glob(osp.join(idir, '*.png')))
                    flows = sorted(glob(osp.join(fdir, '*.pfm')))
                    for i in range(len(flows)-1):
                        if direction == 'into_future':
                            self.image_list += [[images[i], images[i+1]]]
                            self.flow_list += [flows[i]]
                        elif direction == 'into_past':
                            self.image_list += [[images[i+1], images[i]]]
                            self.flow_list += [flows[i+1]]


class KITTI(FlowDataset):
    def __init__(self, aug_params=None, split='training', root='datasets/KITTI'):
        super(KITTI, self).__init__(aug_params, sparse=True)
        if split == 'testing':
            self.is_test = True

        root = osp.join(root, split)
        images1 = sorted(glob(osp.join(root, 'image_2/*_10.png')))
        images2 = sorted(glob(osp.join(root, 'image_2/*_11.png')))

        for img1, img2 in zip(images1, images2):
            frame_id = img1.split('/')[-1]
            self.extra_info += [[frame_id]]
            self.image_list += [[img1, img2]]

        if split == 'training':
            self.flow_list = sorted(glob(osp.join(root, 'flow_occ/*_10.png')))


class HD1K(FlowDataset):
    def __init__(self, aug_params=None, root='datasets/HD1k'):
        super(HD1K, self).__init__(aug_params, sparse=True)

        seq_ix = 0
        while 1:
            flows = sorted(glob(os.path.join(root, 'hd1k_flow_gt', 'flow_occ/%06d_*.png' % seq_ix)))
            images = sorted(glob(os.path.join(root, 'hd1k_input', 'image_2/%06d_*.png' % seq_ix)))

            if len(flows) == 0:
                break

            for i in range(len(flows)-1):
                self.flow_list += [flows[i]]
                self.image_list += [[images[i], images[i+1]]]

            seq_ix += 1


def fetch_dataloader(args, TRAIN_DS='C+T+K+S+H'):
    """ Create the data loader for the corresponding trainign set """

    if args.stage == 'chairs':
        aug_params = {
            'crop_size': args.image_size,
            'min_scale': -0.1,
            'max_scale': 1.0,
            'do_flip': True
        }
        train_dataset = FlyingChairs(aug_params, split='training')

    elif args.stage == 'things':
        aug_params = {
            'crop_size': args.image_size,
            'min_scale': -0.4,
            'max_scale': 0.8,
            'do_flip': True
        }
        train_dataset = FlyingThingsSubset(aug_params)

    elif args.stage == 'sintel':
        aug_params = {
            'crop_size': args.image_size,
            'min_scale': -0.2,
            'max_scale': 0.6,
            'do_flip': True
        }
        things = FlyingThingsSubset(aug_params)
        sintel_clean = MpiSintel(aug_params, split='training', dstype='clean')
        sintel_final = MpiSintel(aug_params, split='training', dstype='final')
        # train_dataset = 100*sintel_clean + 100*sintel_final

        if TRAIN_DS == 'C+T+K+S+H':
            kitti = KITTI(
                {
                    'crop_size': args.image_size,
                    'min_scale': -0.3,
                    'max_scale': 0.5,
                    'do_flip': True
                }
            )
            hd1k = HD1K(
                {
                    'crop_size': args.image_size,
                    'min_scale': -0.5,
                    'max_scale': 0.2,
                    'do_flip': True
                }
            )
            train_dataset = 100*sintel_clean + 100*sintel_final + 200*kitti + 5*hd1k + things

        elif TRAIN_DS == 'C+T+K/S':
            train_dataset = 100*sintel_clean + 100*sintel_final + things

    elif args.stage == 'kitti':
        aug_params = {
            'crop_size': args.image_size,
            'min_scale': -0.2,
            'max_scale': 0.4,
            'do_flip': False
        }
        train_dataset = KITTI(aug_params, split='training')

    train_loader = data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        pin_memory=False,
        shuffle=True,
        num_workers=4,
        drop_last=True
    )

    print(f'Training with {len(train_dataset)} image pairs')
    return train_loader
