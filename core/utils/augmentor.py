# This code was originally taken from RAFT without modification
# https://github.com/princeton-vl/RAFT/blob/master/utils/augmentor.py

import numpy as np
from PIL import Image

import cv2
from torchvision.transforms import ColorJitter

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)


class FlowAugmentor:
    def __init__(self, crop_size, min_scale=-0.2, max_scale=0.5, do_flip=True):

        # spatial augmentation params
        self.crop_size = crop_size
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.spatial_aug_prob = 0.8
        self.stretch_prob = 0.8
        self.max_stretch = 0.2

        # flip augmentation params
        self.do_flip = do_flip
        self.h_flip_prob = 0.5
        self.v_flip_prob = 0.1

        # photometric augmentation params
        self.photo_aug = ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.5/3.14)
        self.asymmetric_color_aug_prob = 0.2
        self.eraser_aug_prob = 0.5

    def color_transform(self, imgs):
        """ Photometric augmentation """

        # asymmetric
        if np.random.rand() < self.asymmetric_color_aug_prob:
            imgs = [np.array(
                self.photo_aug(Image.fromarray(img)), dtype=np.uint8
            ) for img in imgs]

        # symmetric
        else:
            img_amount = len(imgs)
            image_stack = np.concatenate(imgs, axis=0)
            image_stack = np.array(
                self.photo_aug(Image.fromarray(image_stack)), dtype=np.uint8
            )
            imgs = np.split(image_stack, img_amount, axis=0)

        return imgs

    def eraser_transform(self, imgs, bounds=[50, 100]):
        """ Occlusion augmentation """

        ht, wd = imgs[0].shape[:2]
        if np.random.rand() < self.eraser_aug_prob:
            for i in range(1, len(imgs)):
                mean_color = np.mean(imgs[i].reshape(-1, 3), axis=0)
                for _ in range(np.random.randint(1, 3)):
                    x0 = np.random.randint(0, wd)
                    y0 = np.random.randint(0, ht)
                    dx = np.random.randint(bounds[0], bounds[1])
                    dy = np.random.randint(bounds[0], bounds[1])
                    imgs[i][y0:y0+dy, x0:x0+dx, :] = mean_color

        return imgs

    def spatial_transform(self, imgs, flows):
        # randomly sample scale
        ht, wd = imgs[0].shape[:2]
        min_scale = np.maximum(
            (self.crop_size[0] + 8) / float(ht),
            (self.crop_size[1] + 8) / float(wd))

        scale = 2 ** np.random.uniform(self.min_scale, self.max_scale)
        scale_x = scale
        scale_y = scale
        if np.random.rand() < self.stretch_prob:
            scale_x *= 2 ** np.random.uniform(-self.max_stretch, self.max_stretch)
            scale_y *= 2 ** np.random.uniform(-self.max_stretch, self.max_stretch)

        scale_x = np.clip(scale_x, min_scale, None)
        scale_y = np.clip(scale_y, min_scale, None)

        if np.random.rand() < self.spatial_aug_prob:
            # rescale the images
            for i in range(len(imgs)):
                imgs[i] = cv2.resize(
                    imgs[i], None, fx=scale_x, fy=scale_y,
                    interpolation=cv2.INTER_LINEAR
                )
                flows[i] = cv2.resize(
                    flows[i], None, fx=scale_x, fy=scale_y,
                    interpolation=cv2.INTER_LINEAR
                )
                flows[i] = flows[i] * [scale_x, scale_y]

        imgs = np.stack(imgs, axis=0)
        flows = np.stack(flows, axis=0)

        if self.do_flip:
            if np.random.rand() < self.h_flip_prob:  # h-flip
                imgs = imgs[:, :, ::-1]
                flows = flows[:, :, ::-1] * [-1.0, 1.0]

            if np.random.rand() < self.v_flip_prob:  # v-flip
                imgs = imgs[:, ::-1, :]
                flows = flows[:, ::-1, :] * [1.0, -1.0]

        y0 = np.random.randint(0, imgs[0].shape[0] - self.crop_size[0])
        x0 = np.random.randint(0, imgs[0].shape[1] - self.crop_size[1])

        imgs = imgs[:, y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
        flows = flows[:, y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]

        imgs = np.split(imgs, imgs.shape[0], axis=0)
        imgs = [img.squeeze() for img in imgs]
        flows = np.split(flows, flows.shape[0], axis=0)
        flows = [flow.squeeze() for flow in flows]

        return imgs, flows

    def __call__(self, imgs, flows):
        imgs = self.color_transform(imgs)
        imgs = self.eraser_transform(imgs)
        imgs, flows = self.spatial_transform(imgs, flows)

        for i in range(len(imgs)):
            imgs[i] = np.ascontiguousarray(imgs[i])
            flows[i] = np.ascontiguousarray(flows[i])

        return imgs, flows


class SparseFlowAugmentor:
    def __init__(self, crop_size, min_scale=-0.2, max_scale=0.5, do_flip=False):
        # spatial augmentation params
        self.crop_size = crop_size
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.spatial_aug_prob = 0.8
        self.stretch_prob = 0.8
        self.max_stretch = 0.2

        # flip augmentation params
        self.do_flip = do_flip
        self.h_flip_prob = 0.5
        self.v_flip_prob = 0.1

        # photometric augmentation params
        self.photo_aug = ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3/3.14)
        self.asymmetric_color_aug_prob = 0.2
        self.eraser_aug_prob = 0.5

    def color_transform(self, imgs):
        img_amount = len(imgs)
        image_stack = np.concatenate(imgs, axis=0)
        image_stack = np.array(
            self.photo_aug(Image.fromarray(image_stack)), dtype=np.uint8
        )
        imgs = np.split(image_stack, img_amount, axis=0)
        return imgs

    def eraser_transform(self, imgs):
        ht, wd = imgs[0].shape[:2]
        if np.random.rand() < self.eraser_aug_prob:
            for i in range(1, len(imgs)):
                mean_color = np.mean(imgs[i].reshape(-1, 3), axis=0)
                for _ in range(np.random.randint(1, 3)):
                    x0 = np.random.randint(0, wd)
                    y0 = np.random.randint(0, ht)
                    dx = np.random.randint(50, 100)
                    dy = np.random.randint(50, 100)
                    imgs[i][y0:y0+dy, x0:x0+dx, :] = mean_color

        return imgs

    def resize_sparse_flow_map(self, flow, valid, fx=1.0, fy=1.0):
        ht, wd = flow.shape[:2]
        coords = np.meshgrid(np.arange(wd), np.arange(ht))
        coords = np.stack(coords, axis=-1)

        coords = coords.reshape(-1, 2).astype(np.float32)
        flow = flow.reshape(-1, 2).astype(np.float32)
        valid = valid.reshape(-1).astype(np.float32)

        coords0 = coords[valid >= 1]
        flow0 = flow[valid >= 1]

        ht1 = int(round(ht * fy))
        wd1 = int(round(wd * fx))

        coords1 = coords0 * [fx, fy]
        flow1 = flow0 * [fx, fy]

        xx = np.round(coords1[:, 0]).astype(np.int32)
        yy = np.round(coords1[:, 1]).astype(np.int32)

        v = (xx > 0) & (xx < wd1) & (yy > 0) & (yy < ht1)
        xx = xx[v]
        yy = yy[v]
        flow1 = flow1[v]

        flow_img = np.zeros([ht1, wd1, 2], dtype=np.float32)
        valid_img = np.zeros([ht1, wd1], dtype=np.int32)

        flow_img[yy, xx] = flow1
        valid_img[yy, xx] = 1

        return flow_img, valid_img

    def spatial_transform(self, imgs, flows, valids):
        # randomly sample scale

        ht, wd = imgs[0].shape[:2]
        min_scale = np.maximum(
            (self.crop_size[0] + 1) / float(ht),
            (self.crop_size[1] + 1) / float(wd))

        scale = 2 ** np.random.uniform(self.min_scale, self.max_scale)
        scale_x = np.clip(scale, min_scale, None)
        scale_y = np.clip(scale, min_scale, None)

        if np.random.rand() < self.spatial_aug_prob:
            # rescale the images
            for i in range(len(imgs)):
                imgs[i] = cv2.resize(
                    imgs[i], None, fx=scale_x, fy=scale_y,
                    interpolation=cv2.INTER_LINEAR
                )
                flows[i], valids[i] = self.resize_sparse_flow_map(
                    flows[i], valids[i], fx=scale_x, fy=scale_y
                )

        imgs = np.stack(imgs, axis=0)
        flows = np.stack(flows, axis=0)
        valids = np.stack(valids, axis=0)

        if self.do_flip:
            if np.random.rand() < 0.5:  # h-flip
                imgs = imgs[:, :, ::-1]
                flows = flows[:, :, ::-1] * [-1.0, 1.0]
                valids = valids[:, :, ::-1]

        margin_y = 20
        margin_x = 50

        y0 = np.random.randint(0, imgs[0].shape[0] - self.crop_size[0] + margin_y)
        x0 = np.random.randint(-margin_x, imgs[0].shape[1] - self.crop_size[1] + margin_x)

        y0 = np.clip(y0, 0, imgs[0].shape[0] - self.crop_size[0])
        x0 = np.clip(x0, 0, imgs[0].shape[1] - self.crop_size[1])

        imgs = imgs[:, y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
        flows = flows[:, y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
        valids = valids[:, y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]

        imgs = np.split(imgs, imgs.shape[0], axis=0)
        imgs = [img.squeeze() for img in imgs]
        flows = np.split(flows, flows.shape[0], axis=0)
        flows = [flow.squeeze() for flow in flows]
        valids = np.split(valids, valids.shape[0], axis=0)
        valids = [valid.squeeze() for valid in valids]

        return imgs, flows, valids

    def __call__(self, imgs, flows, valids):
        imgs = self.color_transform(imgs)
        imgs = self.eraser_transform(imgs)
        imgs, flows, valids = self.spatial_transform(imgs, flows, valids)

        for i in range(len(imgs)):
            imgs[i] = np.ascontiguousarray(imgs[i])
            flows[i] = np.ascontiguousarray(flows[i])
            valids[i] = np.ascontiguousarray(valids[i])

        return imgs, flows, valids
