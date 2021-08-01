
from random import sample
import torch
import torch.utils.data as data
import numpy as np
import quaternion
import cv2
import os
import os.path

import datasets
from fcn.config import cfg

class IsaacSimDataset(data.Dataset, datasets.imdb):

    def __init__(self, viewport_name, data_dir, classes, symmetries):

        self._data_dir = data_dir
        self._viewport_name = viewport_name
        self._rgb_dir = os.path.join(self._data_dir, self._viewport_name, 'rgb')
        self._depth_dir = os.path.join(self._data_dir, self._viewport_name, 'depth')
        self._semantic_dir = os.path.join(self._data_dir, self._viewport_name, 'semantic')
        self._instance_dir = os.path.join(self._data_dir, self._viewport_name, 'instance')
        self._camera_dir = os.path.join(self._data_dir, self._viewport_name, 'camera')
        self._pose_dir = os.path.join(self._data_dir, self._viewport_name, 'poses')
        self._bbox_loose_dir = os.path.join(self._data_dir, self._viewport_name, 'bbox_2d_loose')
        self._bbox_tight_dir = os.path.join(self._data_dir, self._viewport_name, 'bbox_2d_tight')

        self._classes = [int(c) for c in classes]

        # 3D dimensions of each object class
        self._extents = np.array([[10, 10, 10] for _ in self._classes])
        self._symmetries = symmetries

        # iteration relevant information
        self._sample_names = [os.path.splitext(os.path.basename(p))[0] for p in os.listdir(self._rgb_dir) if os.path.isfile(p)]

    def __getitem__(self, index):
        '''

        '''

        sample_name = self._sample_names[index]

        image = self.__load_rgb(sample_name)
        depth, depth_xyz, depth_mask = self.__load_depth(sample_name)
        semantic_mask, semantic_blob = self.__load_semantic(sample_name)
        pose_blob = self.__load_poses(sample_name)
        bboxes_loose, bboxes_tight = self.__load_bboxes(sample_name)
        intrinsics = self.__load_intrinsics(sample_name)

        image_info = np.array([image.shape[1], image.shape[2], cfg.TRAIN.SCALES_BASE[0], 1], dtype=np.float32)

        return {
            'image_color': image,
            'image_depth': depth_xyz,
            'im_depth': depth,
            'label': semantic_blob,
            'mask': semantic_mask,
            'mask_depth': depth_mask,
            'meta_data':intrinsics,
            'poses': pose_blob,
            'extents': self._extents,
            'points': np.array([[]]),
            'symmetry': self._symmetries,
            'gt_boxes': bboxes_tight,
            'im_info': image_info
        }


    def __len__(self):
        return len(self._sample_names)


    def __load_rgb(self, sample_name):
        filename = os.path.join(self._rgb_dir, f'{sample_name}.png')

        # read image
        im = cv2.imread(filename, cv2.IMREAD_COLOR).astype(np.float32)
        # change shape to match PoseCNN format (d, h, w)
        im = im.transpose(2, 0, 1)
        # scale to interval [0,1]
        im /= 255.0

        # convert to cuda
        return torch.from_numpy(im).cuda().float()


    def __load_depth(self, sample_name):
        filename = os.path.join(self._depth_dir, f'{sample_name}.png')

        # read image
        depth = cv2.imread(filename, cv2.IMREAD_UNCHANGED).astype(np.float32)
        # scale to interval [0,1]
        depth /= 255.0

        # get mask for which depth fields are valid
        depth_mask = (depth > 0.0).astype(np.float32)

        # TODO how to extract 3D coordinates for each pixel?
        depth_xyz = np.zeros((depth.shape[0], depth.shape[1], 3), dtype=np.float32)

        # convert to cuda
        return torch.from_numpy(depth).cuda().float(), \
            torch.from_numpy(depth_xyz).cuda().float(), \
            torch.from_numpy(depth_mask).cuda().float()


    def __load_semantic(self, sample_name):
        filename = os.path.join(self._semantic_dir, f'{sample_name}_data.png')

        semantic_data = cv2.imread(filename, cv2.IMREAD_UNCHANGED)

        w, h, d = semantic_data.shape

        semantic_blob = np.zeros((self.num_classes, h, w), dtype=np.float32)
        # class '0' is background, set it all to true
        semantic_blob[0, :, :] = 1
        for cls in self.classes:
            idx = np.where(semantic_data == cls)
            if len(idx[0] > 0):
                # set the respective values of background to false
                semantic_blob[0, idx[0], idx[1]] = 0
                # set all indices of this class to true
                semantic_blob[cls, idx[0], idx[1]] = 1

        # determine which parts of the image are segmented
        semantic_mask = (semantic_data != 0).transpose(2, 0, 1).repeat(3, 1, 1)

        return semantic_mask, semantic_blob


    def __load_poses(self, sample_name):
        pose_filename = os.path.join(self._pose_dir, f'{sample_name}.npy')
        cam_filename = os.path.join(self._camera_dir, f'{sample_name}.npy')

        pose_data = np.transpose(np.load(pose_filename))
        cam_tf = np.transpose(np.load(cam_filename))
        cam_tf_inv = np.linalg.inv(cam_tf)

        n, _ = pose_data.shape

        pose_blob = np.zeros((self.num_classes, 9), dtype=np.float32)
        for i in range(n):
            # first entry is class index
            cls = pose_data[i, 0]
            # recreate transform from flattened array
            tf = pose_data[i, 1:].reshape(4, 4)
            # transform pose into camera frame
            tf = cam_tf_inv.matmul(tf)
            translation = tf[0, :3]
            rotation = tf[1:, :3]
            qt = quaternion.from_rotation_matrix(rotation)
            # enable this entry
            pose_blob[i, 0] = 1
            # set the class label
            pose_blob[i, 1] =  cls
            # add transform
            pose_blob[i, 2:6] = qt
            pose_blob[i, 6:] = translation

        return pose_blob


    def __load_bboxes(self, sample_name):

        filename = os.path.join(self._bbox_tight_dir, f'{sample_name}.npy')
        bbox_tight_data = np.load(filename, allow_pickle=True)
        n, = bbox_tight_data.shape

        bbox_tight = np.zeros((self.num_classes, 5), dtype=np.float32)

        for i in range(n):
            cls = bbox_tight_data[i, 0]
            bbox_tight[i, 0] = bbox_tight_data[i, 6]
            bbox_tight[i, 1] = bbox_tight_data[i, 7]
            bbox_tight[i, 2] = bbox_tight_data[i, 8]
            bbox_tight[i, 3] = bbox_tight_data[i, 9]
            bbox_tight[i, 4] = cls

        filename = os.path.join(self._bbox_loose_dir, f'{sample_name}.npy')
        bbox_loose_data = np.load(filename, allow_pickle=True)
        n, = bbox_loose_data.shape

        bbox_loose = np.zeros((self.num_classes, 5), dtype=np.float32)

        for i in range(n):
            cls = bbox_loose_data[i, 0]
            bbox_loose[i, 0] = bbox_loose_data[i, 6]
            bbox_loose[i, 1] = bbox_loose_data[i, 7]
            bbox_loose[i, 2] = bbox_loose_data[i, 8]
            bbox_loose[i, 3] = bbox_loose_data[i, 9]
            bbox_loose[i, 4] = cls

        return bbox_loose, bbox_tight


    def __load_intrinsics(self, sample_name):
        filename = os.path.join(self._camera_dir, f'{sample_name}_intrinsics.npy')

        K = np.load(filename)
        K_inv = np.linalg.pinv(K)

        intrinsics = np.zeros((1, 18), dtype=np.float32)
        intrinsics[:9] = K.flatten()
        intrinsics[9:] = K_inv.flatten()

        return intrinsics


