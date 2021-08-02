
import torch
import torch.utils.data as data
import numpy as np
import quaternion
import cv2
import os
import os.path
import math
import datasets


class IsaacSimDataset(data.Dataset, datasets.imdb):

    def __init__(self, viewport_name, data_dir, classes, symmetries, vertex_weight_inside):
        self._name = 'isaac_sim'
        self._data_dir = os.path.abspath(data_dir)
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
        self._symmetries = symmetries
        self._vertex_weight_inside = vertex_weight_inside

        # iteration relevant information
        self._sample_names = [os.path.splitext(os.path.basename(p))[0] for p in os.listdir(self._rgb_dir)]

    def __getitem__(self, index):
        '''

        '''

        sample_name = self._sample_names[index]

        image = self.__load_rgb(sample_name)
        depth, depth_xyz, depth_mask = self.__load_depth(sample_name)
        semantic_mask, semantic_blob, semantic_data = self.__load_semantic(sample_name)
        pose_blob = self.__load_poses(sample_name)
        bboxes_loose, bboxes_tight = self.__load_bboxes(sample_name)
        intrinsics = self.__load_intrinsics(sample_name)
        mins, maxes, extents = self.__load_bbox3d(sample_name)

        image_info = np.array([image.shape[1], image.shape[2], 1.0, 1.0], dtype=np.float32)

        vertex_targets, vertex_weights = self._generate_vertex_targets(semantic_data, bboxes_tight, pose_blob)

        result = {
            'image_color': image,
            'image_depth': depth_xyz,
            'im_depth': depth,
            'label': semantic_blob,
            'mask': semantic_mask,
            'mask_depth': depth_mask,
            'meta_data': intrinsics,
            'poses': pose_blob,
            'extents': extents,
            'points': np.zeros((self.num_classes, 0, 3), dtype=np.float32),
            'symmetry': self._symmetries,
            'gt_boxes': bboxes_tight,
            'im_info': image_info,
            'vertex_targets': vertex_targets,
            'vertex_weights': vertex_weights
        }

        return result

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

        # convert to torch
        return torch.from_numpy(im)

    def __load_depth(self, sample_name):
        filename = os.path.join(self._depth_dir, f'{sample_name}.png')

        # read image
        depth = cv2.imread(filename, cv2.IMREAD_UNCHANGED).astype(np.float32)
        # scale to interval [0,1]
        depth /= 255.0

        h, w = depth.shape

        # get mask for which depth fields are valid
        depth_mask = (depth > 0.0).astype(np.float32).reshape(1, h, w)

        # TODO how to extract 3D coordinates for each pixel?
        depth_xyz = np.zeros((3, depth.shape[0], depth.shape[1]), dtype=np.float32)

        # convert to cuda
        return torch.from_numpy(depth), \
            torch.from_numpy(depth_xyz), \
            torch.from_numpy(depth_mask)

    def __load_semantic(self, sample_name):
        filename = os.path.join(self._semantic_dir, f'{sample_name}_data.png')

        semantic_data = cv2.imread(filename, cv2.IMREAD_UNCHANGED)

        h, w = semantic_data.shape

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
        semantic_mask = (semantic_data != 0).reshape(h, w, 1).repeat(3, 2).transpose(2, 0, 1)

        return torch.from_numpy(semantic_mask), \
            torch.from_numpy(semantic_blob), \
            torch.from_numpy(semantic_data)

    def __load_poses(self, sample_name):
        pose_filename = os.path.join(self._pose_dir, f'{sample_name}.npy')
        cam_filename = os.path.join(self._camera_dir, f'{sample_name}.npy')

        pose_data = np.load(pose_filename)
        cam_tf = np.load(cam_filename)
        cam_tf_inv = np.linalg.inv(cam_tf)

        n, _ = pose_data.shape

        pose_blob = np.zeros((self.num_classes, 9), dtype=np.float32)
        for i in range(n):
            # first entry is class index
            cls = pose_data[i, 0]
            # recreate transform from flattened array
            tf = pose_data[i, 1:].reshape(4, 4)
            # transform pose into camera frame
            tf = np.matmul(tf, cam_tf_inv)

            translation = tf[3, :3]
            rotation = tf[:3, :3]
            qt = quaternion.from_rotation_matrix(rotation)
            # enable this entry
            pose_blob[i, 0] = 1
            # set the class label
            pose_blob[i, 1] = cls
            # add transform
            pose_blob[i, 2:6] = quaternion.as_float_array(qt)
            pose_blob[i, 6:] = translation

        return torch.from_numpy(pose_blob)

    def __load_bboxes(self, sample_name):

        filename = os.path.join(self._bbox_tight_dir, f'{sample_name}.npy')
        bbox_tight_data = np.load(filename, allow_pickle=True)
        n, = bbox_tight_data.shape

        bbox_tight = np.zeros((self.num_classes, 5), dtype=np.float32)

        for i in range(n):
            cls = bbox_tight_data[i][0]
            bbox_tight[i, 0] = bbox_tight_data[i][6]
            bbox_tight[i, 1] = bbox_tight_data[i][7]
            bbox_tight[i, 2] = bbox_tight_data[i][8]
            bbox_tight[i, 3] = bbox_tight_data[i][9]
            bbox_tight[i, 4] = cls

        filename = os.path.join(self._bbox_loose_dir, f'{sample_name}.npy')
        bbox_loose_data = np.load(filename, allow_pickle=True)
        n, = bbox_loose_data.shape

        bbox_loose = np.zeros((self.num_classes, 5), dtype=np.float32)

        for i in range(n):
            cls = bbox_loose_data[i][0]
            bbox_loose[i, 0] = bbox_loose_data[i][6]
            bbox_loose[i, 1] = bbox_loose_data[i][7]
            bbox_loose[i, 2] = bbox_loose_data[i][8]
            bbox_loose[i, 3] = bbox_loose_data[i][9]
            bbox_loose[i, 4] = cls

        return torch.from_numpy(bbox_loose), torch.from_numpy(bbox_tight)

    def __load_intrinsics(self, sample_name):
        filename = os.path.join(self._camera_dir, f'{sample_name}_intrinsics.npy')

        K = np.load(filename)
        K_inv = np.linalg.pinv(K)

        intrinsics = np.zeros((1, 18), dtype=np.float32)
        intrinsics[0, :9] = K.flatten()
        intrinsics[0, 9:] = K_inv.flatten()

        return torch.from_numpy(intrinsics)

    def _generate_vertex_targets(self, semantic_data, bboxes, pose_blob):

        center = np.zeros((self.num_classes, 2), dtype=np.float32)
        idxs = np.where(bboxes[:, 4] > 0)
        clss = bboxes[idxs, 4].numpy().astype(np.int)
        center[clss, :] = bboxes[idxs, 0:2] + 0.5 * (bboxes[idxs, 2:4] - bboxes[idxs, 0:2])

        pose_z = np.zeros(self.num_classes, dtype=np.float32)
        idxs = np.where(pose_blob[:, 1] > 0)
        clss = pose_blob[idxs, 1].numpy().astype(np.int)
        pose_z[clss] = pose_blob[idxs, 8]

        h, w = semantic_data.shape
        vertex_targets = np.zeros((3 * self.num_classes, h, w), dtype=np.float32)
        vertex_weights = np.zeros((3 * self.num_classes, h, w), dtype=np.float32)
        c = np.zeros((2, 1), dtype=np.float32)

        for i in range(1, self.num_classes):
            cls = self.classes[i]
            y, x = np.where(semantic_data == cls)
            z = abs(pose_z[i])
            if len(x) > 0:
                c[0] = center[i, 0]
                c[1] = center[i, 1]
                R = np.tile(c, (1, len(x))) - np.vstack((x, y))
                # compute the norm
                N = np.linalg.norm(R, axis=0) + 1e-10
                # normalization
                R = np.divide(R, np.tile(N, (2, 1)))
                # assignment
                vertex_targets[3 * i + 0, y, x] = R[0, :]
                vertex_targets[3 * i + 1, y, x] = R[1, :]
                vertex_targets[3 * i + 2, y, x] = math.log(z)

                vertex_weights[3 * i + 0, y, x] = self._vertex_weight_inside
                vertex_weights[3 * i + 1, y, x] = self._vertex_weight_inside
                vertex_weights[3 * i + 2, y, x] = self._vertex_weight_inside

        return vertex_targets, vertex_weights

    def __load_bbox3d(self, sample_name):
        filename = os.path.join(self._bbox3d_dir, f'{sample_name}.npy')

        data = np.load(filename)

        mins = np.zeros((self.num_classes, 3), dtype=np.float32)
        maxes = np.zeros((self.num_classes, 3), dtype=np.float32)
        cls = data[:, 0]

        mins[cls, :] = data[:, 1:4]
        maxes[cls, :] = data[:, 4:]
        extents = maxes - mins

        return mins, maxes, extents
