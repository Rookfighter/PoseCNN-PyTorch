import torch
import time

def loss_cross_entropy(scores, labels):
    """
    scores: a tensor [batch_size, num_classes, height, width]
    labels: a tensor [batch_size, num_classes, height, width]
    """

    cross_entropy = -torch.sum(labels * scores, dim=1)
    loss = torch.div(torch.sum(cross_entropy), torch.sum(labels)+1e-10)

    return loss


def smooth_l1_loss(vertex_pred, vertex_targets, vertex_weights, sigma=1.0):
    sigma_2 = sigma ** 2
    vertex_diff = vertex_pred - vertex_targets
    diff = torch.mul(vertex_weights, vertex_diff)
    abs_diff = torch.abs(diff)
    smoothL1_sign = torch.lt(abs_diff, 1. / sigma_2).float().detach()
    in_loss = torch.pow(diff, 2) * (sigma_2 / 2.) * smoothL1_sign \
            + (abs_diff - (0.5 / sigma_2)) * (1. - smoothL1_sign)
    loss = torch.div( torch.sum(in_loss), torch.sum(vertex_weights) + 1e-10 )
    return loss


def step_epoch(network, optimizer, samples, start_ts, epoch, translation_weight=1.0):
    '''
    Trains the network for a single epoch, where epoch means one iteration through all samples.

    A sample is a dictionary with the following entries:

    {
        'image_color': im_blob,
        'im_depth': im_depth,
        'label': label_blob,
        'mask': mask,
        'meta_data': meta_data_blob,
        'poses': pose_blob,
        'extents': self._extents,
        'points': self._point_blob,
        'symmetry': self._symmetry,
        'gt_boxes': gt_boxes,
        'im_info': im_info,
        'vertex_targets': vertex_targets,
        'vertex_weights': vertex_weights
    }

    :param network: the network to be trained
    :param optimizer: the optimizer that is used to calculte the network weights
    :param samples: iterable object with sample data
    :param start_ts: timestamp in seconds when training started
    :param epoch: number of current epoch
    '''

    # put the network into training mode
    network.train()

    for i, sample in enumerate(samples):

        # extract data from sample
        inputs = sample['image_color'].cuda()
        labels = sample['label'].cuda()
        meta_data = sample['meta_data'].cuda()
        extents = sample['extents'][0, :, :].cuda()
        gt_boxes = sample['gt_boxes'].cuda()
        poses = sample['poses'].cuda()
        points = sample['points'][0, :, :, :].cuda()
        symmetry = sample['symmetry'][0, :].cuda()
        vertex_targets = sample['vertex_targets'].cuda()
        vertex_weights = sample['vertex_weights'].cuda()

        # forward pass through network
        if network.train_translation:
            if network.train_rotation:
                out_logsoftmax, out_weight, out_vertex, out_logsoftmax_box, \
                    bbox_labels, bbox_pred, bbox_targets, bbox_inside_weights, loss_pose_tensor, poses_weight \
                    = network(inputs, labels, meta_data, extents, gt_boxes, poses, points, symmetry)

                loss_label = loss_cross_entropy(out_logsoftmax, out_weight)
                loss_vertex = translation_weight * smooth_l1_loss(out_vertex, vertex_targets, vertex_weights)
                loss_box = loss_cross_entropy(out_logsoftmax_box, bbox_labels)
                loss_location = smooth_l1_loss(bbox_pred, bbox_targets, bbox_inside_weights)
                loss_pose = torch.mean(loss_pose_tensor)
                loss = loss_label + loss_vertex + loss_box + loss_location + loss_pose
            else:
                out_logsoftmax, out_weight, out_vertex, out_logsoftmax_box, \
                    bbox_labels, bbox_pred, bbox_targets, bbox_inside_weights \
                    = network(inputs, labels, meta_data, extents, gt_boxes, poses, points, symmetry)

                loss_label = loss_cross_entropy(out_logsoftmax, out_weight)
                loss_vertex = translation_weight * smooth_l1_loss(out_vertex, vertex_targets, vertex_weights)
                loss_box = loss_cross_entropy(out_logsoftmax_box, bbox_labels)
                loss_location = smooth_l1_loss(bbox_pred, bbox_targets, bbox_inside_weights)
                loss = loss_label + loss_vertex + loss_box + loss_location
        else:
            out_logsoftmax, out_weight = network(inputs, labels, meta_data, extents, gt_boxes, poses, points, symmetry)
            loss = loss_cross_entropy(out_logsoftmax, out_weight)

        # compute gradient and do optimization step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        duration = time.perf_counter() - start_ts
        learning_rate = optimizer.param_groups[0]['lr']
        print(f'[{duration}s] epoch={epoch} sample={i} loss={loss} lr={learning_rate}')