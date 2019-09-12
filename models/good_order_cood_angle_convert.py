import numpy as np
import pdb
import torch

def absolute_angles(prediction_3d):
    absolute_angles = np.zeros([7, 3])
    offset = prediction_3d[1] ## offset !!!!!!!
    limbs = np.zeros([7, 1])
    for i in range(len(ordered_top_edges)):
        i1, i2 = ordered_top_edges[i]
        e1 = prediction_3d[i1] - prediction_3d[i2]
        l = np.linalg.norm(e1)
        limbs[i] = l
        absolute_angles[i] = np.arccos(e1/l)
    return absolute_angles, limbs, offset

ordered_top_edges = [[4, 3], [3, 2], [2, 5], [5, 6], [6, 7], [8, 11], [1, 14]]

def anglelimbtoxyz2(offset, absolute_angles, limbs):
    b = offset.shape[0]
    res_3d = -1 * torch.ones([b, 14, 3]).cuda()
    
    norm_direction = torch.cos(absolute_angles).squeeze()
    # pdb.set_trace()
    limbs = limbs.repeat(1,1,3)
    direction = limbs * norm_direction

    res_3d[:, 1] = offset
    mid_hip = res_3d[:, 1] - direction[:,6]
    res_3d[:, 8] = mid_hip + 0.5 * direction[:,5]
    res_3d[:, 11] = mid_hip - 0.5 * direction[:,5]
    res_3d[:, 2] = res_3d[:, 1] + 0.5 * direction[:,2]
    res_3d[:, 5] = res_3d[:, 1] - 0.5 * direction[:,2]
    res_3d[:, 3] = res_3d[:, 2] + direction[:,1]
    res_3d[:, 4] = res_3d[:, 3] + direction[:,0]
    res_3d[:, 6] = res_3d[:, 5] - direction[:,3]
    res_3d[:, 7] = res_3d[:, 6] - direction[:,4]

    return res_3d

def check_visibility(pose):
    # check if left arm is at behind with depth
    if pose[2, 2] > pose[5, 2] + 5:
        pose = check_arm(pose, [2, 3, 4])
    elif pose[2, 2] +5 < pose[5, 2]:
        pose = check_arm(pose, [5, 6, 7])
    
    return pose

def check_arm(pose, arm_index):
    neck_y = pose[1, 0]
    shoulder_y = pose[arm_index[0], 0]
    for i in arm_index[1:]:
        if abs(pose[i, 0] - neck_y) < 55:
            pose[i] = -1

    return pose
