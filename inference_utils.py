import time
import numpy as np
import saverloader
from nets.pips2 import Pips
import utils.improc
import utils.misc
import random
from utils.basic import print_, print_stats
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF


def load_pips(checkpoint_dir, stride=8):
    model = Pips(stride=stride)

    utils.misc.count_parameters(model)

    _ = saverloader.load(checkpoint_dir, model)
    model.eval()
    return model

def run_pips(model, video, query_points, iters=8, S_max=8, image_size=(384, 512)):
    video = torch.from_numpy(video).permute(0, 3, 1, 2).cuda().float()
    # tracks = torch.from_numpy(tracks).permute(1, 0, 2).cuda().float()
    tracks = torch.fliplr(torch.from_numpy(query_points[..., 1:])).unsqueeze(0).cuda().float()

    S, C, H, W = video.shape
    H_, W_ = image_size
    sy = H_ / H
    sx = W_ / W
    rgbs = TF.resize(video, (H_, W_))
    tracks[..., 0] *= sx
    tracks[..., 1] *= sy

    trajs_e = tracks[0].repeat(S, 1, 1)

    # Batch inputs
    rgbs = rgbs.unsqueeze(0)
    trajs_e = trajs_e.unsqueeze(0)

    cur_frame = 0
    done = False
    feat_init = None
    while not done:
        end_frame = cur_frame + S_max

        if end_frame > S:
            diff = end_frame-S
            end_frame = end_frame-diff
            cur_frame = max(cur_frame-diff,0)

        traj_seq = trajs_e[:, cur_frame:end_frame]
        rgb_seq = rgbs[:, cur_frame:end_frame]
        S_local = rgb_seq.shape[1]

        if feat_init is not None:
            feat_init = [fi[:,:S_local] for fi in feat_init]
        
        preds, preds_anim, feat_init, _ = model(traj_seq, rgb_seq, iters=iters, feat_init=feat_init)

        trajs_e[:, cur_frame:end_frame] = preds[-1][:, :S_local]
        trajs_e[:, end_frame:] = trajs_e[:, end_frame-1:end_frame]

        if end_frame >= S:
            done = True
        else:
            cur_frame = cur_frame + S_max - 1

    predicted_tracks = trajs_e.squeeze(0).cpu().numpy()
    predicted_tracks[..., 0] /= sx
    predicted_tracks[..., 1] /= sy

    return predicted_tracks




    
    