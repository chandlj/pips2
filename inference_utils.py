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
import matplotlib.pyplot as plt


def load_pips(checkpoint_dir, stride=8):
    model = Pips(stride=stride).cuda()

    _ = saverloader.load(checkpoint_dir, model)
    model.eval()
    return model

def run_pips(model, video, query_points, iters=8, S_max=8, image_size=(384, 512)):
    video = torch.from_numpy(video).permute(0, 3, 1, 2).float()
    xy0 = torch.fliplr(torch.from_numpy(query_points[..., 1:])).unsqueeze(0).float() # 1, N, 2

    S, C, H, W = video.shape
    H_, W_ = image_size
    sy = H_ / H
    sx = W_ / W
    rgbs = TF.resize(video, (H_, W_))
    xy0[..., 0] *= sx
    xy0[..., 1] *= sy

    # zero-vel init
    trajs_e = xy0.unsqueeze(1).repeat(1,S,1,1)

    # Batch inputs
    rgbs = rgbs.unsqueeze(0) # 1,S,C,H,W

    idx = list(range(0, max(S-S_max,1), S_max))
    feat_init = None
    for si in idx:
        rgb_seq = rgbs[:, si:si+S_max].cuda()
        traj_seq = trajs_e[:, si:si+S_max].cuda()
        S_local = rgb_seq.shape[1]

        if feat_init is not None:
            feat_init = [fi[:,:S_local] for fi in feat_init]
            
        preds, preds_anim, feat_init, _ = model(traj_seq, rgb_seq, iters=iters, feat_init=feat_init, beautify=True)

        trajs_e[:, si:si+S_max] = preds[-1][:, :S_local]
        trajs_e[:, si+S_max:] = trajs_e[:, si+S_max-1:si+S_max]

        del preds
        del preds_anim
        del rgb_seq
        del traj_seq
        torch.cuda.empty_cache()

    if S % S_max != 0:
        rgb_seq = rgbs[:, idx[-1]+S_max:].cuda()
        traj_seq = trajs_e[:, idx[-1]+S_max:].cuda()
        S_local = rgb_seq.shape[1]

        if feat_init is not None:
            feat_init = [fi[:,:S_local] for fi in feat_init]

        preds, preds_anim, feat_init, _ = model(traj_seq, rgb_seq, iters=iters, feat_init=feat_init, beautify=True)

        trajs_e[:, idx[-1]+S_max:] = preds[-1][:, :S_local]
        trajs_e[:, idx[-1]+S_max:] = trajs_e[:, idx[-1]+S_max-1:idx[-1]+S_max]

    predicted_tracks = trajs_e.squeeze(0).cpu().numpy()
    predicted_tracks[..., 0] /= sx
    predicted_tracks[..., 1] /= sy

    return predicted_tracks.transpose(1, 0, 2)




    
    