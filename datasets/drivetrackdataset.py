import torch
import multiprocessing as mp
import numpy as np
import glob
import pickle
import os
import tqdm

import cv2


def preprocess(path, S=24, N=64):
    with np.load(path) as npzfile:
        video = npzfile['video']
        target_points = npzfile['tracks'].astype(np.float32)
        visibles = npzfile['visibles']

    indices = list(range(0, video.shape[0], 24))[1:]
    video_splits = np.array_split(video, indices, axis=0)
    target_points_splits = np.array_split(target_points, indices, axis=1)
    visibles_splits = np.array_split(visibles, indices, axis=1)

    for i, (v, t, vi) in enumerate(zip(video_splits, target_points_splits, visibles_splits)):
        if v.shape != (24, 1280, 1920, 3) or t.shape[1] != 24 or vi.shape[1] != 24:
            continue

        valid = np.any(vi, axis=-1)
        t = t[valid]
        vi = vi[valid]
        
        if t.shape[0] < 256 or t.shape[0] != vi.shape[0]:
            continue
        
        yield {
            'video': v,
            'target_points': t,
            'visibles': vi,
        }


class DrivetrackDataset(torch.utils.data.Dataset):
    def __init__(self,
        dataset_location="/microtel/nfs/datasets/waymo/point_tracks/pips/training",
        S=24,
        N=64,
        strides=[1,2],
        crop_size=(512,896),
        shuffle=True,
        **kwargs,
    ):
        self.data = glob.glob(os.path.join(dataset_location, "*.npz"))
        if shuffle:
            np.random.shuffle(self.data)

        self.S = S
        self.N = N
        self.strides = strides
        self.crop_size = crop_size

    def __getitem__(self, index):
        path = self.data[index]
        with np.load(path) as npzfile:
            data = dict(npzfile)

        rgbs = data["video"][:self.S]  # Video is 1280 x 1920
        trajs = data["target_points"][:, :self.S]  # N,S,2 array
        valids = data["visibles"][:, :self.S].astype(np.float32)  # N,S array

        H, W, _ = rgbs[0].shape
        rgbs = np.stack([cv2.resize(rgb, self.crop_size[::-1], interpolation=cv2.INTER_AREA) for rgb in rgbs], axis=0)
        scale_x = self.crop_size[1] / W
        scale_y = self.crop_size[0] / H
        trajs[..., 0] *= scale_x
        trajs[..., 1] *= scale_y

        trajs = trajs.transpose(1, 0, 2)  # S,N,2
        valids = valids.transpose(1, 0)  # S,N

        # we won't supervise with the extremes, but let's clamp anyway just to be safe
        # trajs = np.minimum(np.maximum(trajs, np.array([-64,-64])), np.array([W+64, H+64])) # S,N,2

        # ensure point is good in frame0
        vis_ok = valids[0] > 0
        trajs = trajs[:, vis_ok]
        valids = valids[:, vis_ok]

        # ensure point is good in frame1
        vis_ok = valids[1] > 0
        trajs = trajs[:, vis_ok]
        valids = valids[:, vis_ok]

        N = trajs.shape[1]
        N_ = min(N, self.N)
        inds = np.random.choice(N, N_, replace=False)
        # prep for batching, by fixing N
        trajs_full = np.zeros((self.S, self.N, 2)).astype(np.float32)
        valids_full = np.zeros((self.S, self.N)).astype(np.float32)
        trajs_full[:, :N_] = trajs[:, inds]
        valids_full[:, :N_] = valids[:, inds]

        rgbs = torch.from_numpy(np.stack(rgbs, 0)).permute(0, 3, 1, 2).byte()  # S,C,H,W
        trajs = torch.from_numpy(trajs_full).float()  # S,N,2
        valids = torch.from_numpy(valids_full).float()  # S,N

        sample = {
            "rgbs": rgbs,
            "trajs": trajs,
            "valids": valids,
            "visibs": valids,
        }
        return sample, True

    def __len__(self):
        return len(self.data)
    

if __name__ == "__main__":
    with mp.Pool(128) as p:
        for i, result in enumerate(p.imap(preprocess, glob.glob("/data1/jchand/datasets/waymo/point_tracks/0.5.0/training"))):
            np.savez(
                f"/data1/jchand/datasets/waymo/point_tracks/pips/{i:05}.npz",
                **result,
            )

