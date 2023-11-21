from numpy import random
from numpy.core.numeric import full
import torch
import numpy as np
import pickle

import tensorflow as tf
import tensorflow_datasets as tfds
tf.config.set_visible_devices([], 'GPU')

from kubric.challenges.point_tracking.dataset import create_point_tracking_dataset

class KubricDataset(torch.utils.data.IterableDataset):
    def __init__(self,
        dataset_location='../datasets/tapvid_davis',
        train_size=(256, 256),
        N=64,    
    ):

        print('loading Kubric dataset...')

        res = create_point_tracking_dataset(
            split='validation',
            train_size=train_size,
            tracks_to_sample=N,
            batch_dims=[1],
            shuffle_buffer_size=None,
            repeat=False,
            random_crop=False,
        )
        self.data = tfds.as_numpy(res)

    # def __get_item_helper(self):
    #     for data in self.data:
    #         rgbs = (data['video'] + 1.) * (255. / 2.)
    #         trajs = data["target_points"]
    #         valids = (~data["visibles"]).astype(np.float32)

    #         trajs = trajs.transpose(1,0,2) # S,N,2
    #         valids = valids.transpose(1,0) # S,N

    #         vis_ok = valids[0] > 0
    #         trajs = trajs[:,vis_ok]
    #         valids = valids[:,vis_ok]

    #         # 1.0,1.0 should lie at the bottom-right corner pixel
    #         H, W, C = rgbs[0].shape
    #         # trajs[:,:,0] *= W-1
    #         # trajs[:,:,1] *= H-1

    #         rgbs = torch.from_numpy(np.stack(rgbs,0)).permute(0,3,1,2) # S,C,H,W
    #         trajs = torch.from_numpy(trajs) # S,N,2
    #         valids = torch.from_numpy(valids) # S,N

    #         sample = {
    #             'rgbs': rgbs,
    #             'trajs': trajs,
    #             'valids': valids,
    #             'visibs': valids,
    #         }
    #         yield sample
        
    def __iter__(self):
        return iter(self.data)

    def __len__(self):
        return len(self.data)


if __name__ == "__main__":
    dataset = KubricDataset()
    for i, data in dataset:
        print(data)
        print(i)
        if i > 10:
            break

