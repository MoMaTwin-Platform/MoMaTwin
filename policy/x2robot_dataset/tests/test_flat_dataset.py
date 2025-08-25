'''
Load data from raw data and create a dataset for training.
'''

from x2robot_dataset.iterable_dataset_tfds import FlatIterator, X2RobotDataset, collate_wrapper

if __name__ == "__main__":
    dataset_path = '/x2robot/zhengwei/10000/20240501-clean-dish-addition'

    dataset, num_data = X2RobotDataset.make_interleaved_dataset(
        dataset_paths =[dataset_path],
        split='train',
        is_bi_mode=True,
        from_rawdata=True,
        train_val_split=0.9,
        train_split_seed=42,
        preload_pool_size = 10,
        num_preloader_threads = 16,
        max_epoch=1000)
    

    # collate_fn = prefrenece_collate_wrapper(rgb_keys=["face_view", "left_wrist_view", "right_wrist_view"], 
    #                                         rank=0,
    #                                         batch_size=128)
    collate_fn = collate_wrapper(rgb_keys=["face_view", "left_wrist_view", "right_wrist_view"],
                                n_obs_steps = 10,
                                horizon = 20,
                                action_dim=14,
                                rank=1,
                                batch_size=32)

    iterator = FlatIterator( 
                 tf_dataset = dataset,
                 max_buffersize = 10000,
                 num_processes = 8,
                 batch_size = 32,
                 num_dp  = 1,
                 rank =0,
                 n_obs_steps = 10,
                 horizon  = 20,
                 n_sub_samples = 1024,
                 n_preferences=6,
                 skip_every_n_frames = 1,
                 collate_fn=collate_fn,
                #  sampler_type='preference',
                sampler_type='subsequence',
                 seed = 42)
    
    import tqdm
    for i, batch in enumerate(tqdm.tqdm(iterator)):
        pass