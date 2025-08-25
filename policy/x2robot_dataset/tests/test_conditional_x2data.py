from x2robot_dataset.lazy_dataset import IterChunkDataset, X2RDataProcessingConfig, X2RDataChunkConfig

from pathlib import Path


if __name__ == '__main__':
	data_folders = [
	        './data_builder/x2_cond_folder/x2_cond1.json'
	]
	
	data_configs = [X2RDataProcessingConfig(
	        train_test_split=1.0,
	    ).as_dict() for _ in range(len(data_folders))]
	data_configs[0].update({'class_type':'x2'})
	
	data_chunk_config = X2RDataChunkConfig()
	dataset = IterChunkDataset(
	            data_folders=data_folders, 
	            data_configs=data_configs,
	            data_chunk_config=data_chunk_config,
	            force_overwrite = True,
	            save_meta_data = True,
	            preload_pool_size = 2,
	            num_preloader_threads = 2,
	            max_frame_buffer_size = 100,
	            num_frame_producer_threads = 2,
	            root_dir = Path('./data_builder/.cache')
	    )
	dataset.reset_epoch(0)
	import tqdm
	for data in tqdm.tqdm(dataset, total=dataset.num_frames):
	    if data['uid'] == 'factory10000_20241105-pick_up-sponge@TEACH_ARM@2024_11_05_17_22_08' and data['frame'] == 20:
	        print(data['conditions.text'], data['conditions.image'])
	        break
