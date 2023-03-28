import glob
import pytorch_lightning as pl

from weaver.utils.dataset import SimpleIterDataset
from torch.utils.data import DataLoader


class WeaverDataModule(pl.LightningDataModule):
    def __init__(self, train_file, val_file, data_path, data_config_file, batch_size=1024, num_workers=4, fetch_step=0.25, in_memory=False):
        super().__init__()
        self.train_file = train_file
        self.val_file = val_file
        self.data_path = data_path
        self.data_config_file = data_config_file
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.fetch_step = fetch_step
        self.in_memory = in_memory

    @staticmethod
    def create_file_dict(filename,  data_path):
        file_dict = {}
        flist = [filename]
        # Process each file pattern in flist
        for pattern in flist:
            # Add prefix 'a:' or 'b:' to file patterns
            prefix = 'a:' if pattern == flist[0] else 'b:'
    
            full_pattern = prefix + data_path + pattern

            # Separate prefix and file path
            name, file_pattern = full_pattern.split(':')

            # Find all files that match the file pattern
            matched_files = glob.glob(file_pattern)

            # Add matched files to the dictionary
            if name in file_dict:
                file_dict[name] += matched_files
            else:
                file_dict[name] = matched_files

        # Sort the files in the dictionary
        for name, files in file_dict.items():
            file_dict[name] = sorted(files)

        return file_dict

    def prepare_data(self):
        self.train_file_dict = self.create_file_dict(self.train_file, self.data_path)
        self.val_file_dict = self.create_file_dict(self.val_file, self.data_path)

    def setup(self, stage=None):
        # common kwargs for the datasets    
        common_kwargs = {
            'data_config_file': self.data_config_file,
            'load_range_and_fraction': None,
            'extra_selection': None,
            'fetch_by_files': False,
            'fetch_step': self.fetch_step,
            'in_memory': self.in_memory
        }

        self.train_data = SimpleIterDataset(self.train_file_dict, for_training=True, **common_kwargs)
        self.val_data = SimpleIterDataset(self.val_file_dict, for_training=False, **common_kwargs)

        
        # simplify by combining the common kwargs for the datasets 

    def train_dataloader(self):
        return DataLoader(
            self.train_data, batch_size=self.batch_size, drop_last=True, pin_memory=True,
            num_workers=self.num_workers, shuffle=False)
    
    def val_dataloader(self):
        return DataLoader(
            self.val_data, batch_size=self.batch_size, drop_last=True, pin_memory=True,
            num_workers=self.num_workers, shuffle=False)



if __name__ == '__main__':
    import yaml
    with open('config.yaml') as f:
        config = yaml.safe_load(f)['data']

    dm = WeaverDataModule(
        train_file=config['train_file'],
        val_file=config['val_file'],
        data_path=config['data_path'],
        data_config_file=config['data_config_file'],
        batch_size=config['batch_size'],
        num_workers=1,
        fetch_step=1.0,
        in_memory=False,
    )

    common_kwargs = {
            'data_config_file': dm.data_config_file,
            'load_range_and_fraction': None,
            'extra_selection': None,
            'fetch_by_files': False,
            'fetch_step': dm.fetch_step,
            'in_memory': dm.in_memory
        }

    import time
    start = time.time()
    # create file dict
    dm.prepare_data()

    # time with the datamodule
    start = time.time()
    dm.setup()
    batch = next(iter(dm.val_dataloader()))
    X, y, _ = batch
    #print batch size
    print(X['pf_features'].shape)
    print(f'Time taken: ', time.time() - start)
    
