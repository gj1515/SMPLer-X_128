import random
import numpy as np
from torch.utils.data.dataset import Dataset
from config import cfgM


# Fixed by SH Heo (260108) - Print dataset info for train/valid datasets
def print_dataset_info(infos, split_name='Train'):
    """Print dataset information summary.

    Args:
        infos: List of dataset_info dicts
        split_name: 'Train' or 'Valid'
    """
    line = '=' * 80
    print(line)
    print(f'[{split_name} Datasets]')

    total_original, total_sampled, total_final = 0, 0, 0
    for info in infos:
        print(f"  - {info['name']}: original {info['original']}, "
              f"sampled {info['sampled']}, {split_name.lower()} {info['final']} "
              f"(interval {info['sample_interval']})")
        total_original += info['original']
        total_sampled += info['sampled']
        total_final += info['final']

    print(f"  - Total: original {total_original}, sampled {total_sampled}, {split_name.lower()} {total_final}")
    print(line)


class MultipleDatasets(Dataset):
    def __init__(self, dbs, make_same_len=True, total_len=None, verbose=False):
        self.dbs = dbs
        self.db_num = len(self.dbs)
        self.max_db_data_num = max([len(db) for db in dbs])
        self.db_len_cumsum = np.cumsum([len(db) for db in dbs])
        self.make_same_len = make_same_len
        
        if total_len == 'auto':
            self.total_len = self.db_len_cumsum[-1]
            self.auto_total_len = True
        else:
            self.total_len = total_len
            self.auto_total_len = False

        if total_len is not None:
            self.per_db_len = self.total_len // self.db_num
        if verbose:
            print('datasets:', [len(self.dbs[i]) for i in range(self.db_num)])
            print(f'Auto total length: {self.auto_total_len}, {self.total_len}')

            

    def __len__(self):
        # all dbs have the same length
        if self.make_same_len:
            if self.total_len is None:
                # match the longest length
                return self.max_db_data_num * self.db_num
            else:
                # each dataset has the same length and total len is fixed
                return self.total_len
        else:
            # each db has different length, simply concat
            return sum([len(db) for db in self.dbs])

    def __getitem__(self, index):
        if self.make_same_len:
            if self.total_len is None:
                # match the longest length
                db_idx = index // self.max_db_data_num
                data_idx = index % self.max_db_data_num 
                if data_idx >= len(self.dbs[db_idx]) * (self.max_db_data_num // len(self.dbs[db_idx])): # last batch: random sampling
                    data_idx = random.randint(0,len(self.dbs[db_idx])-1)
                else: # before last batch: use modular
                    data_idx = data_idx % len(self.dbs[db_idx])
            else:
                db_idx = index // self.per_db_len 
                data_idx = index % self.per_db_len 
                if db_idx > (self.db_num - 1):
                    # last batch: randomly choose one dataset
                    db_idx = random.randint(0,self.db_num - 1)

                if len(self.dbs[db_idx]) < self.per_db_len  and \
                        data_idx >= len(self.dbs[db_idx]) * (self.per_db_len  // len(self.dbs[db_idx])): 
                    # last batch: random sampling in this dataset
                    data_idx = random.randint(0,len(self.dbs[db_idx]) - 1)
                else: 
                    # before last batch: use modular
                    data_idx = data_idx % len(self.dbs[db_idx])


        else:
            for i in range(self.db_num):
                if index < self.db_len_cumsum[i]:
                    db_idx = i
                    break
            if db_idx == 0:
                data_idx = index
            else:
                data_idx = index - self.db_len_cumsum[db_idx-1]

        return self.dbs[db_idx][data_idx]
