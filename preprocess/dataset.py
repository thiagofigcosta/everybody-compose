from torch.utils.data import IterableDataset
from preprocess.constants import DATASETS_CONFIG_PATH
from preprocess.prepare import generate_sequences_and_shifted, load_prepared_dataset, prepare_raw_beats_notes
from preprocess.fetch import download

import torch.utils.data
import math
import numpy as np
import toml
import warnings

class BeatsRhythmsDataset(IterableDataset):
    def __init__(self, mono = True, num_files = -1, max_files_to_parse = -1, seq_len = 64, save_freq = 128, indices = []):
        need_process = max_files_to_parse != -1
        if not need_process:
            config = toml.load(DATASETS_CONFIG_PATH)
            self.beats_list, self.notes_list = [], []
            for name, value in config["datasets"].items():
                fp = download(f"{name}.pkl", value["prepared"])
                if fp is None:
                    need_process = True
                    warnings.warn(f"Failed to download {name}.pkl, will process raw data.")
                    break
                b, n = load_prepared_dataset(fp)
                print(f"Downloaded {name}.pkl: {len(b)} files.")
                self.beats_list.extend(b)
                self.notes_list.extend(n)
        if need_process:
            self.beats_list, self.notes_list = prepare_raw_beats_notes(mono, max_files_to_parse, False, progress_save_freq=save_freq)
        self.seq_len = seq_len
        self.num_files = num_files if num_files != -1 else len(self.beats_list)
        if len(indices) > 0:
            self.indices = indices
        else:
            indices = np.arange(len(self.beats_list))
            np.random.seed(0)
            np.random.shuffle(indices)
            self.indices = indices[:self.num_files]

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            for idx in self.indices:
                # here: beats and notes represents beats and notes of one midi file
                beats, notes = self.beats_list[idx], self.notes_list[idx]
                yield from generate_sequences_and_shifted(beats, notes, self.seq_len)
        else:
            # lo, hi = 0, len(self.beats_list)
            # per_worker = int(math.ceil((hi - lo) / float(worker_info.num_workers)))
            # worker_id = worker_info.id
            # worker_lo, worker_hi = lo + worker_id * per_worker, min(lo + (worker_id + 1) * per_worker, hi)
            # for idx in self.indices[worker_lo:worker_hi]:
            #     beats, notes = self.beats_list[idx], self.notes_list[idx]
            #     yield from generate_sequences_shifted(beats, notes, self.seq_len)
            raise NotImplementedError("Multi-process data loading is not supported yet.")
    
    def __len__(self):
        return len(self.indices)


def collate_fn(batch):
    X, y, y_prev = zip(*batch)
    X = np.array(X)
    y = np.array(y)
    y_prev = np.array(y_prev)
    return X, y, y_prev
