import torch.utils.data as data
import numpy as np
import torch


class h36m_dataloader(data.Dataset):
    def __init__(self, data_path, input_len, seq_len):
        self.data = np.load(data_path)
        self.input_len = input_len
        self.seq_len = seq_len

    # def __getitem__(self, item):
    #     item_data = self.data[item]
    #     input = item_data[:self.input_len, :]
    #     last_frame = item_data[self.input_len - 1, :]
    #     last_frame = np.expand_dims(last_frame, axis=0)
    #     last_frame_repeat = np.repeat(last_frame, 10, axis=0)
    #     input = np.concatenate((input, last_frame_repeat), axis=0)
    #     gth = item_data[self.input_len: self.seq_len, :]
    #     return input, gth

    def __getitem__(self, item):
        item_data = self.data[item]
        # input = item_data[:self.input_len, :]
        # last_frame = item_data[self.input_len - 1, :]
        # last_frame = np.expand_dims(last_frame, axis=0)
        # last_frame_repeat = np.repeat(last_frame, 10, axis=0)
        # input = np.concatenate((input, last_frame_repeat), axis=0)
        # gth = item_data[self.input_len: self.seq_len, :]
        return item_data

    def __len__(self):
        return len(self.data)

if __name__ == '__main__':
    train_dataloader = torch.utils.data.DataLoader(
        h36m_dataloader(data_path='data/h36m20_minitrain_3d.npy', input_len=10, seq_len=20),
        batch_size=4, num_workers=1, pin_memory=True)
    for batch_id, (input_data, gth) in enumerate(train_dataloader):
        print('{} {} {} {} {} {}'.format('batch id:', batch_id, 'data type:',
                                         type(input_data), 'input shape:', input_data.shape))








