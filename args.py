import argparse

parser = argparse.ArgumentParser()
# parser.add_argument('--device', default='cpu', type=str)
parser.add_argument('--device', default='cuda', type=str)
parser.add_argument('--train_data_path', default='data/h36m20_val_3d.npy', type=str)
# parser.add_argument('--train_data_path', default='data/h36m20_1batch.npy', type=str)
parser.add_argument('--epochs', default=100, type=int)
# parser.add_argument('--epochs', default=1, type=int)
parser.add_argument('--batch_size', default=16, type=int)
# parser.add_argument('--batch_size', default=4, type=int)
parser.add_argument('--input_len', default=10, type=int)
parser.add_argument('--seq_len', default=20, type=int)

parser.add_argument('--joints_input', default=22, type=int)
parser.add_argument('--lr', default=1e-4)
parser.add_argument('--interval', default=400)
parser.add_argument('--val_root', default='data/test20_npy/', type=str)
parser.add_argument('--val_batch_size', default=8, type=int)
parser.add_argument('--joints_total', default=32, type=int)
parser.add_argument('--tb_log_dir', default='tflog', type=str)
parser.add_argument('--log_name', default='log/8.27.1.txt', type=str)

