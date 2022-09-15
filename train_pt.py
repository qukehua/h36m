from model_pt import TrajectoryNet
from dataloader_pt import h36m_dataloader
import torch
import torch.optim as optim
import time
import numpy as np
import recoverh36m_3d
from args import parser
import torch.nn as nn
import os
from time import strftime, localtime


def train(model, train_dataloader, epoch):
    for batch_id, load_data in enumerate(train_dataloader):
        # input_data = input_data[:, :, 0:args.joints_input, :]
        # gth = gth[:, :, 0:args.joints_input, :]
        # gth = torch.tensor(gth, dtype=torch.float32).to(device)
        load_data = torch.tensor(load_data, dtype=torch.float32)
        # input = input.to(device)
        load_data = load_data[:, :, 0:args.joints_input, :]
        input_seq = load_data[:, :args.input_len, :, :]
        # last_frame = load_data[:, args.input_len - 1, :]
        # last_frame = last_frame.unsqueeze(1)
        # last_frame_repeat = last_frame.repeat(1, 10, 1, 1)
        # input_data = torch.cat((input_seq, last_frame_repeat), 1)
        gth = load_data[:, args.input_len: args.seq_len, :, :]

        input_data = input_seq.to(device)
        gth = gth.to(device)
        out = model(input_data)
        x = out.data.cpu().numpy().copy()
        np.save(os.path.join('pred', str(epoch) + '.npy'), x)
        loss = 0
        loss_input = out - gth
        # loss_input = loss_input.type(torch.DoubleTensor)
        loss = torch.mean(torch.norm(loss_input, p=2, dim=3, keepdim=True))
        optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.95, 0.9995))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # load_data = load_data.detach().numpy()
        # load_data_r = load_data[:, ::-1]
        # load_data_r = np.ascontiguousarray(load_data_r)
        # load_data_r = torch.from_numpy(load_data_r)
        # input_seq_r = load_data_r[:, :args.input_len, :, :]
        # last_frame_r = load_data_r[:, args.input_len - 1, :]
        # last_frame_r = last_frame_r.unsqueeze(1)
        # last_frame_r_repeat = last_frame_r.repeat(1, 10, 1, 1)
        # input_data_r = torch.cat((input_seq_r, last_frame_r_repeat), 1)
        # gth_r = load_data_r[:, args.input_len: args.seq_len, :, :]
        # input_data_r = input_data_r.to(device)
        # gth_r = gth_r.to(device)
        # out_r, p_r = model(input_data_r)
        # loss_r = 0
        # loss_input_r = out_r - gth_r
        # loss_r = torch.mean(torch.norm(loss_input_r, p=2, dim=3, keepdim=True))
        #  optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.95, 0.9995))
        # print(type(model.parameters()))
        #  optimizer.zero_grad()
        # loss_r.backward()
        #  optimizer.step()
        # tf_writer.add_scalar('loss', (loss.item()+loss_r.item())/2, batch_id+epoch*len(train_dataloader))

        if batch_id % args.interval == 0:
            file = open(args.log_name, 'a+')
            print('{} {} {} {}'.format('batch:', batch_id, 'epoch:', epoch))
            file.write('{} {} {} {}\n'.format('batch:', batch_id, 'epoch:', epoch))
            print('{} {} {} {}'.format('total:', len(train_dataloader), 'loss:', (loss.item())))
            file.write('{} {} {} {}\n'.format('total:', len(train_dataloader), 'loss:', (loss.item())))
            print('{} {}'.format('cost time:', time.time() - start_time))
            file.write('{} {}\n'.format('cost time:', time.time() - start_time))
            str1 = 'walking eating smoking discussion directions greeting phoning posing purchases sitting sittingdown ' \
                   'takingphoto waiting walkingdog walkingtogether'
            actions = str1.split(' ')
            mpjpe = np.zeros((1, 10))
            mpjpe_v = np.zeros((1, 10))
            for action in actions:
                # print('{} {} {}'.format('action', action, 'mpjpe:'))
                val_dataloader = torch.utils.data.DataLoader(
                    h36m_dataloader(data_path=args.val_root + action + '.npy', input_len=args.input_len,
                                    seq_len=args.seq_len),
                    batch_size=args.val_batch_size, num_workers=1, pin_memory=True)
                mpjpe_x, mpjpe_x_v = evaluate(model, val_dataloader, action, file)
                mpjpe += mpjpe_x
                mpjpe_v += mpjpe_x_v
            mpjpe_ave = mpjpe / len(actions)
            mpjpe_ave_v = mpjpe_v / len(actions)
            print('mean mpjpe:')
            file.write('mean mpjpe:\n')
            for i in [1, 3, 7, 9]:
                print(mpjpe_ave[0][i])
                # tf_writer.add_scalar('mpjpe'+str(i), mpjpe_ave[0][i], batch_id+epoch*len(train_dataloader))
                file.write(str(mpjpe_ave[0][i]) + '\n')

            print('mean mpjpe_v:')
            file.write('mean mpjpe_v:\n')
            for i in [1, 3, 7, 9]:
                print(mpjpe_ave_v[0][i])
                # tf_writer.add_scalar('mpjpe_v' + str(i), mpjpe_ave_v[0][i], batch_id + epoch * len(train_dataloader))
                file.write(str(mpjpe_ave_v[0][i]) + '\n')

            # print(mpjpe_ave)
            mpjpe_train = np.zeros((1, 10))

            for i in range(10):
                x = out[:, i, :, :].cpu().detach().numpy()
                x_gt = gth[:, i, :, :].cpu().detach().numpy()
                for j in range(args.batch_size):
                    tem1 = 0
                    for k in range(args.joints_input):
                        tem1 += np.sqrt(np.square(x[j, k] - x_gt[j, k]).sum())
                    mpjpe_train[0, i] += tem1 / args.joints_input
            mpjpe_train = mpjpe_train / args.batch_size
            print('train mpjpe:')
            for i in [1, 3, 7, 9]:
                print(mpjpe_train[0][i])
                file.write(str(mpjpe_train[0][i]) + '\n')

            state = {'epoch': epoch + 1,
                      'lr': args.lr,
                      'state_dict': model.state_dict(),
                      'optimizer': optimizer.state_dict()}
            torch.save(state,
                       save_model_root + strftime("%m-%d-%H_%M-", localtime()) + str(epoch + epoch_pretrained) + '.pth')

            file.close()


def evaluate(model, val_dataloader, action, file):
    model.eval()
    mpjpe1 = np.zeros((1, 10))
    mpjpev = np.zeros((1, 10))
    for batch_id, load_data in enumerate(val_dataloader):
        # input_data = input_data.to(device)
        # gth = gth.to(device)
        # input_data = input_data[:, :, 0:args.joints_input, :]
        # input_data = torch.tensor(input_data, dtype=torch.float32)
        load_data = torch.tensor(load_data, dtype=torch.float32)
        gth = load_data[:, args.input_len:, :, :]
        load_data = load_data[:, :, 0:args.joints_input, :]
        input_seq = load_data[:, :args.input_len, :, :]
        # last_frame = load_data[:, args.input_len - 1, :]
        # last_frame = last_frame.unsqueeze(1)
        # last_frame_repeat = last_frame.repeat(1, 10, 1, 1)
        # input_data = torch.cat((input_seq, last_frame_repeat), 1)
        input_data = input_seq.to(device)
        out = model(input_data)
        gth = gth.cpu().detach().numpy()
        out = out.cpu().detach().numpy()
        out = recoverh36m_3d.recoverh36m_3d(gth, out)
        out_vel = np.zeros((out.shape))
        gth_vel = np.zeros((out.shape))
        for i in range(10):
            x = out[:, i, :, :]
            if i == 0:
                continue
            else:
                out_vel[:, i, :, :] = out[:, i, :, :] - out[:, i - 1, :, :]
            x_v = out_vel[:, i, :, :]
            x_gt = gth[:, i, :, :]
            if i == 0:
                continue
            else:
                gth_vel[:, i, :, :] = gth[:, i, :, :] - gth[:, i - 1, :, :]
            x_gt_v = gth_vel[:, i, :, :]
            for j in range(args.val_batch_size):
                tem1 = 0
                for k in range(args.joints_total):
                    tem1 += np.sqrt(np.square(x[j, k] - x_gt[j, k]).sum())
                mpjpe1[0, i] += tem1 / args.joints_total

            for j in range(args.val_batch_size):
                tem_v = 0
                for k in range(args.joints_total):
                    tem_v += np.sqrt(np.square(x_v[j, k] - x_gt_v[j, k]).sum())
                mpjpev[0, i] += tem_v / args.joints_total
        mpjpe1 = mpjpe1 / args.val_batch_size
        mpjpev = mpjpev / args.val_batch_size
        # for i in [1, 3, 7, 9]:
        #     print(mpjpe1[0][i])

    return mpjpe1[0], mpjpev[0]


if __name__ == '__main__':
    args = parser.parse_args()
    start_time = time.time()
    file = open(args.log_name, 'w+')
    # tf_writer = SummaryWriter(log_dir='tflog/')
    device = args.device
    train_dataloader = torch.utils.data.DataLoader(
        h36m_dataloader(data_path=args.train_data_path, input_len=args.input_len, seq_len=args.seq_len),
        batch_size=args.batch_size, num_workers=1, pin_memory=True, shuffle=True)
    model = TrajectoryNet().to(device)
    epochs = args.epochs

    save_model_root = './save_model/'
    model_pretrained = save_model_root + '09-15-15_23-0.pth'
    epoch_pretrained = os.path.splitext(model_pretrained)
    print(epoch_pretrained)
    epoch_pretrained = epoch_pretrained[0].split('-')[3]
    print(epoch_pretrained)
    epoch_pretrained = int(epoch_pretrained)
    print(epoch_pretrained)
    # epoch_pretrained = 0
    #state_dict = torch.load(model_pretrained, map_location=torch.device(device))
    model = torch.nn.DataParallel(model)
    cudnn.benchmark = True
    model.load_state_dict(torch.load(model_pretrained)['state_dict'])
   # optimizer.load_state_dict(torch.load(model_pretrained)['optimizer'])

    #model.load_state_dict(state_dict["shared_layers"],strict=False)
    #print(model.state_dict()['TB_foward_0.0.weight'])
    model = torch.nn.DataParallel(model).to(device)

    for epoch in range(epochs):
        train(model, train_dataloader, epoch)
        if epoch % 1 == 0:
          # save_state = {'shared_layers': model.state_dict()}
           state = {'epoch': epoch + 1,
                  'lr': args.lr,
                  'state_dict': model.state_dict(),
                  'optimizer': optimizer.state_dict()}
           torch.save(save_state,
                       save_model_root + strftime("%m-%d-%H_%M-", localtime()) + str(epoch + epoch_pretrained) + '.pth')
           print('model saving done!')
    #save_state_end = {'shared_layers': model.state_dict()}
   # torch.save(save_state_end,
              # save_model_root + strftime("%m-%d-%H:%M-", localtime()) + str(args.epochs + epoch_pretrained) + '.pth')
    #print('last model saving done!')
    # file.close()

