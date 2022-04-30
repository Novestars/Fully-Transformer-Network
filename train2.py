import os
import time
import random
import argparse
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from util import GradualWarmupSchedulerV2
from dataset import get_df, get_transforms, MelanomaDataset
from classification import FTN_12

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--kernel-type', type=str, default='FTN12_32_768_12_4')
    parser.add_argument('--data-dir', type=str)
    parser.add_argument('--data-folder', type=int, default='512')
    parser.add_argument('--image-size', type=int, default='384')
    parser.add_argument('--enet-type', type=str, default='FTN_12')
    parser.add_argument('--batch-size', type=int, default=20)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--init-lr', type=float, default=1e-6)
    parser.add_argument('--weight-decay', type=float, default=1e-2)
    parser.add_argument('--out-dim', type=int, default=9)
    parser.add_argument('--n-epochs', type=int, default=4)
    parser.add_argument('--use-amp', action='store_true',default=True)
    parser.add_argument('--use-meta', action='store_true',default=False)
    parser.add_argument('--DEBUG', action='store_true')
    parser.add_argument('--model-dir', type=str, default='./weights')
    parser.add_argument('--log-dir', type=str, default='./logs')
    parser.add_argument('--CUDA_VISIBLE_DEVICES', type=str, default='1')
    parser.add_argument('--fold', type=str, default='0, 1,2,3,4')
    parser.add_argument('--n-meta-dim', type=str, default='512,128')

    args, _ = parser.parse_known_args()
    return args


def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def train_epoch(model, loader, optimizer):
    model.train()
    train_loss = []
    bar = tqdm(loader)
    if args.use_amp:
        scaler = torch.cuda.amp.GradScaler(enabled=True)

    for (data, target) in bar:

        optimizer.zero_grad()

        if args.use_meta:
            data, meta = data
            data, meta, target = data.to(device), meta.to(device), target.to(device)
            logits = model(data, meta)
        else:
            data, target = data.to(device), target.to(device)
            logits = model(data)

        loss = criterion(logits, target)

        if not args.use_amp:
            loss.backward()
            optimizer.step()
        else:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()  # set_to_none=True here can modestly improve performance

        if args.image_size in [896, 576]:
            pass
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)

        loss_np = loss.detach().cpu().numpy()
        train_loss.append(loss_np)
        smooth_loss = sum(train_loss[-100:]) / min(len(train_loss), 100)
        bar.set_description('loss: %.5f, smth: %.5f' % (loss_np, smooth_loss))

    train_loss = np.mean(train_loss)
    return train_loss


def get_trans(img, I):
    if I >= 4:
        img = img.transpose(2, 3)
    if I % 4 == 0:
        return img
    elif I % 4 == 1:
        return img.flip(2)
    elif I % 4 == 2:
        return img.flip(3)
    elif I % 4 == 3:
        return img.flip(2).flip(3)


def val_epoch(model, loader, mel_idx, is_ext=None, n_test=1, get_output=False):
    model.eval()
    val_loss = []
    LOGITS = []
    PROBS = []
    TARGETS = []
    with torch.no_grad():
        for (data, target) in tqdm(loader):

            if args.use_meta:
                data, meta = data
                data, meta, target = data.to(device), meta.to(device), target.to(device)
                logits = torch.zeros((data.shape[0], args.out_dim)).to(device)
                probs = torch.zeros((data.shape[0], args.out_dim)).to(device)
                for I in range(n_test):
                    l = model(get_trans(data, I), meta)
                    logits += l
                    probs += l.softmax(1)
            else:
                data, target = data.to(device), target.to(device)
                logits = torch.zeros((data.shape[0], args.out_dim)).to(device)
                probs = torch.zeros((data.shape[0], args.out_dim)).to(device)
                for I in range(n_test):
                    l = model(get_trans(data, I))
                    logits += l
                    probs += l.softmax(1)
            logits /= n_test
            probs /= n_test

            LOGITS.append(logits.detach().cpu())
            PROBS.append(probs.detach().cpu())
            TARGETS.append(target.detach().cpu())

            loss = criterion(logits, target)
            val_loss.append(loss.detach().cpu().numpy())

    val_loss = np.mean(val_loss)
    LOGITS = torch.cat(LOGITS).numpy()
    PROBS = torch.cat(PROBS).numpy()
    TARGETS = torch.cat(TARGETS).numpy()

    if get_output:
        return LOGITS, PROBS
    else:
        acc = (PROBS.argmax(1) == TARGETS).mean() * 100.
        auc = roc_auc_score((TARGETS == mel_idx).astype(float), PROBS[:, mel_idx])
        auc_20 = roc_auc_score((TARGETS[is_ext == 0] == mel_idx).astype(float), PROBS[is_ext == 0, mel_idx])
        return val_loss, acc, auc, auc_20


def run(fold, df, meta_features, n_meta_features, transforms_train, transforms_val, mel_idx):
    if args.DEBUG:
        args.n_epochs = 5
        df_train = df[df['fold'] != fold].sample(args.batch_size * 5)
        df_valid = df[df['fold'] == fold].sample(args.batch_size * 5)
    else:
        df_train = df[df['fold'] != fold]
        df_valid = df[df['fold'] == fold]

    dataset_train = MelanomaDataset(df_train, 'train', meta_features, transform=transforms_train)
    dataset_valid = MelanomaDataset(df_valid, 'valid', meta_features, transform=transforms_val)
    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size,
                                               sampler=RandomSampler(dataset_train), num_workers=args.num_workers)
    valid_loader = torch.utils.data.DataLoader(dataset_valid, batch_size=args.batch_size, num_workers=args.num_workers)

    model = ModelClass()
    model = model.to(device)

    auc_max = 0.
    auc_20_max = 0.
    model_file = os.path.join(args.model_dir, f'{args.kernel_type}_best_fold{fold}.pth')
    model_file2 = os.path.join(args.model_dir, f'{args.kernel_type}_best_20_fold{fold}.pth')
    model_file3 = os.path.join(args.model_dir, f'{args.kernel_type}_final_fold{fold}.pth')

    optimizer = optim.AdamW(model.parameters(), lr=args.init_lr,weight_decay=args.weight_decay)
    scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, args.n_epochs - 1)
    scheduler_warmup = GradualWarmupSchedulerV2(optimizer, multiplier=10, total_epoch=1,
                                                after_scheduler=scheduler_cosine)

    print(len(dataset_train), len(dataset_valid))

    print('Continuing with model from ' + model_file3)
    try:
        checkpoint = torch.load(model_file)
        model.load_state_dict(checkpoint,strict=False)
    except:
        print('error')
        pass

    for epoch in range(1, args.n_epochs + 1):
        print(time.ctime(), f'Fold {fold}, Epoch {epoch}')
        #         scheduler_warmup.step(epoch - 1)

        train_loss = train_epoch(model, train_loader, optimizer)
        val_loss, acc, auc, auc_20 = val_epoch(model, valid_loader, mel_idx, is_ext=df_valid['is_ext'].values)

        content = time.ctime() + ' ' + f'Fold {fold}, Epoch {epoch}, lr: {optimizer.param_groups[0]["lr"]:.7f}, train loss: {train_loss:.5f}, valid loss: {(val_loss):.5f}, acc: {(acc):.4f}, auc: {(auc):.6f}, auc_20: {(auc_20):.6f}.'
        print(content)
        with open(os.path.join(args.log_dir, f'log_{args.kernel_type}.txt'), 'a') as appender:
            appender.write(content + '\n')

        scheduler_warmup.step()
        if epoch == 2: scheduler_warmup.step()  # bug workaround

        if auc > auc_max:
            print('auc_max ({:.6f} --> {:.6f}). Saving model ...'.format(auc_max, auc))
            torch.save(model.state_dict(), model_file)
            auc_max = auc
        if auc_20 > auc_20_max:
            print('auc_20_max ({:.6f} --> {:.6f}). Saving model ...'.format(auc_20_max, auc_20))
            torch.save(model.state_dict(), model_file2)
            auc_20_max = auc_20
        torch.save({
            'net': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, model_file3)

    #torch.save(model.state_dict(), model_file3)


def main():
    df, df_test, meta_features, n_meta_features, mel_idx = get_df(
        args.kernel_type,
        args.out_dim,
        args.data_dir,
        args.data_folder,
        args.use_meta
    )

    transforms_train, transforms_val = get_transforms(args.image_size)

    folds = [int(i) for i in args.fold.split(',')]
    for fold in folds:
        run(fold, df, meta_features, n_meta_features, transforms_train, transforms_val, mel_idx)

if __name__ == '__main__':
 #   torch.backends.cudnn.benchmark = True

    args = parse_args()
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.CUDA_VISIBLE_DEVICES

    if args.enet_type == 'FTN_12':
        ModelClass = FTN_12
    else:
        raise NotImplementedError()

    set_seed()

    device = torch.device('cuda')
    criterion = nn.CrossEntropyLoss()

    main()