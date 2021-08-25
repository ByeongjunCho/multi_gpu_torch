import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist

import apex
from apex.parallel import DistributedDataParallel as DDP
from apex.fp16_utils import *
from apex import amp, optimizers
from apex.multi_tensor_apply import multi_tensor_applier

import pandas as pd
import time
import datetime
import os

from transformers import ElectraForSequenceClassification, ElectraTokenizer
from transformers import AdamW, get_linear_schedule_with_warmup

# load model and tokenizer
def get_model():
    model = ElectraForSequenceClassification.from_pretrained("monologg/koelectra-small-discriminator")
    tokenizer = ElectraTokenizer.from_pretrained("monologg/koelectra-small-discriminator")
    return model, tokenizer

# NSMC dataset class
class NSMCDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def load_dataset(tokenizer):
    # load nsmc dataset
    nsmc_train = pd.read_csv('../../data/nsmc/ratings_train.txt', sep='\t', encoding='utf-8')
    nsmc_test = pd.read_csv('../../data/nsmc/ratings_test.txt', sep='\t', encoding='utf-8')

    nsmc_train = nsmc_train[:10000]
    nsmc_test = nsmc_test[:1000]
    nsmc_train['document'] = nsmc_train['document'].apply(str)
    nsmc_test['document'] = nsmc_test['document'].apply(str)
    # encoding
    # train_encodings = tokenizer(list(map(str, nsmc_train['document'])), truncation=True, padding=True)
    # test_encodings = tokenizer(list(map(str, nsmc_test['document'])), truncation=True, padding=True)
    train_encodings = tokenizer(list(nsmc_train['document']), truncation=True, padding=True)
    test_encodings = tokenizer(list(nsmc_test['document']), truncation=True, padding=True)

    train_dataset = NSMCDataset(train_encodings, nsmc_train['label'])
    test_dataset = NSMCDataset(test_encodings, nsmc_test['label'])

    return train_dataset, test_dataset


def train(epoch, model, train_loader, optimizer, scheduler, device, writer):
    '''

    :param epoch: current epoch for summary writer
    :return:
    '''
    train_loss = 0
    correct = 0
    total = 0

    model.train()

    for batch_idx, inputs in enumerate(train_loader):
        start = time.time()
        inputs = {k: v.cuda(device) for k, v in inputs.items()}
        outputs = model(**inputs)

        optimizer.zero_grad()
        outputs.loss.backward()
        optimizer.step()
        scheduler.step()

        train_loss += outputs.loss.item()
        total += inputs['labels'].size(0)
        correct += inputs['labels'].eq(outputs.logits.argmax(axis=1)).sum().item()

        acc = 100 * correct / total
        batch_time = time.time() - start

        if args.rank == 0:
            writer.add_scalar('Loss/train', outputs.loss.item(), epoch * len(train_loader) +batch_idx)
            writer.add_scalars('Loss', {'train_loss': outputs.loss.item()}, epoch * len(train_loader) +batch_idx)
            if batch_idx % 200 == 0:
                print('=================== Training =======================')
                print(f'total_steps: {batch_idx} \n'
                      f'loss: {train_loss / (batch_idx+1):.3f} \n'
                      f'acc : {acc:.3f} \n'
                      f'batch_time : {batch_time} \n'
                      )

            if batch_idx % 400 == 0:
                acc, loss = test()
                writer.add_scalar('Loss/test', loss,
                                       epoch * len(train_loader) + batch_idx)
                writer.add_scalar('Accuracy/test', acc, epoch * len(train_loader) + batch_idx)
                writer.add_scalars('Loss', {'test_loss': loss},
                                       epoch * len(train_loader) + batch_idx)

def test(model, test_loader, device):
    model.eval()
    total = 0
    total_loss = 0
    correct = 0
    with torch.no_grad():
        for inputs in test_loader:
            start = time.time()
            inputs = {k: v.cuda(device) for k, v in inputs.items()}
            outputs = model(**inputs)

            total += inputs['labels'].size(0)
            correct += inputs['labels'].eq(outputs.logits.argmax(axis=1)).sum().item()
            total_loss += outputs.loss

            acc = 100 * correct / total
            batch_time = time.time() - start

            print(f'test time : {batch_time}')

    return acc, total_loss / total

def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)


def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu
    ngpus_per_node = torch.cuda.device_count()

    print("Use GPU: {} for training".format(args.gpu))

    args.rank = args.rank * ngpus_per_node + gpu
    dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                            world_size=args.world_size, rank=args.rank)

    print('==> Making model..')
    model, tokenizer = get_model()

    ############################################################
    # For multiprocessing distributed, DistributedDataParallel constructor
    # should always set the single device scope, otherwise,
    # DistributedDataParallel will use all available devices.
    torch.cuda.set_device(gpu)
    model.cuda(gpu)

    # When using a single GPU per process and per
    # DistributedDataParallel, we need to divide the batch size
    # ourselves based on the total number of GPUs we have
    args.batch_size = int(args.batch_size / ngpus_per_node)  # calculate local batch size for each GPU
    args.num_workers = int(args.num_workers / ngpus_per_node)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu])
    #############################################################

    # Data loading code
    train_dataset, test_dataset = load_dataset(tokenizer)

    ############################################################
    # makes sure that each process gets a different slice of the training data
    # during distributed training
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)

    # notice we turn off shuffling and use distributed data sampler
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.num_workers, pin_memory=True, sampler=train_sampler)

    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True)


    # define optimizer and scheduler

    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(args.epochs * len(train_loader)/4), args.epochs*len(train_loader))

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('The number of parameters of model is', num_params)

    end = time.time()

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            # Map model to be loaded to specified single gpu.
            loc = 'cuda:{}'.format(gpu)
            checkpoint = torch.load(args.resume, map_location=loc)
            start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    else:
        start_epoch = 0


    trainer = Trainer(model, optimizer, scheduler, train_loader, test_loader, device=gpu, args=args)
    for epoch in range(start_epoch, args.epochs):
        train_sampler.set_epoch(epoch)

        # train for one epoch
        trainer.train()

        # validation
        acc, loss = trainer.test()

        if args.rank == 0:
            trainer.save_checkpoint({
                'epoch': epoch + 1,
                'accuracy': acc,
                'loss': loss,
                'state_dict': trainer.model.state_dict(),
                'optimizer': trainer.optimizer.state_dict(),
                'scheduler': trainer.scheduler.state_dict()
            }, filename=f'./saved_models/epoch_{epoch}_{loss:.3f}_accuracy_{acc:.3f}.pt')

def main():
    args = parser.parse_args()

    ############################################################
    ngpus_per_node = torch.cuda.device_count()
    # on each node we have: ngpus_per_node processes and ngpus_per_node gpus
    # that is, 1 process for each gpu on each node.
    # world_size is the total number of processes to run
    args.world_size = ngpus_per_node * args.world_size

    # Use torch.multiprocessing.spawn to launch distributed processes: the
    # main_worker process function
    mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    ############################################################

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='NSMC classification models')
    parser.add_argument('--epochs', default=10, help='')
    parser.add_argument('--arch', default='ELECTRA_NSMC', help='model architecture')
    parser.add_argument('--lr', default=5e-5, help='')
    parser.add_argument('--resume', default=None, help='')
    parser.add_argument('--batch_size', type=int, default=256, help='')
    parser.add_argument('--num_workers', type=int, default=4, help='')
    parser.add_argument("--gpu_devices", nargs='+', default=[4,5,6,7], help="")

    parser.add_argument('--gpu', default=None, type=int, help='GPU id to use.')
    parser.add_argument('--dist-url', default='tcp://127.0.0.1:3456', type=str, help='')
    parser.add_argument('--dist-backend', default='nccl', type=str, help='')
    parser.add_argument('--rank', default=0, type=int, help='')
    parser.add_argument('--world_size', default=1, type=int, help='')
    parser.add_argument('--distributed', action='store_true', help='')
    args = parser.parse_args()

    gpu_devices = ','.join([str(id) for id in args.gpu_devices])
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_devices
    main()
