import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

import pandas as pd
import time
import datetime

from transformers import ElectraForSequenceClassification, ElectraTokenizer
from transformers import AdamW, get_linear_schedule_with_warmup

# load model and tokenizer
model = ElectraForSequenceClassification.from_pretrained("monologg/koelectra-small-discriminator")
tokenizer = ElectraTokenizer.from_pretrained("monologg/koelectra-small-discriminator")

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

def load_dataset():
    # load nsmc dataset
    nsmc_train = pd.read_csv('../../data/nsmc/ratings_train.txt', sep='\t', encoding='utf-8')
    nsmc_test = pd.read_csv('../../data/nsmc/ratings_test.txt', sep='\t', encoding='utf-8')

    # nsmc_train = nsmc_train[:5000]
    # nsmc_test = nsmc_test[:1000]
    nsmc_train['document'] = nsmc_train['document'].apply(str)
    nsmc_test['document'] = nsmc_test['document'].apply(str)

    # encoding
    train_encodings = tokenizer(list(nsmc_train['document']), truncation=True, padding=True)
    test_encodings = tokenizer(list(nsmc_test['document']), truncation=True, padding=True)

    train_dataset = NSMCDataset(train_encodings, nsmc_train['label'])
    test_dataset = NSMCDataset(test_encodings, nsmc_test['label'])

    return train_dataset, test_dataset

class trainer():
    def __init__(self,
                 model,
                 optimizer,
                 scheduler,
                 train_loader,
                 test_loader,
                 device,
                 ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.writer = SummaryWriter('./runs/SingleGPU_NSMC')

    def train(self, epochs):
        train_loss = 0
        correct = 0
        total = 0

        self.model.train()
        self.model.to(self.device)

        total_epoch_start = time.time()
        print('training is begin!')
        epoch_start = time.time()
        for epoch in range(epochs):
            print(f'training epoch {epoch}')
            for batch_idx, inputs in enumerate(self.train_loader):
                start = time.time()
                inputs = {k: v.cuda(self.device) for k, v in inputs.items()}
                outputs = model(**inputs)

                self.optimizer.zero_grad()
                outputs.loss.backward()
                self.optimizer.step()
                self.scheduler.step()

                train_loss += outputs.loss.item()
                total += inputs['labels'].size(0)
                correct += inputs['labels'].eq(outputs.logits.argmax(axis=1)).sum().item()

                acc = 100 * correct / total
                batch_time = time.time() - start

                if batch_idx % 200 == 0:
                    print('=================== Training =======================')
                    print(f'Epoch: {epoch} \n'
                          f'total_steps: {epoch * len(self.train_loader) + batch_idx} \n'
                          f'loss: {train_loss / (batch_idx+1):.3f} \n'
                          f'acc : {acc:.3f} \n'
                          f'batch_time : {batch_time} \n'
                          )
                    self.writer.add_scalar('Loss/train',
                                           outputs.loss.item(),
                                           epoch * len(self.train_loader) + batch_idx)
                if batch_idx % 400 == 0:
                    test_acc, test_loss = self.test()
                    print('=================== Validation ==========================')
                    print(f'Epoch: {epoch} \n'
                          f'total_steps: {epoch * len(self.train_loader) + batch_idx} \n'
                          f'loss: {test_loss:.3f} \n'
                          f'test_acc : {acc:.3f} \n'
                          )
                    self.writer.add_scalar('Loss/test',
                                           test_loss,
                                           epoch * len(self.train_loader) + batch_idx
                                           )

            elapse_time = time.time() - epoch_start
            elapse_time = datetime.timedelta(seconds=elapse_time)
            print(f"Training time per batch: {elapse_time}")

        elapse_time = time.time() - total_epoch_start
        elapse_time = datetime.timedelta(seconds=elapse_time)
        print(f"Training time {epochs} batch: {elapse_time}")

    def test(self):
        self.model.eval()
        total = 0
        total_loss = 0
        correct = 0
        with torch.no_grad():
            for inputs in self.test_loader:
                start = time.time()
                inputs = {k: v.cuda(self.device) for k, v in inputs.items()}
                outputs = model(**inputs)

                total += inputs['labels'].size(0)
                correct += inputs['labels'].eq(outputs.logits.argmax(axis=1)).sum().item()
                total_loss += outputs.loss

                acc = 100 * correct / total
                batch_time = time.time() - start

            # print(f'test time : {batch_time}\n'
            #       f'total_loss : {total_loss / total:.3f}')
        self.model.train()
        return acc, total_loss / total


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='cifar10 classification models')
    parser.add_argument('--epoch', default=10, help='')
    parser.add_argument('--lr', default=5e-5, help='')
    parser.add_argument('--resume', default=None, help='')
    parser.add_argument('--batch_size', type=int, default=16, help='')
    parser.add_argument('--num_workers', type=int, default=8, help='')
    parser.add_argument("--gpu_devices", type=int, nargs='+', default=None, help="")

    parser.add_argument('--gpu', default=None, type=int, help='GPU id to use.')
    parser.add_argument('--dist-url', default='tcp://127.0.0.1:3456', type=str, help='')
    parser.add_argument('--dist-backend', default='nccl', type=str, help='')
    parser.add_argument('--rank', default=0, type=int, help='')
    parser.add_argument('--world_size', default=1, type=int, help='')
    parser.add_argument('--distributed', action='store_true', help='')
    args = parser.parse_args()

    train_dataset, test_dataset = load_dataset()

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr, )
    scheduler = get_linear_schedule_with_warmup(optimizer, len(train_dataloader) * args.epoch / 4, len(train_dataloader)*args.epoch)
    trainer = trainer(model, optimizer=optimizer, train_loader=train_dataloader, test_loader=test_dataloader, scheduler=scheduler, device='cuda')

    trainer.train(args.epoch)