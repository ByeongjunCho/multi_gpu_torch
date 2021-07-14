import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
import torch.multiprocessing as mp

import pandas as pd
import time
import datetime
import os
from IPython.display import clear_output
from transformers import BertTokenizerFast, BertForSequenceClassification
from transformers import AdamW, get_linear_schedule_with_warmup

import time
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

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

class Args():
    def __init__(self):
        self.epochs = 5
        self.arch = 'BERT_NSMC_singleGPU'
        self.lr = 5e-5
        self.batch_size = 64
        self.gpu = 1

if __name__ == '__main__':
    model = BertForSequenceClassification.from_pretrained("kykim/bert-kor-base")
    tokenizer = BertTokenizerFast.from_pretrained("kykim/bert-kor-base")

    args = Args()

    # load nsmc dataset
    nsmc_train = pd.read_csv('./nsmc/ratings_train.txt', sep='\t', encoding='utf-8')
    nsmc_test = pd.read_csv('./nsmc/ratings_test.txt', sep='\t', encoding='utf-8')

    # slicing dataset
    nsmc_train = nsmc_train[:10000]
    nsmc_test = nsmc_test[:2000]

    nsmc_train['document'] = nsmc_train['document'].apply(str)
    nsmc_test['document'] = nsmc_test['document'].apply(str)

    # encoding
    train_encodings = tokenizer(list(nsmc_train['document']), truncation=True, padding=True)
    test_encodings = tokenizer(list(nsmc_test['document']), truncation=True, padding=True)

    train_dataset = NSMCDataset(train_encodings, nsmc_train['label'])
    test_dataset = NSMCDataset(test_encodings, nsmc_test['label'])

    # dataloader
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    # scheduler reference => https://huggingface.co/transformers/main_classes/optimizer_schedules.html#transformers.get_linear_schedule_with_warmup
    # using scheduler
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr)
    scheduler = get_linear_schedule_with_warmup(optimizer, len(train_loader) * args.epochs // 8,
                                                len(train_loader) * args.epochs)

    # training
    # Single GPU

    torch.cuda.empty_cache()

    writer = SummaryWriter(f'./runs/{args.arch}')
    print_train = len(train_loader) // 10
    total_epoch_start = time.time()

    # for load model
    best_acc = 0
    best_model_name = ''

    # for plot train and test loss
    train_iter_list = []
    train_loss_list = []
    train_acc_list = []
    test_iter_list = []
    test_loss_list = []
    test_acc_list = []
    for epoch in range(args.epochs):
        model.train()
        model.cuda(args.gpu)

        train_loss = 0
        correct = 0
        total = 0
        print('====================================================')
        print('=================== Training =======================')
        print('====================================================')

        epoch_start = time.time()
        for idx, batch in enumerate(train_loader):
            start = time.time()
            inputs = {k: v.cuda(args.gpu) for k, v in batch.items()}
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

            train_iter_list.append(epoch * len(train_loader) + idx)
            train_loss_list.append(outputs.loss.item())
            train_acc_list.append(acc)
            if idx % print_train == 0:
                print(f'Epoch: {epoch} \n'
                      f'total_steps: {epoch * len(train_loader) + idx} \n'
                      f'loss: {train_loss / (idx + 1):.3f} \n'
                      f'acc : {acc:.3f} \n'
                      f'batch_time : {batch_time} \n'
                      )
            writer.add_scalar('Loss/train',
                              outputs.loss.item(),
                              epoch * len(train_loader) + idx)
            writer.add_scalars('Loss',
                               {'Train_Loss': outputs.loss.item()},
                               epoch * len(train_loader) + idx)
            writer.add_scalar('Accuracy/train',
                              acc,
                              epoch * len(train_loader) + idx)

        # test
        test_loss = 0
        correct = 0
        total = 0
        model.eval()
        elapse_time = time.time() - epoch_start
        elapse_time = datetime.timedelta(seconds=elapse_time)
        print(f"Epoch training: {elapse_time}")
        clear_output(wait=True)
        # test
        print('===================================================')
        print('=================== test ==========================')
        print('===================================================')

        with torch.no_grad():
            for inputs in test_loader:
                start = time.time()
                inputs = {k: v.cuda(args.gpu) for k, v in inputs.items()}
                outputs = model(**inputs)

                total += inputs['labels'].size(0)
                correct += inputs['labels'].eq(outputs.logits.argmax(axis=1)).sum().item()
                test_loss += outputs.loss.item()
                acc = 100 * correct / total

        test_loss = test_loss / len(test_loader)

        # write test result
        test_iter_list.append(epoch * len(train_loader) + idx)
        test_loss_list.append(test_loss)
        test_acc_list.append(acc)
        print(f'Epoch: {epoch} \n'
              f'total_steps: {epoch * len(train_loader) + idx} \n'
              f'loss: {test_loss:.3f} \n'
              f'test_acc : {acc:.3f} \n'
              )
        writer.add_scalar('Loss/test',
                          test_loss,
                          epoch * len(train_loader) + idx
                          )
        writer.add_scalars('Loss',
                           {'Test_Loss': test_loss},
                           epoch * len(train_loader) + idx
                           )
        writer.add_scalar('Accuracy/test',
                          acc,
                          epoch * len(train_loader) + idx)

        print(f'Epoch {epoch} finished!. Save model')
        os.makedirs(f'./{args.arch}', exist_ok=True)
        # model parameter save per epoch.
        torch.save(model.state_dict(), f'./{args.arch}/{epoch}_{test_loss:.3f}_{acc:.3f}.pt')
        if best_acc < acc:
            best_acc = acc
            best_model_name = f'./{args.arch}/{epoch}_{test_loss:.3f}_{acc:.3f}.pt'

    print(f'Total training time: {time.time() - total_epoch_start}')