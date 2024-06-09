import argparse
import torch
from torch import optim
import torch.nn as nn
import torch.onnx
from torchsummary import summary
from torch.optim.lr_scheduler import ExponentialLR
from sklearn.metrics import f1_score
from loadData import get_dataloader
from models.model_info import model_dic
from models.WeightedFocalLoss import get_weights, WeightedFocalLoss
from progressbar import ProgressBar, Percentage, Bar, Timer, ETA, FileTransferSpeed


def train(opt: argparse.Namespace, saver):

    trainDataLoader = get_dataloader(
        batch_size=opt.batch_size,
        shuffle=True,
        mode='train',
        num_classes=opt.num_classes,
        num_images=opt.num_images,
        ds_enhance=opt.ds_enhance,
        dl_num_worker=opt.tr_dl_num_worker
    )
    testDataLoader = get_dataloader(
        batch_size=opt.batch_size,
        shuffle=False,
        mode='test',
        num_classes=opt.num_classes,
        num_images=opt.num_images,
        ds_enhance=False,
        dl_num_worker=opt.te_dl_num_worker
    )

    set_model = model_dic[opt.model]

    loss_func = nn.BCEWithLogitsLoss()
    model = set_model(opt).to(opt.device)
    optimizer = optim.Adam(model.parameters(), lr=opt.learning_rate)

    # Initialize the learning rate scheduler
    scheduler = ExponentialLR(optimizer, gamma=opt.lr_gamma)  # Adjust gamma to your needs

    model.train()
    for e in range(opt.epochs):
        total_loss = 0
        total_correct_num = 0
        total_num = 0

        widgets = [f'Epoch [{e + 1}/{opt.epochs}]: ', Percentage(), ' ', Bar('#'), ' ', Timer(),
                   ' ', ETA(), ' ', FileTransferSpeed()]
        pbar = ProgressBar(widgets=widgets, maxval=len(trainDataLoader)).start()

        for batch_id, (x, y) in enumerate(trainDataLoader):
            y_pre = model(x.to(opt.device))

            # -- compute accuracy and loss --
            correct_num = (y_pre.argmax(dim=1) == y.to(opt.device)).sum()

            total_correct_num += correct_num
            loss = loss_func(y_pre, y.to(opt.device))
            total_loss += loss.item() * x.size(0)

            total_num += y.shape[0]
            # -- backward and optimize --
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # -- finish an epoch --
            pbar.update(batch_id + 1)

        pbar.finish()
        acc = total_correct_num / total_num
        Loss = total_loss / total_num

        # Logging
        mf_dic = {'epoch': e + 1, 'loss': Loss, 'acc': acc, 'learning_rate': optimizer.param_groups[0]['lr']}
        saver.save_train_info(mf_dic)

        # Step the scheduler
        scheduler.step()

        model.eval()
        with torch.no_grad():

            total_loss = 0
            total_correct_num = 0
            total_num = 0

            widgets = [f'Evaluating: ', Percentage(), ' ', Bar('#'), ' ', Timer(),
                       ' ', ETA(), ' ', FileTransferSpeed()]

            pbar = ProgressBar(widgets=widgets, maxval=testDataLoader.__len__()).start()
            for batch_id, (x, y) in enumerate(testDataLoader):
                y_pre = model(x.to(opt.device))

                # -- compute accuracy and loss --
                correct_num = (y_pre.argmax(dim=1) == y.to(opt.device)).sum()

                total_correct_num += correct_num

                loss = loss_func(y_pre, y.to(opt.device))
                total_loss += loss.item() * opt.batch_size

                total_num += y.shape[0]
                # -- finish an epoch --
                pbar.update(batch_id)

            pbar.finish()
            acc = total_correct_num / total_num
            Loss = total_loss / total_num

            tf_dic = {'epoch': e + 1, 'loss': Loss, 'acc': acc}

            saver.save_test_info(tf_dic)

    torch.save(model, saver.result_dir + 'model.pt')
