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

    trainDataLoader, label_statistics = get_dataloader(
        batch_size=opt.batch_size,
        shuffle=True,
        mode='train',
        num_classes=opt.num_classes,
        num_images=opt.num_images,
        ds_enhance=opt.ds_enhance,
        dl_num_worker=opt.tr_dl_num_worker
    )
    print(label_statistics)
    testDataLoader = get_dataloader(
        batch_size=opt.batch_size,
        shuffle=False,
        mode='test',
        num_classes=opt.num_classes,
        num_images=opt.num_images,
        ds_enhance=False,
        dl_num_worker=opt.te_dl_num_worker
    )

    class_weights = get_weights(label_statistics).to(opt.device)
    criterion = WeightedFocalLoss(weights=class_weights, gamma=2.0, reduction='mean').to(opt.device)
    # summary(criterion, [(14,), (14,)], device='cuda')
    # torch.onnx.export(criterion, (torch.randn(1, 14).to(opt.device), torch.randn(1, 14).to(opt.device)), 'criterion.onnx')

    set_model = model_dic[opt.model]

    loss_func = nn.BCEWithLogitsLoss()
    model = set_model(opt).to(opt.device)
    optimizer = optim.SGD(model.parameters(), lr=opt.learning_rate)

    # Initialize the learning rate scheduler
    scheduler = ExponentialLR(optimizer, gamma=opt.lr_gamma)  # Adjust gamma to your needs

    model.train()
    for e in range(opt.epochs):
        total_loss = 0
        total_correct_num = 0
        strict_correct_num = 0
        total_num = 0

        all_preds = []
        all_labels = []

        widgets = [f'Epoch [{e + 1}/{opt.epochs}]: ', Percentage(), ' ', Bar('#'), ' ', Timer(),
                   ' ', ETA(), ' ', FileTransferSpeed()]
        pbar = ProgressBar(widgets=widgets, maxval=len(trainDataLoader)).start()

        for batch_id, (x, y) in enumerate(trainDataLoader):
            y_pre = model(x.to(opt.device))

            # -- compute accuracy and loss --
            ay_pre = (y_pre.float() > 0.1).cpu()
            judge = (y == ay_pre)

            total_correct_num += judge.sum().item()
            # loss = loss_func(y_pre, y.to(opt.device))
            loss = criterion(y_pre, y.to(opt.device))
            total_loss += loss.item() * x.size(0)

            # - strict acc -
            for i in range(judge.shape[0]):
                if judge[i].all():
                    strict_correct_num += 1

            total_num += y.shape[0]

            # -- save predictions and labels --
            all_preds.extend(ay_pre.numpy())
            all_labels.extend(y.numpy())

            # -- backward and optimize --
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # -- finish an epoch --
            pbar.update(batch_id + 1)

        pbar.finish()
        acc = total_correct_num / total_num / opt.num_classes
        strict_acc = strict_correct_num / total_num
        Loss = total_loss / total_num

        f1 = f1_score(all_labels, all_preds, average='weighted')

        # Logging
        mf_dic = {'epoch': e + 1, 'loss': Loss, 'acc': acc, 'strict_acc': strict_acc, 'f1_score': f1, 'learning_rate': optimizer.param_groups[0]['lr']}
        saver.save_model_info(**mf_dic)

        # Step the scheduler
        scheduler.step()

        model.eval()
        with torch.no_grad():

            total_loss = 0
            total_correct_num = 0
            strict_correct_num = 0
            total_num = 0

            all_preds = []
            all_labels = []

            widgets = [f'Evaluating: ', Percentage(), ' ', Bar('#'), ' ', Timer(),
                       ' ', ETA(), ' ', FileTransferSpeed()]

            pbar = ProgressBar(widgets=widgets, maxval=testDataLoader.__len__()).start()
            for batch_id, (x, y) in enumerate(testDataLoader):
                y_pre = model(x.to(opt.device))

                # -- compute accuracy and loss --
                ay_pre = (y_pre.float() > 0.1).cpu()
                judge = (y == ay_pre)

                total_correct_num += judge.sum().item()

                loss = loss_func(y_pre, y.to(opt.device))
                total_loss += loss.item() * opt.batch_size

                # - strict acc -
                for i in range(judge.shape[0]):
                    if judge[i].all():
                        strict_correct_num += 1

                total_num += y.shape[0]

                all_preds.extend(ay_pre.numpy())
                all_labels.extend(y.numpy())

                # -- finish an epoch --
                pbar.update(batch_id)

            pbar.finish()
            acc = total_correct_num / total_num / opt.num_classes
            strict_acc = strict_correct_num / total_num
            Loss = total_loss / total_num

            f1 = f1_score(all_labels, all_preds, average='weighted')

            tf_dic = {'epoch': e + 1, 'loss': Loss, 'acc': acc, 'strict_acc': strict_acc, 'f1_score': f1}

            saver.save_test_info(**tf_dic)

    torch.save(model, saver.result_dir + 'model.pt')
