import os
import torch
import torch.nn as nn
import time
from loss import imitation_loss
import numpy as np


def train(epoch, device, vis, train_loader, model, criterion, optimizer, scheduler, save_path, save_file_name):

    print('Training of epoch [{}]'.format(epoch))
    tic = time.time()
    model.train()

    # 10. train
    for idx, (images, boxes, labels, _) in enumerate(train_loader):
        images = images.cuda()
        boxes = [b.to(device) for b in boxes]
        labels = [l.to(device) for l in labels]

        preds = model(images)
        preds = preds.permute(0, 2, 3, 1)  # B, 13, 13, 125

        loss, losses = criterion(preds, boxes, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        toc = time.time() - tic

        for param_group in optimizer.param_groups:
            lr = param_group['lr']

        # for each steps
        if idx % 100 == 0:
            print('Epoch: [{0}]\t'
                  'Step: [{1}/{2}]\t'
                  'Loss: {loss:.4f}\t'
                  'Learning rate: {lr:.7f} s \t'
                  'Img size: {3} \t'
                  'Time : {time:.4f}\t'
                  .format(epoch, idx, len(train_loader), train_loader.dataset.img_size,
                          loss=loss,
                          lr=lr,
                          time=toc))

            if vis is not None:
                vis.line(X=torch.ones((1, 6)).cpu() * idx + epoch * train_loader.__len__(),  # step
                         Y=torch.Tensor([loss, losses[0], losses[1], losses[2], losses[3], losses[4]]).unsqueeze(
                             0).cpu(),
                         win='train_loss',
                         update='append',
                         opts=dict(xlabel='step',
                                   ylabel='Loss',
                                   title='training loss',
                                   legend=['Total Loss', 'xy_loss', 'wh_loss', 'conf_loss', 'no_conf_loss',
                                           'cls_loss']))

    if not os.path.exists(save_path):
        os.mkdir(save_path)
    checkpoint = {'epoch': epoch,
                  'model_state_dict': model.state_dict(),
                  'optimizer_state_dict': optimizer.state_dict()}

    if scheduler is not None:
        checkpoint = {'epoch': epoch,
                      'model_state_dict': model.state_dict(),
                      'optimizer_state_dict': optimizer.state_dict(),
                      'scheduler_state_dict': scheduler.state_dict()}
    torch.save(checkpoint, os.path.join(save_path, save_file_name) + '.{}.pth.tar'.format(epoch))


def train_with_teacher(epoch, device, vis, train_loader, teacher, student, criterion, optimizer, scheduler, save_path, save_file_name):

    print('Training of epoch [{}]'.format(epoch))
    tic = time.time()
    teacher.eval()
    student.train()
    imitation_criterion = nn.MSELoss()
    # 10. train
    for idx, (images, boxes, labels, _) in enumerate(train_loader):
        images = images.cuda()
        boxes = [b.to(device) for b in boxes]
        labels = [l.to(device) for l in labels]

        with torch.no_grad():
            _, teacher_feature, mask = teacher(images, boxes)

        preds, stu_feature, _ = student(images, boxes)

        preds = preds.permute(0, 2, 3, 1)  # B, 13, 13, 125

        loss, losses = criterion(preds, boxes, labels)

        i_loss = imitation_loss(student.stu_feature_adap(stu_feature), teacher_feature, mask)
        loss += i_loss * 0.01

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        toc = time.time() - tic

        for param_group in optimizer.param_groups:
            lr = param_group['lr']

        # for each steps
        if idx % 100 == 0:
            print('Epoch: [{0}]\t'
                  'Step: [{1}/{2}]\t'
                  'Loss: {loss:.4f}\t'
                  'Learning rate: {lr:.7f} s \t'
                  'Img size: {3} \t'
                  'Time : {time:.4f}\t'
                  .format(epoch, idx, len(train_loader), train_loader.dataset.img_size,
                          loss=loss,
                          lr=lr,
                          time=toc))

            if vis is not None:
                vis.line(X=torch.ones((1, 6)).cpu() * idx + epoch * train_loader.__len__(),  # step
                         Y=torch.Tensor([loss, losses[0], losses[1], losses[2], losses[3], losses[4], i_loss.item()]).unsqueeze(
                             0).cpu(),
                         win='train_loss',
                         update='append',
                         opts=dict(xlabel='step',
                                   ylabel='Loss',
                                   title='training loss',
                                   legend=['Total Loss', 'xy_loss', 'wh_loss', 'conf_loss', 'no_conf_loss',
                                           'cls_loss', 'imitation_loss']))

    if not os.path.exists(save_path):
        os.mkdir(save_path)
    checkpoint = {'epoch': epoch,
                  'model_state_dict': student.state_dict(),
                  'optimizer_state_dict': optimizer.state_dict()}

    if scheduler is not None:
        checkpoint = {'epoch': epoch,
                      'model_state_dict': student.state_dict(),
                      'optimizer_state_dict': optimizer.state_dict(),
                      'scheduler_state_dict': scheduler.state_dict()}
    torch.save(checkpoint, os.path.join(save_path, save_file_name) + '.{}.pth.tar'.format(epoch))





