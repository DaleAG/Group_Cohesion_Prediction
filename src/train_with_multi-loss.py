import argparse
import os,sys,shutil
import time
# from sampler import ImbalancedDatasetSampler
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.nn.functional as F
#import transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
import math
from part_attention import resnet18,resnet34
from part_attentioon_sample import MsCelebDataset, CaffeCrop
import scipy.io as sio  
import numpy as np
import pdb
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))


parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('--img_dir', metavar='DIR', default='', help='path to dataset')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18', choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: alexnet)')
parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=20, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=64, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('-b_t', '--batch-size_t', default=64, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float, 
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=20, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--pretrained', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--model_dir','-m', default='./models', type=str)
parser.add_argument('--end2end', default=True,\
        help='if true, using end2end with dream block, else, using naive architecture')

best_prec1 = 1

def main():
    global args, best_prec1
    args = parser.parse_args()
    print('end2end?:', args.end2end)
    train_list_file = '/new_face_train_1.txt'
    caffe_crop = CaffeCrop('train')
    train_dataset =  MsCelebDataset(train_list_file, 
            transforms.Compose([caffe_crop,transforms.ToTensor()]))

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        # sampler=ImbalancedDatasetSampler(train_dataset),
        batch_size=args.batch_size,
        num_workers=args.workers, pin_memory=True)
   
    caffe_crop = CaffeCrop('test')
    val_list_file = '/new_face_val_1.txt'
    val_dataset =  MsCelebDataset(val_list_file,
            transforms.Compose([caffe_crop,transforms.ToTensor()]))
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        shuffle=True,
        batch_size=args.batch_size_t,
        num_workers=args.workers, pin_memory=True)
    
    # prepare model
    model = None
    assert(args.arch in ['resnet18','resnet34','resnet101'])
    if args.arch == 'resnet18':
        model = resnet18(end2end=args.end2end)
    if args.arch == 'resnet34':
        model = resnet34(end2end=args.end2end)
    if args.arch == 'resnet101':
        pass

    model = torch.nn.DataParallel(model).cuda()
    #model.module.theta.requires_grad = True
    
    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()
    criterion1 = nn.MSELoss().cuda()
    criterion2 = kw_rank_loss().cuda()
    # criterion3 = dg_hourglass_loss().cuda()
    criterion3 = dg_hourglass_direction_loss().cuda()
    #criterion=Cross_Entropy_Sample_Weight.CrossEntropyLoss_weight().cuda()
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), args.lr,
                                 momentum=args.momentum,
                                 weight_decay=args.weight_decay)
#    optimizer = torch.optim.SGD(model.parameters(), args.lr,
#                                momentum=args.momentum,
#                                weight_decay=args.weight_decay)
   # optionally resume from a checkpoint
    
    if args.pretrained:
        
        checkpoint = torch.load('resnet18_pretrained.pth.tar')
        pretrained_state_dict = checkpoint['state_dict']
        model_state_dict = model.state_dict()
        
        for key in pretrained_state_dict:
            if  ((key=='module.fc.weight')|(key=='module.fc.bias')):

            #if  ((key=='module.fc.weight')|(key=='module.fc.bias')|(key == 'module.feature.weight')|(key == 'module.feature.bias')):
                pass
            else:    
                model_state_dict[key] = pretrained_state_dict[key]

        model.load_state_dict(model_state_dict, strict = False)
        #pdb.set_trace()

    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    result_list = []
    print ('args.evaluate',args.evaluate)
    if args.evaluate:
        validate(val_loader, model, criterion1, criterion, criterion2, criterion3)
        return

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        train(train_loader, model, criterion1,criterion, criterion2, criterion3, optimizer, epoch)

        # evaluate on validation set
        prec1 = validate(val_loader, model, criterion1,criterion, criterion2, criterion3)

        # remember best prec@1 and save checkpoint
        is_best = prec1 < best_prec1
        
        best_prec1 = min(prec1, best_prec1)
        
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer' : optimizer.state_dict(),
        }, is_best)

        result_list.append(prec1)

    print("result_list:", result_list)
    print("best_result:", best_prec1)

def train(train_loader, model, criterion1,criterion, criterion2, criterion3, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    cla_losses = AverageMeter()
    yaw_losses = AverageMeter()
    losses = AverageMeter()
    classification_losses = AverageMeter()
    regression_losses = AverageMeter()
    rank_losses = AverageMeter()
    hourglass_losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input_first, target_first,gr_em_label, input_second,target_second,gr_em_label, input_third, target_third,gr_em_label) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        
        input = torch.zeros([input_first.shape[0],input_first.shape[1],input_first.shape[2],input_first.shape[3],3])
        #input = torch.cat((input_first,input_second),0)
        #input = torch.cat((input,input_third),0)


        input[:,:,:,:,0] = input_first
        input[:,:,:,:,1] = input_second
        input[:,:,:,:,2] = input_third

        target = target_first.float()
        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target).unsqueeze(1)
        gr_em_label = gr_em_label.cuda(async=True)
        gr_em_label_var = torch.autograd.Variable(gr_em_label)
        
        
        # compute output
        # pdb.set_trace()
        pred_score, classification_pred,feature_results = model(input_var)
        #pdb.set_trace()
        regression_loss = criterion1(pred_score, target_var)
        classification_loss = criterion(classification_pred, gr_em_label_var)
        #pdb.set_trace()
        rank_loss = criterion2(feature_results, target_var)
        hourglass_loss = criterion3(pred_score, target_var)
        # loss = regression_loss + 1.5*classification_loss + 1.5*rank_loss + 1*hourglass_loss
        loss = regression_loss + 1.5*classification_loss + 1.5*rank_loss + 0.3*hourglass_loss
        # print("regression_loss:",regression_loss)
        # print("classification_loss:",classification_loss)
        # print("rank_loss:",rank_loss)
        # print("hourglass_loss:",hourglass_loss)

        # loss = regression_loss
        # loss = classification_loss*(1/math.exp(-regression_loss))
        
        # measure accuracy and record loss
        prec1 = accuracy(classification_pred.data, gr_em_label_var, topk=(1,))
        # pdb.set_trace()
        regression_losses.update(regression_loss.item(), input.size(0))
        classification_losses.update(classification_loss.item(), input.size(0))
        rank_losses.update(rank_loss, input.size(0))
        hourglass_losses.update(hourglass_loss, input.size(0))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1[0], input.size(0))
        

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val} ({batch_time.avg})\t'
                  'Data {data_time.val} ({data_time.avg})\t'
                  'regLoss {regression_loss.val} ({regression_loss.avg})\t'
                  'claLoss {classification_loss.val} ({classification_loss.avg})\t'
                  'rankLoss {rank_loss.val} ({rank_loss.avg})\t'
                  'hourglassLoss {hourglass_loss.val} ({hourglass_loss.avg})\t'
                  'Prec@1 {top1.val} ({top1.avg})\t'
                                                              .format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, regression_loss=regression_losses, classification_loss = classification_losses, rank_loss = rank_losses, hourglass_loss = hourglass_losses, top1 = top1))


def validate(val_loader, model, criterion1,criterion, criterion2, criterion3):
    batch_time = AverageMeter()
    cla_losses = AverageMeter()
    yaw_losses = AverageMeter()
    losses = AverageMeter()
    classification_losses = AverageMeter()
    regression_losses = AverageMeter()
    rank_losses = AverageMeter()
    hourglass_losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input_first, target_first,gr_em_label, input_second,target_second,gr_em_label, input_third, target_third,gr_em_label) in enumerate(val_loader):
        # target = target.cuda(async=True)
        # input_var = torch.autograd.Variable(input, volatile=True)
        # target_var = torch.autograd.Variable(target, volatile=True)
        # compute output
        input = torch.zeros([input_first.shape[0],input_first.shape[1],input_first.shape[2],input_first.shape[3],3])
        #input = torch.cat((input_first,input_second),0)
        #input = torch.cat((input,input_third),0)


        input[:,:,:,:,0] = input_first
        input[:,:,:,:,1] = input_second
        input[:,:,:,:,2] = input_third
        #pdb.set_trace()
        
        target = target_first.float()
         

        target = target.cuda(async=True)
 
        
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target).unsqueeze(1)
        gr_em_label = gr_em_label.cuda(async=True)
        gr_em_label_var = torch.autograd.Variable(gr_em_label)

        pred_score, classification_pred, feature_results = model(input_var)
        #pdb.set_trace()
        regression_loss = criterion1(pred_score, target_var)
        classification_loss = criterion(classification_pred, gr_em_label_var)
        rank_loss = criterion2(feature_results, target_var)
        hourglass_loss = criterion3(pred_score, target_var)
        # loss = regression_loss + 1.5*classification_loss + 1.5*rank_loss + 1 * hourglass_loss
        loss = regression_loss + 1.5*classification_loss + 1.5*rank_loss + 0.3 * hourglass_loss
        # loss = regression_loss
        # loss = classification_loss*(1/math.exp(-regression_loss))

        # measure accuracy and record loss
        prec1 = accuracy(classification_pred.data, gr_em_label_var, topk=(1,))
        regression_losses.update(regression_loss.item(), input.size(0))
        classification_losses.update(classification_loss.item(), input.size(0))
        rank_losses.update(rank_loss, input.size(0))
        hourglass_losses.update(hourglass_loss, input.size(0))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1[0], input.size(0))
        

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val} ({batch_time.avg})\t'
                  'regLoss {regression_loss.val} ({regression_loss.avg})\t'
                  'claLoss {classification_loss.val} ({classification_loss.avg})\t'
                  'rankLoss {rank_loss.val} ({rank_loss.avg})\t'
                  'hourglassLoss {hourglass_loss.val} ({hourglass_loss.avg})\t'
                  'Prec@1 {top1.val} ({top1.avg})\t'
                  .format(
                   i, len(val_loader), batch_time=batch_time, regression_loss=regression_losses, classification_loss=classification_losses,
                   rank_loss = rank_losses, hourglass_loss = hourglass_losses, top1=top1
                   ))

    print(' * Prec@1 {regression_loss.avg} '
          .format(regression_loss=regression_losses))

    return regression_losses.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):

    full_filename = os.path.join(args.model_dir, filename)
    full_bestname = os.path.join(args.model_dir, 'model_best.pth.tar')
    torch.save(state, full_filename)
    epoch_num = state['epoch']
    if epoch_num%1==0 and epoch_num>=0:
        torch.save(state, full_filename.replace('checkpoint','checkpoint_'+str(epoch_num)))
    if is_best:
        shutil.copyfile(full_filename, full_bestname)


class AverageMeter(object): 
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    #lr = args.lr * (0.1 ** (epoch // 30))
    if epoch in [int(args.epochs*0.6), int(args.epochs*0.8), int(args.epochs*0.8)]:
    # if epoch in [int(args.epochs*0.3), int(args.epochs*0.5), int(args.epochs*0.8)]:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= 0.1


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res




class kw_rank_loss(nn.Module):
    def __init__(self):        
        super(kw_rank_loss, self).__init__()          
    def forward(self, feature_results, target_var):
        engagement_level_1 = np.zeros((1, feature_results.shape[1]))
        engagement_level_2 = np.zeros((1, feature_results.shape[1]))
        engagement_level_3 = np.zeros((1, feature_results.shape[1]))
        engagement_level_4 = np.zeros((1, feature_results.shape[1]))
        level_1_num = level_2_num = level_3_num = level_4_num = 1

        size = feature_results.shape[0]
        feature_results = feature_results.cpu().data.numpy()
        margin = 0.75

        for i in range(size):
            if target_var[i] <= 0.1:
                engagement_level_1 += feature_results[i]
                level_1_num += 1
            if 0.3 <target_var[i] <= 0.4:
                engagement_level_2 += feature_results[i]
                level_2_num += 1
            if 0.6 <target_var[i] <= 0.7:
                engagement_level_3 += feature_results[i]
                level_3_num += 1
            if 0.8 <target_var[i] <= 1:
                engagement_level_4 += feature_results[i]
                level_4_num += 1

        engagement_level_1 = engagement_level_1/level_1_num
        engagement_level_2 = engagement_level_2/level_2_num
        engagement_level_3 = engagement_level_3/level_3_num
        engagement_level_4 = engagement_level_4/level_4_num

        dist_1_2 = np.linalg.norm(engagement_level_1 - engagement_level_2)
        dist_1_3 = np.linalg.norm(engagement_level_1 - engagement_level_3)
        dist_1_4 = np.linalg.norm(engagement_level_1 - engagement_level_4)
        dist_2_3 = np.linalg.norm(engagement_level_2 - engagement_level_3)
        dist_2_4 = np.linalg.norm(engagement_level_2 - engagement_level_4)
        dist_3_4 = np.linalg.norm(engagement_level_3 - engagement_level_4)

        loss = max(0.0,(dist_1_2 - dist_1_3 + margin)) + max(0.0,(dist_1_2 - dist_1_4 +2*margin)) + \
               max(0.0,(dist_2_3 - dist_2_4 + margin)) + max(0.0,(dist_2_3 - dist_1_4 +2*margin)) + \
               max(0.0,(dist_3_4 - dist_2_4 + margin)) + max(0.0,(dist_3_4 - dist_1_4 +2*margin))
        ####beigin calculate the distance between center and samples
        center_dist_1 = 0
        center_dist_2 = 0
        center_dist_3 = 0
        center_dist_4 = 0
        for i in range(size):
            if target_var[i] <= 0.1:
                center_dist_1 += np.linalg.norm(engagement_level_1 - feature_results[i])
            if 0.3 <target_var[i] <= 0.4:
                center_dist_2 += np.linalg.norm(engagement_level_2 - feature_results[i])
            if 0.6 <target_var[i] <= 0.7:
                center_dist_3 += np.linalg.norm(engagement_level_3 - feature_results[i])
            if 0.8 <target_var[i] <= 1:
                center_dist_4 += np.linalg.norm(engagement_level_4 - feature_results[i])
        center_rank_loss = loss + 0.001*(center_dist_1 + center_dist_2 + center_dist_3 + center_dist_4)
        return center_rank_loss

class dg_hourglass_loss(nn.Module):
    def __init__(self):
        super(dg_hourglass_loss, self).__init__()

    def hourglass(self, left, pred_score, right):
        return (pred_score - left)*(right - pred_score)

    def forward(self, pred_score, target_var):
        size = pred_score.shape[0]
        pred_score = pred_score.cpu().data.numpy()
        hourglass_loss = 0

        dg_loss = 0
        for i in range(size):
            if target_var[i] == 0.0:
                if pred_score[i] >= 0.6667:
                    hourglass_loss += self.hourglass(0.0, pred_score[i], 1.0)
                elif 0.33 <= pred_score[i] < 0.6667:
                    hourglass_loss += self.hourglass(0.0, pred_score[i], 0.666)
                elif 0.0 <= pred_score[i] < 0.33:
                    hourglass_loss += self.hourglass(0.0, pred_score[i], 0.333)

            if 0.3 <target_var[i] <= 0.4: #lebel = 0.3333333
                if pred_score[i] >= 0.6667:
                    hourglass_loss += self.hourglass(0.333, pred_score[i], 1.0)
                elif 0.33 <= pred_score[i] < 0.66:
                    hourglass_loss += self.hourglass(0.333, pred_score[i], 0.6667)
                elif 0.0 <= pred_score[i] < 0.33:
                    hourglass_loss += self.hourglass(0.0, pred_score[i], 0.333)

            if 0.6 <target_var[i] <= 0.7: #lebel = 0.6666666
                if pred_score[i] >= 0.66:
                    hourglass_loss += self.hourglass(0.6667, pred_score[i], 1.0)
                elif 0.33 <= pred_score[i] < 0.66:
                    hourglass_loss += self.hourglass(0.3333, pred_score[i], 0.6667)
                elif 0.0 <= pred_score[i] < 0.33:
                    hourglass_loss += self.hourglass(0.0, pred_score[i], 0.666)

            if target_var[i] == 1.0:
                if pred_score[i] >= 0.66:
                    hourglass_loss += self.hourglass(0.666, pred_score[i], 1.0)
                elif 0.33 <= pred_score[i] < 0.66:
                    hourglass_loss += self.hourglass(0.333, pred_score[i], 1.0)
                elif 0.0 <= pred_score[i] < 0.33:
                    hourglass_loss += self.hourglass(0.0, pred_score[i], 1.0)
            dg_loss += float(hourglass_loss)
        return 0.1 * dg_loss

if __name__ == '__main__':
    main()
