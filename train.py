import pickle
import os
import time
import shutil
import yaml
import numpy as np

import torch
import pytorch_warmup as warmup

import data
from models.loss import AlignmentContrastiveLoss
from utils import get_model, cosine_sim, dot_sim
from evaluation import i2t, t2i, AverageMeter, LogCollector, encode_data
from evaluate_utils.dcg import DCG

import logging
from torch.utils.tensorboard import SummaryWriter

import argparse


def main():
    # Hyper Parameters
    parser = argparse.ArgumentParser()
    # parser.add_argument('--data_path', default='/w/31/faghri/vsepp_data/',
    #                     help='path to datasets')
    # parser.add_argument('--data_name', default='precomp',
    #                     help='{coco,f8k,f30k,10crop}_precomp|coco|f8k|f30k')
    parser.add_argument('--num_epochs', default=30, type=int,
                        help='Number of training epochs.')
    # parser.add_argument('--crop_size', default=224, type=int,
    #                     help='Size of an image crop as the CNN input.')
    parser.add_argument('--lr_update', default=15, type=int,
                        help='Number of epochs to update the learning rate.')
    parser.add_argument('--workers', default=10, type=int,
                        help='Number of data loader workers.')
    parser.add_argument('--log_step', default=10, type=int,
                        help='Number of steps to print and record the log.')
    parser.add_argument('--val_step', default=500, type=int,
                        help='Number of steps to run validation.')
    parser.add_argument('--test_step', default=100000000, type=int,
                        help='Number of steps to run validation.')
    parser.add_argument('--logger_name', default='runs/runX',
                        help='Path to save the model and Tensorboard log.')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none). Loads model, optimizer, scheduler')
    parser.add_argument('--load-model', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none). Loads only the model')
    parser.add_argument('--use_restval', action='store_true',
                        help='Use the restval data for training on MSCOCO.')
    parser.add_argument('--reinitialize-scheduler', action='store_true', help='Reinitialize scheduler. To use with --resume')
    parser.add_argument('--config', type=str, help="Which configuration to use. See into 'config' folder")

    opt = parser.parse_args()
    print(opt)

    # torch.cuda.set_enabled_lms(True)
    # if (torch.cuda.get_enabled_lms()):
    #     torch.cuda.set_limit_lms(11000 * 1024 * 1024)
    #     print('[LMS=On limit=' + str(torch.cuda.get_limit_lms()) + ']')

    with open(opt.config, 'r') as ymlfile:
        config = yaml.load(ymlfile)

    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
    tb_logger = SummaryWriter(log_dir=opt.logger_name, comment='')

    # Load data loaders
    train_loader, val_loader = data.get_loaders(
        config, opt.workers)
    # test_loader = data.get_test_loader(config, vocab=vocab, workers=4, split_name='test')

    # Construct the model
    model = get_model(config)
    if torch.cuda.is_available() and not (opt.resume or opt.load_model):
        model.cuda()

    assert not ((config['image-model']['fine-tune'] or config['text-model']['fine-tune']) and config['dataset']['pre-extracted-features'])
    # Construct the optimizer

    # if config['model']['name'] == 'transformthem':      # TODO: handle better
    #     params = model.parameters()
    #     for p in model.img_txt_enc.txt_enc.parameters():
    #         p.requires_grad = False
    # else:
    #     params = list(model.txt_enc.parameters())
    #     params += list(model.img_enc.fc.parameters())
    #
    #     if not config['dataset']['pre-extracted-features']:
    #         if config['image-model']['fine-tune']:
    #             print('Finetuning image encoder')
    #             params += list(model.img_enc.get_finetuning_params())
    #         if config['text-model']['fine-tune']:
    #             print('Finetuning text encoder')
    #             params += list(model.txt_enc.get_finetuning_params())

    params, secondary_lr_multip = model.get_parameters()
    # validity check
    all_params = params[0] + params[1]
    if len(all_params) != len(list(model.parameters())):
        raise ValueError('Not all parameters are being returned! Correct get_parameters() method')

    if secondary_lr_multip > 0:
        optimizer = torch.optim.Adam([{'params': params[0]},
                                      {'params': params[1], 'lr': config['training']['lr']*secondary_lr_multip}],
                                     lr=config['training']['lr'])
    else:
        optimizer = torch.optim.Adam(params[0], lr=config['training']['lr'])

    # LR scheduler
    scheduler_name = config['training']['scheduler']
    if scheduler_name == 'steplr':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, config['training']['step-size'], gamma=config['training']['gamma'])
    elif scheduler_name is None:
        scheduler = None
    else:
        raise ValueError('{} scheduler is not available'.format(scheduler_name))

    # Warmup scheduler
    warmup_scheduler_name = config['training']['warmup'] if not opt.resume else None
    if warmup_scheduler_name == 'linear':
        warmup_scheduler = warmup.LinearWarmup(optimizer, warmup_period=config['training']['warmup-period'])
    elif warmup_scheduler_name is None:
        warmup_scheduler = None
    else:
        raise ValueError('{} warmup scheduler is not available'.format(warmup_scheduler_name))

    # optionally resume from a checkpoint
    start_epoch = 0
    if opt.resume or opt.load_model:
        filename = opt.resume if opt.resume else opt.load_model
        if os.path.isfile(filename):
            print("=> loading checkpoint '{}'".format(filename))
            checkpoint = torch.load(filename, map_location='cpu')
            model.load_state_dict(checkpoint['model'], strict=False)
            if torch.cuda.is_available():
                model.cuda()
            if opt.resume:
                start_epoch = checkpoint['epoch']
                # best_rsum = checkpoint['best_rsum']
                optimizer.load_state_dict(checkpoint['optimizer'])
                if checkpoint['scheduler'] is not None and not opt.reinitialize_scheduler:
                    scheduler.load_state_dict(checkpoint['scheduler'])
                # Eiters is used to show logs as the continuation of another
                # training
                model.Eiters = checkpoint['Eiters']
                print("=> loaded checkpoint '{}' (epoch {})"
                      .format(opt.resume, start_epoch))
            else:
                print("=> loaded only model from checkpoint '{}'"
                      .format(opt.load_model))
        else:
            print("=> no checkpoint found at '{}'".format(opt.resume))

    if torch.cuda.is_available():
        model.cuda()
    model.train()

    # load the ndcg scorer
    ndcg_val_scorer = DCG(config, len(val_loader.dataset), 'val', rank=25, relevance_methods=['rougeL', 'spice'])
    # ndcg_test_scorer = DCG(config, len(test_loader.dataset), 'test', rank=25, relevance_methods=['rougeL', 'spice'])

    # Train the Model
    best_rsum = 0
    best_ndcg_sum = 0
    alignment_mode = config['training']['alignment-mode'] if config['training']['loss-type'] == 'alignment' else None

    # validate(val_loader, model, tb_logger, measure=config['training']['measure'], log_step=opt.log_step,
    #          ndcg_scorer=ndcg_val_scorer, alignment_mode=alignment_mode)

    for epoch in range(start_epoch, opt.num_epochs):
        # train for one epoch
        train(opt, train_loader, model, optimizer, epoch, tb_logger, val_loader, None,
              measure=config['training']['measure'], grad_clip=config['training']['grad-clip'],
              scheduler=scheduler, warmup_scheduler=warmup_scheduler, ndcg_val_scorer=ndcg_val_scorer, ndcg_test_scorer=None, alignment_mode=alignment_mode)

        # evaluate on validation set
        rsum, ndcg_sum = validate(val_loader, model, tb_logger, measure=config['training']['measure'], log_step=opt.log_step,
                        ndcg_scorer=ndcg_val_scorer, alignment_mode=alignment_mode)

        # remember best R@ sum and save checkpoint
        is_best_rsum = rsum > best_rsum
        best_rsum = max(rsum, best_rsum)

        is_best_ndcg = ndcg_sum > best_ndcg_sum
        best_ndcg_sum = max(ndcg_sum, best_ndcg_sum)
        #
        # is_best_r1 = r1 > best_r1
        # best_r1 = max(r1, best_r1)

        # is_best_val_loss = val_loss < best_val_loss
        # best_val_loss = min(val_loss, best_val_loss)

        save_checkpoint({
            'epoch': epoch + 1,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict() if scheduler is not None else None,
            'opt': opt,
            'config': config,
            'Eiters': model.Eiters,
        }, is_best_rsum, is_best_ndcg, prefix=opt.logger_name + '/')


def train(opt, train_loader, model, optimizer, epoch, tb_logger, val_loader, test_loader, measure='cosine', grad_clip=-1, scheduler=None, warmup_scheduler=None, ndcg_val_scorer=None, ndcg_test_scorer=None, alignment_mode=None):
    # average meters to record the training statistics
    batch_time = AverageMeter()
    data_time = AverageMeter()
    train_logger = LogCollector()

    end = time.time()
    for i, train_data in enumerate(train_loader):
        model.train()
        if scheduler is not None:
            scheduler.step(epoch)

        if warmup_scheduler is not None:
            warmup_scheduler.dampen()

        optimizer.zero_grad()

        # measure data loading time
        data_time.update(time.time() - end)

        # make sure train logger is used
        model.logger = train_logger

        # Update the model
        loss_dict = model(*train_data)
        loss = sum(loss for loss in loss_dict.values())

        # compute gradient and do SGD step
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # Print log info
        if model.Eiters % opt.log_step == 0:
            logging.info(
                'Epoch: [{0}][{1}/{2}]\t'
                '{e_log}\t'
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                .format(
                    epoch, i, len(train_loader), batch_time=batch_time,
                    data_time=data_time, e_log=str(model.logger)))

        # Record logs in tensorboard
        tb_logger.add_scalar('epoch', epoch, model.Eiters)
        tb_logger.add_scalar('step', i, model.Eiters)
        tb_logger.add_scalar('batch_time', batch_time.val, model.Eiters)
        tb_logger.add_scalar('data_time', data_time.val, model.Eiters)
        tb_logger.add_scalar('lr', optimizer.param_groups[0]['lr'], model.Eiters)
        model.logger.tb_log(tb_logger, step=model.Eiters)

        # validate at every val_step
        if model.Eiters % opt.val_step == 0:
            validate(val_loader, model, tb_logger, measure=measure, log_step=opt.log_step, ndcg_scorer=ndcg_val_scorer, alignment_mode=alignment_mode)

        # if model.Eiters % opt.test_step == 0:
        #     test(test_loader, model, tb_logger, measure=measure, log_step=opt.log_step, ndcg_scorer=ndcg_test_scorer)


def validate(val_loader, model, tb_logger, measure='cosine', log_step=10, ndcg_scorer=None, alignment_mode=None):
    # compute the encoding for all the validation images and captions
    img_embs, cap_embs, img_lenghts, cap_lenghts = encode_data(
        model, val_loader, log_step, logging.info)

    # initialize similarity matrix evaluator
    sim_matrix_fn = AlignmentContrastiveLoss(aggregation=alignment_mode, return_similarity_mat=True) if alignment_mode is not None else None

    if measure == 'cosine':
        sim_fn = cosine_sim
    elif measure == 'dot':
        sim_fn = dot_sim

    # caption retrieval
    (r1, r5, r10, medr, meanr, mean_rougel_ndcg, mean_spice_ndcg) = i2t(img_embs, cap_embs, img_lenghts, cap_lenghts, measure=measure, ndcg_scorer=ndcg_scorer, sim_function=sim_matrix_fn)
    logging.info("Image to text: %.1f, %.1f, %.1f, %.1f, %.1f, ndcg_rouge=%.4f ndcg_spice=%.4f" %
                 (r1, r5, r10, medr, meanr, mean_rougel_ndcg, mean_spice_ndcg))
    # image retrieval
    (r1i, r5i, r10i, medri, meanr, mean_rougel_ndcg_i, mean_spice_ndcg_i) = t2i(
        img_embs, cap_embs, img_lenghts, cap_lenghts, ndcg_scorer=ndcg_scorer, measure=measure, sim_function=sim_matrix_fn)

    logging.info("Text to image: %.1f, %.1f, %.1f, %.1f, %.1f, ndcg_rouge=%.4f ndcg_spice=%.4f" %
                 (r1i, r5i, r10i, medri, meanr, mean_rougel_ndcg_i, mean_spice_ndcg_i))
    # sum of recalls to be used for early stopping
    currscore = r1 + r5 + r10 + r1i + r5i + r10i
    spice_ndcg_sum = mean_spice_ndcg + mean_spice_ndcg_i

    # record metrics in tensorboard
    tb_logger.add_scalar('r1', r1, model.Eiters)
    tb_logger.add_scalar('r5', r5, model.Eiters)
    tb_logger.add_scalar('r10', r10, model.Eiters)
    tb_logger.add_scalars('mean_ndcg', {'rougeL': mean_rougel_ndcg, 'spice': mean_spice_ndcg}, model.Eiters)
    tb_logger.add_scalar('medr', medr, model.Eiters)
    tb_logger.add_scalar('meanr', meanr, model.Eiters)
    tb_logger.add_scalar('r1i', r1i, model.Eiters)
    tb_logger.add_scalar('r5i', r5i, model.Eiters)
    tb_logger.add_scalar('r10i', r10i, model.Eiters)
    tb_logger.add_scalars('mean_ndcg_i', {'rougeL': mean_rougel_ndcg_i, 'spice': mean_spice_ndcg_i}, model.Eiters)
    tb_logger.add_scalar('medri', medri, model.Eiters)
    tb_logger.add_scalar('meanr', meanr, model.Eiters)
    tb_logger.add_scalar('rsum', currscore, model.Eiters)
    tb_logger.add_scalar('spice_ndcg_sum', spice_ndcg_sum, model.Eiters)

    return currscore, spice_ndcg_sum


def test(test_loader, model, tb_logger, measure='cosine', log_step=10, ndcg_scorer=None):
    # compute the encoding for all the validation images and captions
    img_embs, cap_embs = encode_data(
        model, test_loader, log_step, logging.info)

    if measure == 'cosine':
        sim_fn = cosine_sim
    elif measure == 'dot':
        sim_fn = dot_sim

    results = []
    for i in range(5):
        r, rt0 = i2t(img_embs[i * 5000:(i + 1) * 5000], cap_embs[i * 5000:(i + 1) * 5000],
                     None, None,
                     return_ranks=True, ndcg_scorer=ndcg_scorer, fold_index=i)
        print("Image to text: %.1f, %.1f, %.1f, %.1f, %.1f, ndcg_rouge=%.4f ndcg_spice=%.4f" % r)
        ri, rti0 = t2i(img_embs[i * 5000:(i + 1) * 5000], cap_embs[i * 5000:(i + 1) * 5000],
                       None, None,
                       return_ranks=True, ndcg_scorer=ndcg_scorer, fold_index=i)
        if i == 0:
            rt, rti = rt0, rti0
        print("Text to image: %.1f, %.1f, %.1f, %.1f, %.1f, ndcg_rouge=%.4f, ndcg_spice=%.4f" % ri)
        ar = (r[0] + r[1] + r[2]) / 3
        ari = (ri[0] + ri[1] + ri[2]) / 3
        rsum = r[0] + r[1] + r[2] + ri[0] + ri[1] + ri[2]
        print("rsum: %.1f ar: %.1f ari: %.1f" % (rsum, ar, ari))
        results += [list(r) + list(ri) + [ar, ari, rsum]]

    print("-----------------------------------")
    print("Mean metrics: ")
    mean_metrics = tuple(np.array(results).mean(axis=0).flatten())
    print("rsum: %.1f" % (mean_metrics[16] * 6))
    print("Average i2t Recall: %.1f" % mean_metrics[14])
    print("Image to text: %.1f %.1f %.1f %.1f %.1f ndcg_rouge=%.4f ndcg_spice=%.4f" %
          mean_metrics[:7])
    print("Average t2i Recall: %.1f" % mean_metrics[15])
    print("Text to image: %.1f %.1f %.1f %.1f %.1f ndcg_rouge=%.4f ndcg_spice=%.4f" %
          mean_metrics[7:14])

    # record metrics in tensorboard
    tb_logger.add_scalar('test/r1', mean_metrics[0], model.Eiters)
    tb_logger.add_scalar('test/r5', mean_metrics[1], model.Eiters)
    tb_logger.add_scalar('test/r10', mean_metrics[2], model.Eiters)
    tb_logger.add_scalars('test/mean_ndcg', {'rougeL': mean_metrics[5], 'spice': mean_metrics[6]}, model.Eiters)
    tb_logger.add_scalar('test/r1i', mean_metrics[7], model.Eiters)
    tb_logger.add_scalar('test/r5i', mean_metrics[8], model.Eiters)
    tb_logger.add_scalar('test/r10i', mean_metrics[9], model.Eiters)
    tb_logger.add_scalars('test/mean_ndcg_i', {'rougeL': mean_metrics[12], 'spice': mean_metrics[13]}, model.Eiters)


def save_checkpoint(state, is_best_rsum, is_best_ndcg, filename='checkpoint.pth.tar', prefix=''):
    torch.save(state, prefix + filename)
    if is_best_rsum:
        shutil.copyfile(prefix + filename, prefix + 'model_best_rsum.pth.tar')
    if is_best_ndcg:
        shutil.copyfile(prefix + filename, prefix + 'model_best_ndcgspice.pth.tar')


# def adjust_learning_rate(opt, optimizer, epoch):
#     """Sets the learning rate to the initial LR
#        decayed by 10 every 30 epochs"""
#     lr = opt.learning_rate * (0.1 ** (epoch // opt.lr_update))
#     for param_group in optimizer.param_groups:
#         param_group['lr'] = lr


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


if __name__ == '__main__':
    main()
