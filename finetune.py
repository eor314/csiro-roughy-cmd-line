import time
import datetime
import os
#from matplotlib import pyplot as plt
import numpy as np
import mxnet as mx
from mxnet import autograd, gluon
import gluoncv as gcv
import argparse
import sys
from gluoncv.utils import viz
from gluoncv.data import VOCDetection
from gluoncv.data.batchify import Tuple, Stack, Pad
from gluoncv.data.transforms.presets.ssd import SSDDefaultTrainTransform, SSDDefaultValTransform
from gluoncv.utils.metrics.voc_detection import VOC07MApMetric


# define string interpreter for argparse
def str2bool(v):
    """
    returns a boolean from argparse input
    """

    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected')

# define a class with VOC structure so roughy data plays nicely with Gluon
class VOCLike(VOCDetection):
    # these are the original classes that are not consistent across the deployments [ECO 112320]
    """
    CLASSES = ['orange_roughy_edge', 'orange_roughy', 'sea_anemone', 'sea_urchin', 'oreo',
               'whiptail', 'eel', 'shark', 'worm', 'misc_fish', 'mollusc', 'shrimp',
               'sea_star']
    """
    # these are the 11 final classes after the merge operation [ECO 112320]
    CLASSES = ['brittle_star', 'cnidaria', 'eel', 'misc_fish', 'mollusc', 'orange_roughy_edge', 
               'orange_roughy', 'sea_anemone', 'sea_feather', 'sea_star','sea_urchin']
    
    #CLASSES = ['person','dog']
    def __init__(self, root, splits, transform=None, index_map=None, preload_label=True):
        super(VOCLike, self).__init__(root, splits, transform, index_map, preload_label)
        
# define the data loader
def get_dataloader(net, train_dataset, val_dataset, data_shape, batch_size, num_workers, ctx):
    """Get dataloader."""
    width, height = data_shape, data_shape
    
    # use fake data to generate fixed anchors for target generation
    with autograd.train_mode():
        _, _, anchors = net(mx.nd.zeros((1, 3, height, width), ctx))
    anchors = anchors.as_in_context(mx.cpu())
    batchify_fn = Tuple(Stack(), Stack(), Stack())  # stack image, cls_targets, box_targets
    
    # the training data loader (with transforms)
    train_loader = gluon.data.DataLoader(
        train_dataset.transform(SSDDefaultTrainTransform(width, height, anchors)),
        batch_size, True, batchify_fn=batchify_fn, last_batch='rollover',
        num_workers=num_workers)
    val_batchify_fn = Tuple(Stack(), Pad(pad_val=-1))
    
    # the validation loader 
    val_loader = gluon.data.DataLoader(
        val_dataset.transform(SSDDefaultValTransform(width, height)),
        batch_size, False, batchify_fn=val_batchify_fn, last_batch='keep',
        num_workers=num_workers)
    
    return train_loader, val_loader

# define validation
def validate(net, val_data, ctx, eval_metric):
    """
    Test on validation dataset
    :param net: network being used
    :param val_data: validation dataset
    :param ctx: training context (flag to set gpu)
    :param eval_metric: metric to quote performance
    :return eval_metric: updated evaluation metric 
    """
    eval_metric.reset()
    # set nms threshold and topk constraint (what bounding boxes are legit)
    net.set_nms(nms_thresh=0.45, nms_topk=400)
    net.hybridize(static_alloc=True, static_shape=True)
    
    flag = 0
    for batch in val_data:
        data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0, even_split=False)
        label = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0, even_split=False)
        det_bboxes = []
        det_ids = []
        det_scores = []
        gt_bboxes = []
        gt_ids = []
        gt_difficults = []
        for x, y in zip(data, label):
            # get prediction results
            ids, scores, bboxes = net(x)
            det_ids.append(ids)
            det_scores.append(scores)
            # clip to image size
            det_bboxes.append(bboxes.clip(0, batch[0].shape[2]))
            # split ground truths
            gt_ids.append(y.slice_axis(axis=-1, begin=4, end=5))
            gt_bboxes.append(y.slice_axis(axis=-1, begin=0, end=4))
            gt_difficults.append(y.slice_axis(axis=-1, begin=5, end=6) if y.shape[-1] > 5 else None)

        # update metric
        eval_metric.update(det_bboxes, det_ids, det_scores, gt_bboxes, gt_ids, gt_difficults)
    return eval_metric.get()

# training loop
def train(network, training_data, validation_data, eval_metric, ctx, epcs=10, val_int=2):
    """
    Fine-tune the network
    :param net: network architecture with weights being used
    :param training_data: training dataset loader
    :param validation_data: validation dataset loader
    :param ctx: training context (flag to set gpu)
    :param eval_metric: metric to quote performance on validation
    :param epcs: number of epochs to train (default=10)
    :param val_int: interval for performing validation (deafult=2)
    """
    
    #net.collect_params().reset_ctx(ctx)
    trainer = gluon.Trainer(
        net.collect_params(), 'sgd',
        {'learning_rate': 0.001, 'wd': 0.0005, 'momentum': 0.9})
    
    # learning rate decay (hard corded for now)
    #lr_decay = 0.1
    
    mbox_loss = gcv.loss.SSDMultiBoxLoss()  # this is the loss function for SSD
    ce_metric = mx.metric.Loss('CrossEntropy')  
    smoothl1_metric = mx.metric.Loss('SmoothL1')
    
    # preform the epochs
    for epoch in range(0, epcs):
        ce_metric.reset()
        smoothl1_metric.reset()
        tic = time.time()
        btic = time.time()
        net.hybridize(static_alloc=True, static_shape=True)
        
        # loop over the training data in batches
        for i, batch in enumerate(train_data):
            batch_size = batch[0].shape[0]
            
            # load the images and annotations
            data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0)
            cls_targets = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0)
            box_targets = gluon.utils.split_and_load(batch[2], ctx_list=ctx, batch_axis=0)
            with autograd.record():
                cls_preds = []
                box_preds = []
                for x in data:
                    cls_pred, box_pred, _ = net(x)
                    cls_preds.append(cls_pred)
                    box_preds.append(box_pred)
                sum_loss, cls_loss, box_loss = mbox_loss(
                    cls_preds, box_preds, cls_targets, box_targets)
                autograd.backward(sum_loss)
            # since we have already normalized the loss, we don't want to normalize
            # by batch-size anymore
            trainer.step(1)
            ce_metric.update(0, [l * batch_size for l in cls_loss])
            smoothl1_metric.update(0, [l * batch_size for l in box_loss])
            name1, loss1 = ce_metric.get()
            name2, loss2 = smoothl1_metric.get()
            if i % 20 == 0:
                print('[Epoch {}][Batch {}], Speed: {:.3f} samples/sec, {}={:.3f}, {}={:.3f}'.format(
                    epoch, i, batch_size/(time.time()-btic), name1, loss1, name2, loss2))
            btic = time.time()
            
        # Validation 
        if (epoch % val_int == 0):
            map_name, mean_ap = validate(net, val_data, ctx, eval_metric)
            val_msg = '\n'.join(['{}={}'.format(k, v) for k, v in zip(map_name, mean_ap)])
            #logger.info('[Epoch {}] Validation: \n{}'.format(epoch, val_msg))
            print(val_msg)
            current_map = float(mean_ap[-1])
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Finetune a lightweight SSD')
    parser.add_argument('voc_base', metavar='voc_base')
    parser.add_argument('train_yr', metavar='train_yr')
    parser.add_argument('val_yr', metavar='val_yr')
    parser.add_argument('--base_net', metavar='base_net', default='ssd_512_mobilenet1.0_voc')
    parser.add_argument('--epochs', metavar='epochs', default=10)
    parser.add_argument('--val_int', metavar='val_int', default=2)
    parser.add_argument('--save', metavar='save', default=True)
    parser.add_argument('--sv_path', metavar='sv_path', default='clf-outputs')
    
    args = parser.parse_args()
    voc_bb = args.voc_base
    train_yr = args.train_yr
    val_yr = args.val_yr
    epochs = int(args.epochs)
    val_int = int(args.val_int)
    sv = str2bool(args.save)
    sv_path = args.sv_path
    
    # load the network
    try:
        net = gcv.model_zoo.get_model(args.base_net, pretrained=True)
    except:
        print('net base does not exist')
        sys.exit()
    
    # find and activate GPU
    contx = [mx.gpu(0)]
    print('GPU found')
    
    # get the training and validation data
    train_dataset = VOCLike(root=voc_bb, splits=(('OP', f'{train_yr}'),))
    val_dataset = VOCLike(root=voc_bb, splits=(('OP', f'{val_yr}'),))
    print(train_dataset.classes)
    
    # define the validation metric (assume VOC07
    val_metric = VOC07MApMetric(iou_thresh=0.5, class_names=val_dataset.classes)
    
    # put the network on the GPU and reset the classes
    net.collect_params().reset_ctx(contx)
    net.reset_class(val_dataset.classes)
    
    # instantiate the dataloader (hardcoding data shape, batch size, and number of workers)
    train_data, val_data = get_dataloader(net, train_dataset, val_dataset, 512, 16, 0,
                                          contx[0])
    
    # actually run the training
    train(net, train_data, val_data, val_metric, contx, epochs, val_int)
    
    # save if desired
    train_name = os.path.split(train_yr)[0]
    if sv:
        now = datetime.datetime.now()
        sv_stub = f"{now.strftime('%m%d%y')}-{args.base_net}_roughy_{train_name}_{epochs}.params"
        sv_name = os.path.join(sv_path, sv_stub)
        
        net.save_parameters(sv_name)