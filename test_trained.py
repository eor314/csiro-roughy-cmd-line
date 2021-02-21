import time
import datetime
import os
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
from finetune import VOCLike, validate, str2bool


# define the data loader
def get_dataloader(net, val_dataset, data_shape, batch_size, num_workers, ctx):
    """Get dataloader."""
    width, height = data_shape, data_shape
    
    #val_batchify_fn = Tuple(Stack(), Stack(), Stack())  # stack image, cls_targets, box_targets
    val_batchify_fn = Tuple(Stack(), Pad(pad_val=-1))
    
    # the validation loader 
    val_loader = gluon.data.DataLoader(
        val_dataset.transform(SSDDefaultValTransform(width, height)),
        batch_size, False, batchify_fn=val_batchify_fn, last_batch='keep',
        num_workers=num_workers)
    
    return val_loader


# main loop
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='tests a finetuned lightweight SSD')
    parser.add_argument('net_weights', metavar='net_weights', help='path to network weights')
    parser.add_argument('test_set', metavar='test_set', help='path to test data parent dir')
    parser.add_argument('test_run', metavar='test_run', help='name of instrument run')

    args = parser.parse_args()
    net_wghts = args.net_weights
    test_set = args.test_set
    test_run = args.test_run
    
    # find and activate GPU
    contx = [mx.gpu(0)]
    print('GPU found')
    
    # load the base network and set the classes
    """
    classes = ['orange_roughy_edge', 'orange_roughy', 'sea_anemone', 'sea_urchin', 'oreo',
               'whiptail', 'eel', 'shark', 'worm', 'misc_fish', 'mollusc', 'shrimp',
               'sea_star']
    """
    classes = ['brittle_star', 'cnidaria', 'eel', 'misc_fish', 'mollusc',
               'orange_roughy_edge', 'orange_roughy', 'sea_anemone', 'sea_feather',
               'sea_star','sea_urchin']
        
    net = gcv.model_zoo.get_model('ssd_512_mobilenet1.0_custom', classes=classes,
                                  pretrained_base=False)
    
    # load the weights
    net.load_parameters(net_wghts)
    
    # put on the GPU
    net.collect_params().reset_ctx(contx)
    
    # get the data loader
    test_dataset = VOCLike(root=test_set, splits=(('OP', f'{test_run}'),))
    
    # define the validation metric (assume VOC07)
    val_metric = VOC07MApMetric(iou_thresh=0.5, class_names=test_dataset.classes)
    print(test_dataset.classes)
    
    # get the test data
    test_data = get_dataloader(net, test_dataset, 512, 16, 0, contx[0])
    
    # run the data
    map_name, mean_ap = validate(net, test_data, contx, val_metric)
    val_msg = '\n'.join(['{}={}'.format(k, v) for k, v in zip(map_name, mean_ap)])
    
    # print the output
    print(val_msg)
                                                              