import os
import torch
import inspect
import shutil
import random
import pickle

import tableprint as tp
import numpy as np

from time import time
from datetime import timedelta
from shutil import copy
from tqdm import tqdm
from torch.utils.data import DataLoader

from dataio.loader import get_dataset, get_dataset_path
from dataio.transformation import get_dataset_transformation

from utils.util import json_file_to_pyobj
from utils.visualiser import Visualiser
from utils.error_logger import ErrorLogger

from logger.plot import history_ploter

from models import get_model
from models.networks import get_model_instance


def manual_s(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

def save_history_model_opt(best_dice_avg, loss_history, dice_avg_history, best_hd95_avg, model, epoch):
    save_dir = model.save_dir
    # history
    history_dict = {'best_dice_avg': best_dice_avg,
                    'loss_history': loss_history,
                    'dice_avg_history': dice_avg_history,
                    'best_hd95_avg': best_hd95_avg}
    with open(os.path.join(save_dir, str(epoch)+'_history.pkl'), 'wb') as f:
        pickle.dump(history_dict, f)
    # model and opt
    model.save(epoch)

def train(arguments):

    # Parse input arguments
    json_filename = arguments.config
    network_debug = arguments.debug

    # Load options
    json_opts = json_file_to_pyobj(json_filename)
    train_opts = json_opts.training
    path_pre_trained_model = json_opts.model.path_pre_trained_model
    resume = json_opts.model.continue_train

    # Architecture type
    arch_type = train_opts.arch_type

    # Setup Dataset and Augmentation
    ds_class = get_dataset(arch_type)                               
    ds_path  = get_dataset_path(arch_type, json_opts.data_path)
    addson = {}
    if json_opts.model.void_classes != []:
        addson = {'ignore_index': json_opts.model.ignore_index, 'void_classes': json_opts.model.void_classes}
    if 'edge' in json_opts.model.criterion:
        addson['edge'] = True
        addson['edge_input'] = json_opts.model.edge_input
        addson['edge_type'] = json_opts.model.edge_type

    ds_transform = get_dataset_transformation(arch_type, opts=json_opts.augmentation)

    # Setup the NN Model
    model = get_model(json_opts.model)
    if network_debug:
        print('# of pars: ', model.get_number_parameters())
        print('fp time: {0:.3f} sec\tbp time: {1:.3f} sec per sample'.format(*model.get_fp_bp_time()))
        exit()

    # Setup Data Loader
    train_dataset = ds_class(ds_path, split='train',      fold_no=json_opts.data_path.fold, transform=ds_transform['train'], preload_data=train_opts.preloadData, **addson)
    valid_dataset = ds_class(ds_path, split='validation', fold_no=json_opts.data_path.fold, transform=ds_transform['valid'], preload_data=train_opts.preloadData, **addson)
    
    train_loader = DataLoader(dataset=train_dataset, num_workers=2, pin_memory=True, batch_size=train_opts.batchSize, shuffle=True)
    valid_loader = DataLoader(dataset=valid_dataset, num_workers=2, pin_memory=True, batch_size=train_opts.batchSize, shuffle=False)
    

    # Visualisation Parameters
    visualizer = Visualiser(json_opts.visualisation, save_dir=model.save_dir)
    log_file = visualizer.log_name
    error_logger = ErrorLogger()
    with open(log_file, "a") as log:
        log.write("Random Seed: " + str(arguments.seed) + "\n")
        log.write("Fold: " + str(json_opts.data_path.fold) + "\n")
        log.write("Trainset: " + ", ".join(train_dataset.image_filenames) + "\n")
        log.write("Validset: " + ", ".join(valid_dataset.image_filenames) + "\n")

    best_dice_avg = {'epoch': 0, 'Dice_avg': 0.0}
    best_hd95_avg = {'epoch': 0, 'HD95_avg': 100.0}
    loss_history = []
    dice_avg_history = []
    if train_opts.lr_policy is not None and train_opts.lr_policy != 'polyLR':
        model.set_scheduler(train_opts)
    if resume:
        with open(os.path.join(model.save_dir, str(json_opts.model.which_epoch)+'_history.pkl'), 'rb') as f:
            history_dict = pickle.load(f)
            best_dice_avg['Dice_avg'] = history_dict['best_dice_avg']['Dice_avg']
            best_dice_avg['epoch'] = history_dict['best_dice_avg']['epoch']
            if 'best_hd95_avg' in history_dict:
                best_hd95_avg['HD95_avg'] = history_dict['best_hd95_avg']['HD95_avg']
                best_hd95_avg['epoch'] = history_dict['best_hd95_avg']['epoch']
            loss_history = history_dict['loss_history']
            dice_avg_history = history_dict['dice_avg_history']
            start_epoch = len(loss_history)
            print('Start epoch: ', start_epoch)
        if train_opts.lr_policy is not None:
            if train_opts.lr_policy != 'polyLR':
                for _ in range(start_epoch):
                    model.update_learning_rate()
        with open(log_file, "a") as log:
            log.write('==== Loading the model {0} - epoch {1} ===='.format(path_pre_trained_model, start_epoch))
    else:
        start_epoch = model.which_epoch
        copy(json_filename, model.save_dir)
        model_filename = os.path.join('models', 'networks', inspect.getfile(get_model_instance(json_opts.model.model_type, '3D')))
        copy(model_filename, model.save_dir)
        loss_filename = os.path.join('models', 'layers', 'edge_loss.py')
        copy(loss_filename, model.save_dir)
        if os.path.isdir(os.path.join(model.save_dir, 'models')):
            shutil.rmtree(os.path.join(model.save_dir, 'models'))
        shutil.copytree('/nvme1date/ghj/lmynet/models', os.path.join(model.save_dir, 'models'))
        with open(log_file, "a") as log:
            print(model.net, file=log)

    time_elapsed = {'train': '', 'validation': ''}
    # Training Function
    for epoch in range(start_epoch, train_opts.n_epochs):
        tp.banner(" "*75 + "EPOCH %d" % epoch + " "*75)
        print('Training...')
        # Training Iterations
        time_a = time()
        for epoch_iter, (images, labels, size) in enumerate(tqdm(train_loader), 1):
            # Make a training update
            model.set_input(images, labels, size)
            model.net.train()
            model.forward(split='train')
            for optimizer in model.optimizers:
                optimizer.zero_grad()
            model.backward()

            model.optimize_parameters()

            # Error visualisation
            errors = model.get_current_errors()
            error_logger.update(errors, split='train')

        time_elapsed['train'] = str(timedelta(seconds=time() - time_a))


        # Validation and Testing Iterations
        print('Validating...')
        time_a = time()
        for loader, split in zip([valid_loader], ['validation']):
            for epoch_iter, (images, labels, size) in enumerate(tqdm(loader), 1):
                with torch.no_grad():
                    # Make a forward pass with the model
                    model.set_input(images, labels, size)
                    model.validate()

                    # Error visualisation
                    errors = model.get_current_errors()
                    stats = model.get_segmentation_stats()
                    error_logger.update({**errors, **stats}, split=split)

                    # Visualise predictions
                    if json_opts.visualisation.display_id != 0:
                        visuals = model.get_current_visuals()
                        visualizer.display_current_results(visuals, epoch=epoch, save_result=False)

        time_elapsed['validation'] = str(timedelta(seconds=time() - time_a))

        current_dice_avg = error_logger.get_errors('validation')['Dice_avg']
        current_hd95_avg = error_logger.get_errors('validation')['HD95_avg']
        # if epoch % train_opts.log_epoch_freq == 0:
            # error and dice plot writing to the disk
        current_loss_log = []
        history_plot_labels = []
        for error_split in ['train', 'validation']:
            error_dict = error_logger.get_errors(error_split)
            for key in error_dict.keys():
                if key.endswith('_Loss'):
                    current_loss_log.append(error_dict[key] if error_dict[key] < 10.0 else 0)
                    history_plot_labels.append(error_split+'_'+key)
        loss_history.append(current_loss_log)
        dice_avg_history.append(current_dice_avg)
        history_ploter(loss_history, os.path.join(model.save_dir, 'loss.png'), train_opts.n_epochs, history_plot_labels)
        history_ploter(dice_avg_history, os.path.join(model.save_dir, 'dice.png'), train_opts.n_epochs)

        if current_dice_avg > best_dice_avg['Dice_avg']:
            best_dice_avg['epoch'] = epoch
            best_dice_avg['Dice_avg'] = current_dice_avg
            save_history_model_opt(best_dice_avg, loss_history, dice_avg_history, best_hd95_avg, model, 'best_dice')
            message = "Best Dice_avg is %.3f at the end of epoch %d." % (best_dice_avg['Dice_avg'], best_dice_avg['epoch'])
            # print(message)
            with open(log_file, "a") as log:
                log.write('%s\n' % message)
                
        if current_hd95_avg < best_hd95_avg['HD95_avg']:
            best_hd95_avg['epoch'] = epoch
            best_hd95_avg['HD95_avg'] = current_hd95_avg
            save_history_model_opt(best_dice_avg, loss_history, dice_avg_history, best_hd95_avg, model, 'best_hd95')
            message = "Best HD95_avg is %.3f at the end of epoch %d." % (current_hd95_avg, epoch)
            # print(message)
            with open(log_file, "a") as log:
                log.write('%s\n' % message)

        # Update the plots
        for split in ['train', 'validation']:
            if json_opts.visualisation.display_id != 0:
                visualizer.plot_current_errors(epoch, error_logger.get_errors(split), split_name=split)
            visualizer.print_current_errors(epoch, error_logger.get_errors(split), split_name=split, time_elapsed=time_elapsed[split])

        # Make Table
        col_names = ['epoch', 'train time', 'val time']
        col_value = [epoch, time_elapsed['train'], time_elapsed['validation']]
        for k, v in error_logger.get_errors('train').items():
            if np.isscalar(v):
                # message += '%s: %.3f ' % (k, v)
                col_names.append('train_'+k)
                col_value.append('%.3f' % v)

        i = 0
        for k, v in error_logger.get_errors('validation').items():
            if np.isscalar(v) and i < 2:
                # message += '%s: %.3f ' % (k, v)
                col_names.append('val_'+k)
                col_value.append('%.3f' % v)
                i += 1
        tp.table([col_value], col_names)

        col_names_dice = ['epoch']
        col_value_dice = [epoch]
        col_names_hd = ['epoch']
        col_value_hd = [epoch]
        i = 0
        for k, v in error_logger.get_errors('validation').items():
            if np.isscalar(v) and i < 2:
                # message += '%s: %.3f ' % (k, v)
                # col_names.append('val_'+k)
                # col_value.append('%.3f' % v)
                i += 1
            else:
                if np.isscalar(v):
                    if i % 2 == 0:
                        col_names_dice.append(k)
                        col_value_dice.append('%.3f' % v)
                    else:
                        col_names_hd.append(k)
                        col_value_hd.append('%.3f' % v)
                    i += 1
        tp.table([col_value_dice], col_names_dice)
        tp.table([col_value_hd], col_names_hd)

        # Save the model parameters
        # if epoch % train_opts.save_epoch_freq == 0:
        #     save_history_model_opt(best_dice_avg, loss_history, dice_avg_history, model, epoch)

        error_logger.reset()
        save_history_model_opt(best_dice_avg, loss_history, dice_avg_history, best_hd95_avg, model, 'tmp')
        # Update the model learning rate
        if best_dice_avg['Dice_avg'] > 0.7 and train_opts.lr_policy is not None:
            model.update_learning_rate(epoch=epoch, policy=train_opts.lr_policy)


    message = "Best Dice_avg is %.3f at the end of epoch %d." % (best_dice_avg['Dice_avg'], best_dice_avg['epoch'])
    print(message)
    with open(log_file, "a") as log:
        log.write('%s\n' % message)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='CNN Seg Training Function')

    parser.add_argument('-c', '--config',  help='training config file', required=True)
    parser.add_argument('-d', '--debug',   help='returns number of parameters and bp/fp runtime', action='store_true')
    parser.add_argument('-s', '--seed',   help='manual seed', type=int, default=1)
    args = parser.parse_args()

    if not args.seed:
        args.seed = random.randint(1, 10000)
    manual_s(args.seed)

    train(args)
