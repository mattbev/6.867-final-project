import argparse
import json
import datetime
import os
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import math

from torchvision import transforms
import torchvision.utils as vutils

from image_helper import ImageHelper
from text_helper import TextHelper

from utils.utils import dict_html

from models.dconv import _netG

logger = logging.getLogger("logger")
# logger.setLevel("ERROR")
import yaml
import time
import visdom
import numpy as np

vis = visdom.Visdom()
import random
from utils.text_load import *

criterion = torch.nn.CrossEntropyLoss()


# torch.manual_seed(1)
# torch.cuda.manual_seed(1)
# random.seed(1)

def _one_hot(x, num_classes=10):
    '''
    One-hot encoding of the vector of classes. It uses number of classes + 1 to
    encode fake images
    :param x: vector of output classes to one-hot encode
    :return: one-hot encoded version of the input vector
    '''
    label_numpy = x.data.cpu().numpy()
    label_onehot = np.zeros((label_numpy.shape[0], num_classes + 1))
    label_onehot[np.arange(label_numpy.shape[0]), label_numpy] = 1
    label_onehot = torch.FloatTensor(label_onehot).cuda()
    return label_onehot


def _to_var(x):
    '''
    Creates a variable for a tensor
    :param x: PyTorch Tensor
    :return: Variable that wraps the tensor
    '''
    x = x.cuda()
    return Variable(x)


def train(helper, epoch, train_data_sets, local_model, target_model, is_poison, last_weight_accumulator=None):
    ### Accumulate weights for all participants.
    weight_accumulator = dict()
    for name, data in target_model.state_dict().items():
        #### don't scale tied weights:
        if helper.params.get('tied', False) and name == 'decoder.weight' or '__' in name:
            continue
        weight_accumulator[name] = torch.zeros_like(data)

    ### This is for calculating distances
    target_params_variables = dict()
    for name, param in target_model.named_parameters():
        target_params_variables[name] = target_model.state_dict()[name].clone().detach().requires_grad_(False)
    current_number_of_adversaries = 0
    for model_id, _ in train_data_sets:
        if model_id == -1 or model_id in helper.params['adversary_list']:
            current_number_of_adversaries += 1
    logger.info(f'There are {current_number_of_adversaries} adversaries in the training.')

    for model_id in range(helper.params['no_models']):
        model = local_model
        ## Synchronize LR and models
        model.copy_params(target_model.state_dict())
        # optimizer = torch.optim.SGD(model.parameters(), lr=helper.params['lr'],
        #                             momentum=helper.params['momentum'],
        #                             weight_decay=helper.params['decay'])
        optimizer = torch.optim.Adam(model.parameters(), .0002, betas=(.5, 0.999))
        model.train()

        start_time = time.time()
        if helper.params['type'] == 'text':
            current_data_model, train_data = train_data_sets[model_id]
            ntokens = len(helper.corpus.dictionary)
            hidden = model.init_hidden(helper.params['batch_size'])
        else:
            _, (current_data_model, train_data) = train_data_sets[model_id]
        batch_size = helper.params['batch_size']
        ### For a 'poison_epoch' we perform single shot poisoning

        if current_data_model == -1:
            ### The participant got compromised and is out of the training.
            #  It will contribute to poisoning,
            continue
        if is_poison and current_data_model in helper.params['adversary_list'] and \
                (epoch >= helper.params['poison_epochs'] or helper.params['random_compromise']):
            logger.info('poison_now')
            poisoned_data = helper.poisoned_data_for_train

            _, acc_p = test_poison(helper=helper, epoch=epoch,
                                   data_source=helper.test_data_poison,
                                   model=model, is_poison=True, visualize=False)
            _, acc_initial = test(helper=helper, epoch=epoch, data_source=helper.test_data,
                                  model=model, is_poison=False, visualize=False)
            logger.info(acc_p)
            poison_lr = helper.params['poison_lr']
            if not helper.params['baseline']:
                if acc_p > 80:
                    poison_lr /= 5
                if acc_p > 90:
                    poison_lr /= 10

            retrain_no_times = helper.params['retrain_poison']
            step_lr = helper.params['poison_step_lr']

            # poison_optimizer = torch.optim.SGD(model.parameters(), lr=poison_lr,
            #                                    momentum=helper.params['momentum'],
            #                                    weight_decay=helper.params['decay'])
            poison_optimizer = torch.optim.Adam(model.parameters(), poison_lr, betas=(.5, 0.999))

            is_stepped = False
            is_stepped_15 = False
            saved_batch = None
            acc = acc_initial
            try:
                for internal_epoch in range(1, retrain_no_times + 1):
                    if helper.params['type'] == 'text':
                        data_iterator = range(0, poisoned_data.size(0) - 1, helper.params['bptt'])
                    else:
                        data_iterator = poisoned_data

                    for batch_id, batch in enumerate(data_iterator):
                        for i in range(1):
                            poison_l = random.sample(helper.poison_list, 15)
                            for pos in range(15):
                                poison_pos = 15 * i + pos
                                batch[0][poison_pos] = helper.train_dataset[poison_l[pos]][0]
                                # print(poison_l[pos])
                                # batch[0][poison_pos].add_(torch.FloatTensor(batch[0][poison_pos].shape).normal_(0, helper.params['noise_level']))
                                batch[1][poison_pos] = helper.params['poison_label_swap']

                        poison_optimizer.zero_grad()
                        data, targets = helper.get_batch(train_data, batch, evaluation=False)
                        if helper.params['type'] == 'text':
                            hidden = helper.repackage_hidden(hidden)
                            output, hidden = model(data, hidden)
                            loss = criterion(output.view(-1, ntokens), targets)
                        else:
                            d_out, d_class_logits_on_data, d_gan_logits_real, d_sample_features = model(data)
                            d_gan_labels_real = torch.LongTensor(64)
                            d_gan_labels_real.resize_as_(targets.data.cpu()).fill_(1)
                            d_gan_labels_real_var = _to_var(d_gan_labels_real).float()
                            d_gan_criterion = nn.BCEWithLogitsLoss()
                            d_gan_loss_real = d_gan_criterion(d_gan_logits_real, d_gan_labels_real_var)

                            svhn_labels_one_hot = _one_hot(targets)
                            d_class_loss_entropy = -torch.sum(svhn_labels_one_hot * torch.log(d_out), dim=1)
                            d_class_loss_entropy = d_class_loss_entropy.squeeze()
                            d_class_loss = torch.sum(d_class_loss_entropy)

                            loss = d_gan_loss_real + d_class_loss

                        loss.backward(retain_graph=True)
                        poison_optimizer.step()

                    loss_p, acc_p = test_poison(helper=helper, epoch=internal_epoch,
                                                data_source=helper.test_data_poison,
                                                model=model, is_poison=True, visualize=False)
                    _, acc = test(helper=helper, epoch=epoch, data_source=helper.test_data,
                                  model=model, is_poison=False, visualize=False)
                    logger.error(
                        f'Distance: {helper.model_dist_norm(model, target_params_variables)}')
            except ValueError:
                logger.info('Converged earlier')

            logger.info(f'Global model norm: {helper.model_global_norm(target_model)}.')
            logger.info(f'Norm before scaling: {helper.model_global_norm(model)}. '
                        f'Distance: {helper.model_dist_norm(model, target_params_variables)}')

            ### Adversary wants to scale his weights. Baseline model doesn't do this
            if not helper.params['baseline']:
                ### We scale data according to formula: L = 100*X-99*G = G + (100*X- 100*G).
                clip_rate = (helper.params['scale_weights'] / current_number_of_adversaries)
                logger.info(f"Scaling by  {clip_rate}")
                for key, value in model.state_dict().items():
                    #### don't scale tied weights:
                    if helper.params.get('tied', False) and key == 'decoder.weight' or '__' in key:
                        continue
                    target_value = target_model.state_dict()[key]
                    new_value = target_value + (value - target_value) * clip_rate

                    model.state_dict()[key].copy_(new_value)
                distance = helper.model_dist_norm(model, target_params_variables)
                logger.info(
                    f'Scaled Norm after poisoning: '
                    f'{helper.model_global_norm(model)}, distance: {distance}')

            if helper.params['diff_privacy']:
                model_norm = helper.model_dist_norm(model, target_params_variables)

                if model_norm > helper.params['s_norm']:
                    norm_scale = helper.params['s_norm'] / (model_norm)
                    for name, layer in model.named_parameters():
                        #### don't scale tied weights:
                        if helper.params.get('tied', False) and name == 'decoder.weight' or '__' in name:
                            continue
                        clipped_difference = norm_scale * (
                                layer.data - target_model.state_dict()[name])
                        layer.data.copy_(target_model.state_dict()[name] + clipped_difference)
                distance = helper.model_dist_norm(model, target_params_variables)
                logger.info(
                    f'Scaled Norm after poisoning and clipping: '
                    f'{helper.model_global_norm(model)}, distance: {distance}')

            for key, value in model.state_dict().items():
                #### don't scale tied weights:
                if helper.params.get('tied', False) and key == 'decoder.weight' or '__' in key:
                    continue
                target_value = target_model.state_dict()[key]
                new_value = target_value + (value - target_value) * current_number_of_adversaries
                model.state_dict()[key].copy_(new_value)
            distance = helper.model_dist_norm(model, target_params_variables)
            logger.info(f"Total norm for {current_number_of_adversaries} "
                        f"adversaries is: {helper.model_global_norm(model)}. distance: {distance}")

        else:
            ### we will load helper.params later
            if helper.params['fake_participants_load']:
                continue
            # dloss_sum = .0
            # dloss_num = .0
            for internal_epoch in range(1, helper.params['retrain_no_times'] + 1):
                total_loss = 0.
                if helper.params['type'] == 'text':
                    data_iterator = range(0, train_data.size(0) - 1, helper.params['bptt'])
                else:
                    data_iterator = train_data
                for batch_id, batch in enumerate(data_iterator):
                    optimizer.zero_grad()
                    data, targets = helper.get_batch(train_data, batch,
                                                     evaluation=False)
                    if helper.params['type'] == 'text':
                        hidden = helper.repackage_hidden(hidden)
                        output, hidden = model(data, hidden)
                        loss = criterion(output.view(-1, ntokens), targets)
                    else:
                        # output = model(data)
                        # # print(output,"sssssssss",targets)
                        # loss = nn.functional.cross_entropy(output, targets)

                        d_out, d_class_logits_on_data, d_gan_logits_real, d_sample_features = model(data)
                        d_gan_labels_real = torch.LongTensor(64)
                        d_gan_labels_real.resize_as_(targets.data.cpu()).fill_(1)
                        d_gan_labels_real_var = _to_var(d_gan_labels_real).float()
                        d_gan_criterion = nn.BCEWithLogitsLoss()
                        d_gan_loss_real = d_gan_criterion(d_gan_logits_real, d_gan_labels_real_var)

                        svhn_labels_one_hot = _one_hot(targets)
                        d_class_loss_entropy = -torch.sum(svhn_labels_one_hot * torch.log(d_out), dim=1)
                        d_class_loss_entropy = d_class_loss_entropy.squeeze()
                        d_class_loss = torch.sum(d_class_loss_entropy)

                        loss = d_gan_loss_real + d_class_loss
                        # dloss_sum += d_gan_loss_real
                        # dloss_num += 1

                    loss.backward(retain_graph=True)
                    optimizer.step()

                    total_loss += loss.data

                    if helper.params["report_train_loss"] and batch % helper.params[
                        'log_interval'] == 0 and batch > 0:
                        cur_loss = total_loss.item() / helper.params['log_interval']
                        elapsed = time.time() - start_time
                        logger.info('model {} | epoch {:3d} | internal_epoch {:3d} '
                                    '| {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                                    'loss {:5.2f} | ppl {:8.2f}'
                                    .format(model_id, epoch, internal_epoch,
                                            batch, train_data.size(0) // helper.params['bptt'],
                                            helper.params['lr'],
                                            elapsed * 1000 / helper.params['log_interval'],
                                            cur_loss,
                                            math.exp(cur_loss) if cur_loss < 30 else -1.))
                        total_loss = 0
                        start_time = time.time()
                    # logger.info(f'model {model_id} distance: {helper.model_dist_norm(model, target_params_variables)}')

            # print("dloss{}:{}".format(epoch,dloss_sum/dloss_num))

        for name, data in model.state_dict().items():
            #### don't scale tied weights:
            if helper.params.get('tied', False) and name == 'decoder.weight' or '__' in name:
                continue
            weight_accumulator[name].add_(data - target_model.state_dict()[name])

        # traind generative network for each participant

    if helper.params["fake_participants_save"]:
        torch.save(weight_accumulator,
                   f"{helper.params['fake_participants_file']}_"
                   f"{helper.params['s_norm']}_{helper.params['no_models']}")
    elif helper.params["fake_participants_load"]:
        fake_models = helper.params['no_models'] - helper.params['number_of_adversaries']
        fake_weight_accumulator = torch.load(
            f"{helper.params['fake_participants_file']}_{helper.params['s_norm']}_{fake_models}")
        logger.info(f"Faking data for {fake_models}")
        for name in target_model.state_dict().keys():
            #### don't scale tied weights:
            if helper.params.get('tied', False) and name == 'decoder.weight' or '__' in name:
                continue
            weight_accumulator[name].add_(fake_weight_accumulator[name])

    return weight_accumulator


def train_net_g(helper, epoch):
    indices = random.sample(list(range(len(helper.gen_dataset))), 64)
    print(len(helper.gen_dataset))
    svhn_loader_gen = torch.utils.data.DataLoader(
        dataset=helper.gen_dataset,
        batch_size=64,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices),
        drop_last=True)
    gloss_sum = .0
    gloss_num = .0
    dloss_sum = .0
    dloss_num = .0
    model = helper.target_model
    g_optimizer = torch.optim.Adam(helper.g_model.parameters(), .0002, betas=(.5, 0.999))
    d_optimizer = torch.optim.Adam(model.parameters(), .0002, betas=(.5, 0.999))
    model.train()
    helper.g_model.train()
    for batch_id, batch in enumerate(svhn_loader_gen):
        data, targets = batch
        data = data.cuda()
        targets = targets.cuda()
        d_optimizer.zero_grad()

        d_out, d_class_logits_on_data, d_gan_logits_real, d_sample_features = model(data)
        d_gan_labels_real = torch.LongTensor(64)
        d_gan_labels_real.resize_as_(targets.data.cpu()).fill_(1)
        d_gan_labels_real_var = _to_var(d_gan_labels_real).float()
        d_gan_criterion = nn.BCEWithLogitsLoss()
        d_gan_loss_real = d_gan_criterion(d_gan_logits_real, d_gan_labels_real_var)

        # train with fake images
        noise = torch.FloatTensor(64, 100, 1, 1).normal_(0, 1)
        noise_var = _to_var(noise)
        fake = helper.g_model(noise_var)
        # call detach() to avoid backprop for netG here     
        _, _, d_gan_logits_fake, _ = model(fake.detach())
        d_gan_labels_fake = torch.LongTensor(64).resize_(64).fill_(0)
        d_gan_labels_fake_var = _to_var(d_gan_labels_fake).float()
        d_gan_criterion = nn.BCEWithLogitsLoss()
        d_gan_loss_fake = d_gan_criterion(d_gan_logits_fake, d_gan_labels_fake_var)

        d_loss = d_gan_loss_real + d_gan_loss_fake

        dloss_sum += d_loss
        dloss_num += 1

        d_loss.backward()
        d_optimizer.step()

        d_optimizer.zero_grad()
        g_optimizer.zero_grad()
        # train with fake images
        # noise = torch.FloatTensor(64, 100, 1, 1).normal_(0, 1)
        # noise_var = _to_var(noise)
        # fake = helper.g_model(noise_var)

        # train g model  
        _, _, g_gan_logits, d_data_features = model(fake)

        g_loss = -torch.mean(g_gan_logits)
        gloss_sum += g_loss
        gloss_num += 1
        g_loss.backward()
        g_optimizer.step()

    print("gloss{}:{}".format(epoch, gloss_sum / gloss_num))
    print("dloss{}:{}".format(epoch, dloss_sum / dloss_num))
    noise = torch.FloatTensor(64, 100, 1, 1).normal_(0, 1)
    noise_var = _to_var(noise)
    fake = helper.g_model(noise_var)
    vutils.save_image(fake.data, '{}/fake_samples_epoch_{:03d}.png'.format("./mout2", epoch), normalize=True)


def test(helper, epoch, data_source, model, is_poison=False, visualize=True):
    model.eval()
    total_loss = 0
    correct = 0
    total_test_words = 0
    if helper.params['type'] == 'text':
        hidden = model.init_hidden(helper.params['test_batch_size'])
        random_print_output_batch = \
            random.sample(range(0, (data_source.size(0) // helper.params['bptt']) - 1), 1)[0]
        data_iterator = range(0, data_source.size(0) - 1, helper.params['bptt'])
        dataset_size = len(data_source)
    else:
        dataset_size = len(data_source) * helper.params['test_batch_size']
        data_iterator = data_source

    for batch_id, batch in enumerate(data_iterator):
        data, targets = helper.get_batch(data_source, batch, evaluation=True)

        output = model(data)[0]
        total_loss += nn.functional.cross_entropy(output, targets,
                                                  reduction='sum').item()  # sum up batch loss
        pred = output.data.max(1)[1]  # get the index of the max log-probability
        correct += pred.eq(targets.data.view_as(pred)).cpu().sum().item()

    if helper.params['type'] == 'text':
        acc = 100.0 * (correct / total_test_words)
        total_l = total_loss.item() / (dataset_size - 1)
        logger.info('___Test {} poisoned: {}, epoch: {}: Average loss: {:.4f}, '
                    'Accuracy: {}/{} ({:.4f}%)'.format(model.name, is_poison, epoch,
                                                       total_l, correct, total_test_words,
                                                       acc))
        acc = acc.item()
        total_l = total_l.item()
    else:
        acc = 100.0 * (float(correct) / float(dataset_size))
        total_l = total_loss / dataset_size

        logger.info('___Test {} poisoned: {}, epoch: {}: Average loss: {:.4f}, '
                    'Accuracy: {}/{} ({:.4f}%)'.format(model.name, is_poison, epoch,
                                                       total_l, correct, dataset_size,
                                                       acc))

    if visualize:
        model.visualize(vis, epoch, acc, total_l if helper.params['report_test_loss'] else None,
                        eid=helper.params['environment_name'], is_poisoned=is_poison)
    model.train()
    return (total_l, acc)


def test_poison(helper, epoch, data_source, model, is_poison=False, visualize=True):
    model.eval()
    total_loss = 0
    correct = 0
    # dataset_size = len(data_source) * helper.params['test_batch_size']
    data_iterator = data_source
    dataset_size = 0

    for batch_id, batch in enumerate(data_iterator):
        data, targets = helper.get_batch(data_source, batch, evaluation=True)

        output = model(data)[0]
        total_loss += nn.functional.cross_entropy(output, targets,
                                                  reduction='sum').item()  # sum up batch loss
        pred = output.data.max(1)[1]  # get the index of the max log-probability
        for pos, imgage in enumerate(targets.data):
            if imgage == 1:
                if pred[pos] == helper.params['poison_label_swap']:
                    correct += 1
                dataset_size += 1
        # correct += pred.eq(targets.data.view_as(pred)).cpu().sum().item()

    acc = 100.0 * (float(correct) / float(dataset_size))
    total_l = total_loss / dataset_size

    logger.info('___Test {} poisoned: {}, epoch: {}: Average loss: {:.4f}, '
                'Accuracy: {}/{} ({:.4f}%)'.format(model.name, is_poison, epoch,
                                                   total_l, correct, dataset_size,
                                                   acc))

    if visualize:
        model.visualize(vis, epoch, acc, total_l if helper.params['report_test_loss'] else None,
                        eid=helper.params['environment_name'], is_poisoned=is_poison)
    model.train()
    return total_l, acc

# def test_poison(helper, epoch, data_source, model, is_poison=False, visualize=True):
#     model.eval()
#     total_loss = 0.0
#     correct = 0.0
#     total_test_words = 0.0
#     batch_size = helper.params['test_batch_size']
#     if helper.params['type'] == 'text':
#         ntokens = len(helper.corpus.dictionary)
#         hidden = model.init_hidden(batch_size)
#         data_iterator = range(0, data_source.size(0) - 1, helper.params['bptt'])
#         dataset_size = len(data_source)
#     else:
#         data_iterator = data_source
#         dataset_size = len(data_source) * helper.params['test_batch_size']
#
#     for batch_id, batch in enumerate(data_iterator):
#         # if helper.params['type'] == 'image':
#         #     for pos in range(len(batch[1])):
#         #         batch[1][pos] = helper.params['poison_label_swap']
#
#         data, targets = helper.get_batch(data_source, batch, evaluation=True)
#         if helper.params['type'] == 'text':
#             output, hidden = model(data, hidden)
#             output_flat = output.view(-1, ntokens)
#             total_loss += 1 * criterion(output_flat[-batch_size:], targets[-batch_size:]).data
#             hidden = helper.repackage_hidden(hidden)
#             pred = output_flat.data.max(1)[1][-batch_size:]
#
#             correct_output = targets.data[-batch_size:]
#             correct += pred.eq(correct_output).sum()
#             total_test_words += batch_size
#         else:
#             output = model(data)[0]
#             total_loss += nn.functional.cross_entropy(output, targets,
#                                                       reduction='sum').data.item()  # sum up batch loss
#             pred = output.data.max(1)[1]  # get the index of the max log-probability
#             correct += pred.eq(targets.data.view_as(pred)).cpu().sum().to(dtype=torch.float)
#
#     if helper.params['type'] == 'text':
#         acc = 100.0 * (correct / total_test_words)
#         total_l = total_loss.item() / dataset_size
#     else:
#         acc = 100.0 * (correct / dataset_size)
#         total_l = total_loss / dataset_size
#     logger.info('PPPPPPP___Test {} poisoned: {}, epoch: {}: Average loss: {:.4f}, '
#                 'Accuracy: {}/{} ({:.0f}%)'.format(model.name, is_poison, epoch,
#                                                    total_l, correct, dataset_size,
#                                                    acc))
#     if visualize:
#         model.visualize(vis, epoch, acc, total_l if helper.params['report_test_loss'] else None,
#                         eid=helper.params['environment_name'], is_poisoned=is_poison)
#     model.train()
#     return total_l, acc


if __name__ == '__main__':
    print('Start training')
    time_start_load_everything = time.time()

    parser = argparse.ArgumentParser(description='PPDL')
    parser.add_argument('--params', dest='params')
    args = parser.parse_args()

    with open(f'./{args.params}', 'r') as f:
        params_loaded = yaml.load(f)
    current_time = datetime.datetime.now().strftime('%b.%d_%H.%M.%S')
    if params_loaded['type'] == "image":
        helper = ImageHelper(current_time=current_time, params=params_loaded,
                             name=params_loaded.get('name', 'image'))
    else:
        helper = TextHelper(current_time=current_time, params=params_loaded,
                            name=params_loaded.get('name', 'text'))

    helper.load_data()
    helper.create_model()

    ### Create models
    if helper.params['is_poison']:
        helper.params['adversary_list'] = [0] + \
                                          random.sample(range(helper.params['number_of_total_participants']),
                                                        helper.params['number_of_adversaries'] - 1)
        logger.info(f"Poisoned following participants: {len(helper.params['adversary_list'])}")
    else:
        helper.params['adversary_list'] = list()

    best_loss = float('inf')
    vis.text(text=dict_html(helper.params, current_time=helper.params["current_time"]),
             env=helper.params['environment_name'], opts=dict(width=300, height=400))
    logger.info(f"We use following environment for graphs:  {helper.params['environment_name']}")
    participant_ids = range(len(helper.train_data))
    mean_acc = list()

    results = {'poison': list(), 'number_of_adversaries': helper.params['number_of_adversaries'],
               'poison_type': helper.params['poison_type'], 'current_time': current_time,
               'sentence': helper.params.get('poison_sentences', False),
               'random_compromise': helper.params['random_compromise'],
               'baseline': helper.params['baseline']}

    weight_accumulator = None

    # save parameters:
    with open(f'{helper.folder_path}/params.yaml', 'w') as f:
        yaml.dump(helper.params, f)
    dist_list = list()
    for epoch in range(helper.start_epoch, helper.params['epochs'] + 1):
        start_time = time.time()

        if helper.params["random_compromise"]:
            # randomly sample adversaries.
            subset_data_chunks = random.sample(participant_ids, helper.params['no_models'])

            ### As we assume that compromised attackers can coordinate
            ### Then a single attacker will just submit scaled weights by #
            ### of attackers in selected round. Other attackers won't submit.
            ###
            already_poisoning = False
            for pos, loader_id in enumerate(subset_data_chunks):
                if loader_id in helper.params['adversary_list']:
                    if already_poisoning:
                        logger.info(f'Compromised: {loader_id}. Skipping.')
                        subset_data_chunks[pos] = -1
                    else:
                        logger.info(f'Compromised: {loader_id}')
                        already_poisoning = True
        ## Only sample non-poisoned participants until poisoned_epoch
        else:
            if epoch >= helper.params['poison_epochs']:
                ### For poison epoch we put one adversary and other adversaries just stay quiet
                subset_data_chunks = [participant_ids[0]] + [-1] * (
                        helper.params['number_of_adversaries'] - 1) + \
                                     random.sample(participant_ids[1:],
                                                   helper.params['no_models'] - helper.params[
                                                       'number_of_adversaries'])
            else:
                subset_data_chunks = random.sample(participant_ids[1:], helper.params['no_models'])
                logger.info(f'Selected models: {subset_data_chunks}')
        t = time.time()
        weight_accumulator = train(helper=helper, epoch=epoch,
                                   train_data_sets=[(pos, helper.train_data[pos]) for pos in
                                                    subset_data_chunks],
                                   local_model=helper.local_model, target_model=helper.target_model,
                                   is_poison=helper.params['is_poison'], last_weight_accumulator=weight_accumulator)
        logger.info(f'time spent on training: {time.time() - t}')
        # Average the models
        helper.average_shrink_models(target_model=helper.target_model,
                                     weight_accumulator=weight_accumulator, epoch=epoch)
        train_net_g(helper, epoch)

        if helper.params['is_poison']:
            epoch_loss_p, epoch_acc_p = test_poison(helper=helper,
                                                    epoch=epoch,
                                                    data_source=helper.test_data_poison,
                                                    model=helper.target_model, is_poison=True,
                                                    visualize=True)
            mean_acc.append(epoch_acc_p)
            results['poison'].append({'epoch': epoch, 'acc': epoch_acc_p})

        epoch_loss, epoch_acc = test(helper=helper, epoch=epoch, data_source=helper.test_data,
                                     model=helper.target_model, is_poison=False, visualize=True)

        helper.save_model(epoch=epoch, val_loss=epoch_loss)

        logger.info(f'Done in {time.time() - start_time} sec.')

    if helper.params['is_poison']:
        logger.info(f'MEAN_ACCURACY: {np.mean(mean_acc)}')
    logger.info('Saving all the graphs.')
    logger.info(f"This run has a label: {helper.params['current_time']}. "
                f"Visdom environment: {helper.params['environment_name']}")

    if helper.params.get('results_json', False):
        with open(helper.params['results_json'], 'a') as f:
            if len(mean_acc):
                results['mean_poison'] = np.mean(mean_acc)
            f.write(json.dumps(results) + '\n')

    vis.save([helper.params['environment_name']])
