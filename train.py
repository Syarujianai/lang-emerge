# script to train interactive bots in toy world
# author: satwik kottur

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim

import itertools, ipdb, random, pickle, os
import numpy as np
from chatbots import Team
from dataloader import Dataloader
import options
from time import gmtime, strftime

from tensorboardX import SummaryWriter
from datetime import datetime
import os
import gc
import collections

# random seed setting
torch.manual_seed(7);
torch.cuda.manual_seed(7);
np.random.seed(7);
random.seed(7);

# tensorboard
path = './log/' + datetime.now().strftime("%Y%m%d_%H%M%S");
writer = SummaryWriter(path);

# torch cudnn setting
torch.backends.cudnn.benchmark = True;
torch.backends.cudnn.deterministic=True;

# read the command line options
options = options.read();
#------------------------------------------------------------------------
# setup experiment and dataset
#------------------------------------------------------------------------
data = Dataloader(options);
numInst = data.getInstCount();

params = data.params;
# append options from options to params
for key, value in options.items(): params[key] = value;

#------------------------------------------------------------------------
# build agents, and setup optmizer
#------------------------------------------------------------------------
team = Team(params);
team.train();
optimizer = optim.Adam([{'params': team.aBot.parameters(), \
                                'lr':params['learningRate']},\
                        {'params': team.qBot.parameters(), \
                                'lr':params['learningRate']}]);
#------------------------------------------------------------------------
# train agents
#------------------------------------------------------------------------
# begin training
numIterPerEpoch = int(np.ceil(numInst['train']/params['batchSize']));
numIterPerEpoch = max(1, numIterPerEpoch);
count = 0;
savePath = './models/tasks_inter_%dH_%.4flr_%r_%d_%d.pickle' %\
            (params['hiddenSize'], params['learningRate'], params['remember'],\
            options['aOutVocab'], options['qOutVocab']);

matches = {};
accuracy = {};
bestAccuracy = 0;

for iterId in range(params['numEpochs'] * numIterPerEpoch):
    epoch = float(iterId)/numIterPerEpoch;

    # get double attribute tasks
    # if 'train' not in matches:
    #     batchImg, batchTask, batchLabels \
    #                         = data.getBatch(params['batchSize']);
    # else:
    #     batchImg, batchTask, batchLabels \
    #             = data.getBatchSpecial(params['batchSize'], matches['train'],\
    #                                                     params['negFraction']);
    #ipdb.set_trace();

    batchImg = torch.LongTensor([[ 1,  7, 10]]).cuda();
    batchTask = torch.LongTensor([2]).cuda();
    batchLabels = torch.LongTensor([[1, 10]]).cuda();
    #ipdb.set_trace();

    # forward pass
    team.forward(Variable(batchImg), Variable(batchTask), False);
    
    # backward pass
    batchReward = team.backward(optimizer, batchLabels, epoch);

    # take a step by optimizer
    optimizer.step()

    # Syaru: record computaion graph.
    # torch.cuda.empty_cache();
    #--------------------------------------------------------------------------
    # switch to evaluate
    team.evaluate();

    # for dtype in ['train', 'test']:
    #     # get the entire batch
    #     img, task, labels = data.getCompleteData(dtype);
    #     # evaluate on the train dataset, using greedy policy
    #     guess, _, _ = team.forward(Variable(img), Variable(task));
    #     # compute accuracy for color, shape, and both
    #     firstMatch = guess[0].data == labels[:, 0].long();
    #     secondMatch = guess[1].data == labels[:, 1].long();
    #     matches[dtype] = firstMatch & secondMatch;
    #     accuracy[dtype] = 100*torch.sum(matches[dtype])\
    #                                 /float(matches[dtype].size(0));
    #ipdb.set_trace();
    #if iterId % 3000 == 0:
        #for p in team.qBot.named_parameters(): print(p, p[1].grad);
        #for p in team.aBot.named_parameters(): print(p, p[1].grad);

    for dtype in ['train', 'test']:
        img = torch.LongTensor([[ 1,  7, 10]]).cuda();
        task = torch.LongTensor([2]).cuda();
        labels = torch.LongTensor([[1, 10]]).cuda();
        # get the entire batch
        # img, task, labels = data.getCompleteData(dtype);
        # evaluate on the train dataset, using greedy policy
        guess, gd, _ = team.forward(Variable(img), Variable(task));
        if iterId % 3000 == 0 and dtype == 'train': print('Guess! ', guess, ' ', gd, '\n');        
        # compute accuracy for color, shape, and both
        firstMatch = guess[0].data == labels[:, 0].long();
        secondMatch = guess[1].data == labels[:, 1].long();
        matches[dtype] = firstMatch & secondMatch;
        accuracy[dtype] = 100*torch.sum(matches[dtype])\
                                    /float(matches[dtype].size(0));

    # switch to train
    team.train();

    # break if train accuracy reaches 100%
    if accuracy['train'] == 100: break;

    # save for every 5k epochs
    if iterId > 0 and iterId % (10000*numIterPerEpoch) == 0:
        team.saveModel(savePath, optimizer, params);

    if iterId % 100 != 0: continue;

    time = strftime("%a, %d %b %Y %X", gmtime());
    print('[%s][Iter: %d][Ep: %.2f][R: %.4f][Tr: %.2f Te: %.2f]' % \
                                (time, iterId, epoch, team.totalReward,\
                                accuracy['train'], accuracy['test']))

    # Syaru: save tensorboard summary.
    writer.add_scalar('Train/reward', team.totalReward, iterId);
    writer.add_scalar('Train/acc', accuracy['train'], iterId);
    writer.add_scalar('Test/acc', accuracy['test'], iterId);

#------------------------------------------------------------------------
# save final model with a time stamp
timeStamp = strftime("%a-%d-%b-%Y-%X", gmtime());
replaceWith = 'final_%s' % timeStamp;
finalSavePath = savePath.replace('inter', replaceWith);
print('Saving : ' + finalSavePath)
team.saveModel(finalSavePath, optimizer, params);
#------------------------------------------------------------------------
