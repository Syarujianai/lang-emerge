# class defintions for chatbots - questioner and answerer

import torch
import torch.nn as nn
from torch.distributions import Categorical
from torch.autograd import Variable
import torch.autograd as autograd
import sys
from utilities import initializeWeights

import ipdb, pickle
#---------------------------------------------------------------------------
# Parent class for both q and a bots
class ChatBot(nn.Module):
    def __init__(self, params):
        super(ChatBot, self).__init__();

        # absorb all parameters to self
        for attr in params: setattr(self, attr, params[attr]);

        # standard initializations
        self.hState = torch.Tensor();
        self.cState = torch.Tensor();
        self.loss = torch.Tensor(1);
        if self.useGPU:
            self.loss = self.loss.cuda();
        self.actions = [];
        self.outDistr = [];
        self.evalFlag = False;

        # modules (common)
        self.inNet = nn.Embedding(self.inVocabSize, self.embedSize);
        self.outNet = nn.Linear(self.hiddenSize, self.outVocabSize);
        self.softmax = nn.Softmax();

        # initialize weights
        initializeWeights([self.inNet, self.outNet], 'xavier');

    # initialize hidden states
    def resetStates(self, batchSize, retainActions=False):
        # create tensors
        self.hState = torch.Tensor(batchSize, self.hiddenSize);
        self.hState.fill_(0.0);
        self.hState = Variable(self.hState);
        self.cState = torch.Tensor(batchSize, self.hiddenSize);
        self.cState.fill_(0.0);
        self.cState = Variable(self.cState);

        if self.useGPU:
            self.hState = self.hState.cuda();
            self.cState = self.cState.cuda();

        # new episode
        if not retainActions: 
            self.actions = [];
            # Syaru: must reset self.outDistr, or gpu memory leak.
            self.outDistr = [];

    # freeze agent
    def freeze(self):
        for p in self.parameters(): p.requires_grad = False;
    # unfreeze agent
    def unfreeze(self):
        for p in self.parameters(): p.requires_grad = True;

    # given an input token, interact for the next round
    def listen(self, inputToken, imgEmbed = None):
        # embed and pass through LSTM
        tokenEmbeds = self.inNet(inputToken);
        # concat with image representation (case aBot)
        if imgEmbed is not None:
            tokenEmbeds = torch.cat((tokenEmbeds, imgEmbed), 1);

        # now pass it through rnn (1 timestep)
        self.hState, self.cState = self.rnn(tokenEmbeds,\
                                            (self.hState, self.cState));

    # speak a token
    def speak(self):
        # compute softmax and choose a token
        outDistr = self.softmax(self.outNet(self.hState));

        # if evaluating
        if self.evalFlag:
            _, actions = outDistr.max(1);
        else:
            actions = Categorical(outDistr).sample();
            # record actions
            self.actions.append(actions);
            # record output distribution
            self.outDistr.append(outDistr);
        return actions;

    # backward computationtorch.mean
    def performBackward(self, rewards):
        # Syaru:
        # 1. stochastic_node.reinforce() is deprecated,
        # Refer: http://pytorch.org/docs/0.3.0/distributions.html
        # 2. About stochastic_node.reinforce():
        # register reward with gradients of stochastic node.
        # Refer: https://discuss.pytorch.org/t/what-is-action-reinforce-r-doing-actually/1294
        # 3. self.actions: collection of 2 rounds qBot(speak and guess) or aBot(speak) actions.
        # 4. self. outDistr: collection of 2 rounds qBot(speak and guess) or aBot(speak) output distribution.
        # 5. Separtely backward for each actions.
        # Refer: https://blog.csdn.net/qq_17550379/article/details/78939046
        self.loss.fill_(0);
        for actions, outDistr in zip(self.actions, self.outDistr):
            # Syaru:
            # 1. How to debug gpu memory leak in Pytorch?
            # Refer: https://discuss.pytorch.org/t/how-to-debug-causes-of-gpu-memory-leaks/6741/3
            # 2. when set retain_graph=True, computational graph will automatically release when Variable out of python scope,
            # you don't need to manually release by backward(retain_graph=False).
            # Refer: https://discuss.pytorch.org/t/how-to-free-graph-manually/9255
            self.loss = torch.mean(-Categorical(outDistr).log_prob(actions) * rewards) / float(len(actions));  # gradients of log likelihood
            autograd.backward(self.loss, retain_graph=True);       

    # switch mode to evaluate
    def evaluate(self): self.evalFlag = True;

    # switch mode to train
    def train(self): self.evalFlag = False;
    
#---------------------------------------------------------------------------
class Answerer(ChatBot):
    def __init__(self, params):
        self.parent = super(Answerer, self);
        # input-output for current bot
        params['inVocabSize'] = params['aInVocab'];
        params['outVocabSize'] = params['aOutVocab'];
        self.parent.__init__(params);

        # number of attribute values
        numAttrs = sum([len(ii) for ii in self.props.values()]);
        # number of unique attributes
        numUniqAttr = len(self.props);

        # rnn inputSize
        rnnInputSize = numUniqAttr * self.imgFeatSize + self.embedSize;

        self.imgNet = nn.Embedding(numAttrs, self.imgFeatSize);
        self.rnn = nn.LSTMCell(rnnInputSize, self.hiddenSize);
        initializeWeights([self.rnn, self.imgNet], 'xavier');

        # set offset
        self.listenOffset = params['qOutVocab'];

    # Embedding the image
    def embedImage(self, batch):
        embeds = self.imgNet(batch);
                
        # Syaru: 
        # 1. Expansion by first dim, bz change to dim 1, but still 0 after cat.
        # 2. Pytorch update: https://github.com/Kaixhin/Rainbow/issues/17
        
        # concat instead of add
        features = torch.cat(list(embeds.transpose(0, 1)), 1);
        # add features
        #features = torch.sum(embeds, 1).squeeze(1);

        return features;        

#---------------------------------------------------------------------------
class Questioner(ChatBot):
    def __init__(self, params):
        self.parent = super(Questioner, self);
        # input-output for current bot
        params['inVocabSize'] = params['qInVocab'];
        params['outVocabSize'] = params['qOutVocab'];
        self.parent.__init__(params);

        # always condition on task
        #self.rnn = nn.LSTMCell(2*self.embedSize, self.hiddenSize);
        self.rnn = nn.LSTMCell(self.embedSize, self.hiddenSize);

        # additional prediction network
        # start token included
        numPreds = sum([len(ii) for ii in self.props.values()]);
        # network for predicting
        self.predictRNN = nn.LSTMCell(self.embedSize, self.hiddenSize);
        self.predictNet = nn.Linear(self.hiddenSize, numPreds);
        initializeWeights([self.predictNet, self.predictRNN, self.rnn], 'xavier');

        # setting offset
        self.taskOffset = params['aOutVocab'] + params['qOutVocab'];
        self.listenOffset = params['aOutVocab'];

    # make a guess the given image
    def guessAttribute(self, inputEmbeds):
        # compute softmax and choose a token
        self.hState, self.cState = \
                self.predictRNN(inputEmbeds, (self.hState, self.cState));
        outDistr = self.softmax(self.predictNet(self.hState));

        # if evaluating
        if self.evalFlag: _, actions = outDistr.max(1);
        else:
            # ORIGINAL: actions = outDistr.multinomial(num_samples=1);
            actions = Categorical(outDistr).sample();
            # record actions
            self.actions.append(actions);
            self.outDistr.append(outDistr);

        return actions, outDistr;

    # returning the answer, from the task
    def predict(self, tasks, numTokens):
        guessTokens = [];
        # return by team.forward
        guessDistr = [];

        for _ in range(numTokens):
            # explicit task dependence
            taskEmbeds = self.inNet(tasks);
            guess, distr = self.guessAttribute(taskEmbeds);

            # record the guess and distribution
            guessTokens.append(guess);
            guessDistr.append(distr);

        # return prediction
        return guessTokens, guessDistr;

    # Embedding the image
    def embedTask(self, tasks): return self.inNet(tasks + self.taskOffset);
    
#---------------------------------------------------------------------------
class Team:
    # initialize
    def __init__(self, params):
        # memorize params
        for field, value in params.items(): setattr(self, field, value);
        self.aBot = Answerer(params);
        self.qBot = Questioner(params);
        self.criterion = nn.NLLLoss();
        self.reward = torch.Tensor(self.batchSize, 1);
        self.totalReward = None;
        self.rlNegReward = -10*self.rlScale;

        # ship to gpu if needed
        if self.useGPU:
            self.aBot = self.aBot.cuda();
            self.qBot = self.qBot.cuda();
            self.reward = self.reward.cuda();

        print(self.aBot)
        print(self.qBot)

    # switch to train
    def train(self):
        self.aBot.train(); self.qBot.train();

    # switch to evaluate
    def evaluate(self):
        self.aBot.evaluate(); self.qBot.evaluate();

    # forward pass
    def forward(self, batch, tasks, record=False):
        # reset the states of the bots
        batchSize = batch.size(0);
        self.qBot.resetStates(batchSize);
        self.aBot.resetStates(batchSize);

        # get image representation
        imgEmbed = self.aBot.embedImage(batch);

        # ask multiple rounds of questions
        aBotReply = tasks + self.qBot.taskOffset;
        # if the conversation is to be recorded
        talk = [];
        for roundId in range(self.numRounds):
            # listen to answer, ask q_r, and listen to q_r as well
            self.qBot.listen(aBotReply);
            qBotQues = self.qBot.speak();

            # clone
            qBotQues = qBotQues.detach();
            # make this random
            self.qBot.listen(self.qBot.listenOffset + qBotQues);

            # Aer is memoryless, forget
            if not self.remember: self.aBot.resetStates(batchSize, True);
            # listen to question and answer, also listen to answer
            self.aBot.listen(qBotQues, imgEmbed);
            aBotReply = self.aBot.speak();
            aBotReply = aBotReply.detach();
            self.aBot.listen(aBotReply + self.aBot.listenOffset, imgEmbed);

            if record: talk.extend([qBotQues, aBotReply]);

        # listen to the last answer
        self.qBot.listen(aBotReply);

        # predict the image attributes, compute reward
        self.guessToken, self.guessDistr = self.qBot.predict(tasks, 2);

        return self.guessToken, self.guessDistr, talk;

    # backward pass
    def backward(self, optimizer, gtLabels, epoch, baseline=None):
        # compute reward
        self.reward.fill_(self.rlNegReward);
        
        # both attributes need to match
        firstMatch = self.guessToken[0].data == gtLabels[:, 0];
        secondMatch = self.guessToken[1].data == gtLabels[:, 1];
        self.reward[firstMatch & secondMatch] = self.rlScale;

        # optimize for qBot, aBot
        optimizer.zero_grad();
        self.qBot.performBackward(self.reward);
        self.aBot.performBackward(self.reward);

        # clamp the gradients
        for p in self.qBot.parameters(): p.grad.data.clamp_(min=-5., max=5.);
        for p in self.aBot.parameters(): p.grad.data.clamp_(min=-5., max=5.);

        # cummulative reward
        batchReward = torch.mean(self.reward)/self.rlScale;                      
        if self.totalReward is None: self.totalReward = batchReward;
        self.totalReward = 0.95 * self.totalReward + 0.05 * batchReward;

        return batchReward;

    # loading modules from saved model
    def loadModel(self, savedModel):
        modules = ['rnn', 'inNet', 'outNet', 'imgNet', \
                            'predictRNN', 'predictNet'];
        # savedModel is an instance of dict
        dictSaved = isinstance(savedModel['qBot'], dict);

        for agentName in ['aBot', 'qBot']:
            agent = getattr(self, agentName);
            for module in modules:
                if hasattr(agent, module):
                    if dictSaved: savedModule = savedModel[agentName][module];
                    else: savedModule = getattr(savedModel[agentName], module);
                    # assign to current model
                    setattr(agent, module, savedModule);

    # saving module, at given path with params and optimizer
    def saveModel(self, savePath, optimizer, params):
        modules = ['rnn', 'inNet', 'outNet', 'imgNet', \
                            'predictRNN', 'predictNet'];

        toSave = {'aBot':{}, 'qBot':{}, 'params': params, 'optims':optimizer};
        for agentName in ['aBot', 'qBot']:
            agent = getattr(self, agentName);
            for module in modules:
                if hasattr(agent, module):
                    toSaveModule = getattr(agent, module);
                    toSave[agentName][module] = toSaveModule;

        # save as pickle
        with open(savePath, 'wb') as fileId: pickle.dump(toSave, fileId);
