clear;clc;
load('data/input.mat');
load('data/f0.mat');

train_x = input(1:350000,:);
train_y = f0(1:350000,:) / 399.9851;

train_xx = input(350001:440000,:);
train_yy = f0(350001:440000,:) / 399.9851;

val_x = input(440001:460200,:);
val_y = f0(440001:460200,:)/ 399.9851;
test_x = input(1001:1200,:);
test_y  = f0(1001:1200,:) / 399.9851;

train_xxx = input(1:400000,:);
train_yyy = f0(1:400000,:) / 399.9851;

%% ex2 train a 100-100 hidden unit SDAE and use it to initialize a FFNN
%  Setup and train a stacked denoising autoencoder (SDAE)
rng(0);
sae = saesetup([220 100 100]);
sae.ae{1}.activation_function       = 'sigm';
sae.ae{1}.learningRate              = 1;
sae.ae{1}.inputZeroMaskedFraction   = 0.5;

sae.ae{2}.activation_function       = 'sigm';
sae.ae{2}.learningRate              = 1;
sae.ae{2}.inputZeroMaskedFraction   = 0.5;

opts.numepochs =   100;
opts.batchsize = 100;
sae = saetrain(sae, train_x, opts);
% visualize(sae.ae{1}.W{1}(:,2:end)')

% Use the SDAE to initialize a FFNN
nn = nnsetup([220 100 100 100 3]);
nn.activation_function              = 'sigm';
nn.learningRate                     = 1;

%add pretrained weights
nn.W{1} = sae.ae{1}.W{1};
nn.W{2} = sae.ae{2}.W{1};

% Train the FFNN
opts.numepochs =   100;
opts.batchsize = 100;
nn = nntrain(nn, train_x, train_y, opts,val_x,val_y);
[er, bad] = nntest(nn, test_x, test_y);