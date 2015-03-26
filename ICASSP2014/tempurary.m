load('../data/input.mat');
train_x = input(160247:end,:);
%%  ex1 train a 100 hidden unit RBM and visualize its weights
rng(0);
dbn.sizes = [256 256 256];
opts.numepochs =   200;
opts.batchsize = 100;
opts.momentum  =   0;
opts.alpha     =   1;
dbn = dbnsetup(dbn, train_x, opts);
dbn = dbntrain(dbn, train_x, opts);         save('dbn_3LAKH.mat','dbn')

clear;
load('../data/input_2152.mat');
train_x = input(160247:end,:);
%%  ex1 train a 100 hidden unit RBM and visualize its weights
rng(0);
dbn.sizes = [256 256 256];
opts.numepochs =   200;
opts.batchsize = 100;
opts.momentum  =   0;
opts.alpha     =   1;
dbn = dbnsetup(dbn, train_x, opts);
dbn = dbntrain(dbn, train_x, opts);        save('dbn_2152.mat','dbn')

clear;
load('../data/input_6016.mat');
train_x = input(160247:end,:);
%%  ex1 train a 100 hidden unit RBM and visualize its weights
rng(0);
dbn.sizes = [256 256 256];
opts.numepochs =   200;
opts.batchsize = 100;
opts.momentum  =   0;
opts.alpha     =   1;
dbn = dbnsetup(dbn, train_x, opts);
dbn = dbntrain(dbn, train_x, opts);          save('dbn_6016.mat','dbn')