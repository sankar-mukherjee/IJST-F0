clear;clc;
load('data/input.mat');
load('data/duration.mat');

% train_x = double(train_x) / 399.9851;
% train_x = input(1:end-1,:);
% train_y = output(1:end-1,:) / 399.9851;
% test_x = input(1001:1200,:);
% test_y  = output(1001:1200,:) / 399.9851;

train_x = input(1:350000,:);
train_y = duration(1:350000,:) / 52;

train_xx = input(350001:440000,:);
train_yy = duration(350001:440000,:) / 52;

val_x = input(440001:460200,:);
val_y = duration(440001:460200,:)/ 52;
test_x = input(1001:1200,:);
test_y  = duration(1001:1200,:) / 52;

train_xxx = input(1:400000,:);
train_yyy = duration(1:400000,:) / 52;
%%  ex1 train a 100 hidden unit RBM and visualize its weights
rng(0);
% dbn.sizes = [256 256 256];
dbn.sizes = [256 256 256];
opts.numepochs =   200;
opts.batchsize = 100;
opts.momentum  =   0;
opts.alpha     =   1;
dbn = dbnsetup(dbn, train_x, opts);
dbn = dbntrain(dbn, train_x, opts);
% plot(dbn.rbm{2}.W(randi([1 50],1,1),:)); randi([1 50],1,1); %Visualize the RBM weights

%unfold dbn to nn
nn_dur = dbnunfoldtonn(dbn, 1);
nn_dur.activation_function = 'sigm';
nn_dur.output = 'linear';
%train nn
opts.numepochs =  200;
opts.batchsize = 100;
nn_dur = nntrain(nn_dur, train_xx, train_yy, opts,val_x,val_y);
[er, bad] = nntest(nn_dur, test_x, test_y);