clear;clc;
load('data/input.mat');
load('data/f0.mat');

% train_x = double(train_x) / 399.9851;
% train_x = input(1:end-1,:);
% train_y = output(1:end-1,:) / 399.9851;
% test_x = input(1001:1200,:);
% test_y  = output(1001:1200,:) / 399.9851;

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
%%  ex1 train a 100 hidden unit RBM and visualize its weights
rng(0);
% dbn.sizes = [256 256 256];
dbn.sizes = [512 512 512];
opts.numepochs =   100;
opts.batchsize = 100;
opts.momentum  =   0;
opts.alpha     =   1;
dbn = dbnsetup(dbn, train_x, opts);
dbn = dbntrain(dbn, train_x, opts);
% plot(dbn.rbm{2}.W(randi([1 50],1,1),:)); randi([1 50],1,1); %Visualize the RBM weights

%unfold dbn to nn
nn = dbnunfoldtonn(dbn, 3);
nn.activation_function = 'sigm';
nn.output = 'linear';
%train nn
opts.numepochs = 100;
opts.batchsize = 100;
nn = nntrain(nn, train_xx, train_yy, opts,val_x,val_y);
[er, bad] = nntest(nn, test_x, test_y);