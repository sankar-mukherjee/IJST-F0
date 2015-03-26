function dbn = dbn(size)
load('../data/input.mat');load('../data/f0.mat');

train_x = input(160247:end,:);
% train_y = f0(1:end,:) / 399.9851;



test_x = input(1001:1200,:);
test_y  = f0(1001:1200,:) / 399.9851;


%%  ex1 train a 100 hidden unit RBM and visualize its weights
rng(0);
dbn.sizes = [256 256 256];
opts.numepochs =   100;
opts.batchsize = 100;
opts.momentum  =   0;
opts.alpha     =   1;
dbn = dbnsetup(dbn, train_x, opts);
dbn = dbntrain(dbn, train_x, opts);
% plot(dbn.rbm{2}.W(randi([1 50],1,1),:)); randi([1 50],1,1); %Visualize the RBM weights



dbn = dbn;