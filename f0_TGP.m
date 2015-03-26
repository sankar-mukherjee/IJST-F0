clear;close all;

load('data/input.mat');load('data/nn.mat');load('data/f0.mat');load('data/sen_index.mat');

data_amount = 10000;                     sentence_id = 21419;
train_x = input(1:data_amount,:);
train_y = f0(1:data_amount,:);


sen_i = find(sen_index(:,2)==sentence_id);
test_initial = sen_index(sen_i,1);
test_final = sen_index(sen_i+1,1)-1;
test_x = input(test_initial:test_final,:);
test_y = f0(test_initial:test_final,:);

%traning data with NN
for i=1:size(train_x,1)
     train_nn = [1 train_x(i,:)];
     train_nn = sigm(train_nn * nn.W{1,1}');
     train_nn = [1 train_nn];
     train_nn = sigm(train_nn * nn.W{1,2}');
     train_nn = [1 train_nn];
     train_nn_x(i,:) = sigm(train_nn * nn.W{1,3}');
end
%test data with NN
for i=1:size(test_x,1)
     train_nn = [1 test_x(i,:)];
     train_nn = sigm(train_nn * nn.W{1,1}');
     train_nn = [1 train_nn];
     train_nn = sigm(train_nn * nn.W{1,2}');
     train_nn = [1 train_nn];
     test_nn_x(i,:) = sigm(train_nn * nn.W{1,3}');
end

Param.kparam1 = 0.2;
Param.kparam2 = 2*1e-6;
Param.kparam3 = Param.kparam2;
Param.lambda = 1e-3;
Param.knn = 100;

%% Twin Gaussian Processes

%NN transformed features
[InvIK, InvOK] = TGPTrain(train_nn_x, train_y, Param);
predicted_f0 = TGPTest(test_nn_x, train_nn_x, train_y, Param, InvIK, InvOK);

%raw features
% [InvIK, InvOK] = TGPTrain(train_x, train_y, Param);
% predicted_f0 = TGPTest(test_x, train_x, train_y, Param, InvIK, InvOK);

[Error, TGPErrorvec] = JointError(predicted_f0, test_y);
disp(['TGP: ' num2str(Error)]);

p = reshape(predicted_f0',1,[]);
o = reshape(test_y',1,[]);
F0 = p;
F0=medfilt1(p,7);
F0(F0<0) = 0;

plot(F0);hold on;plot(o,'r')