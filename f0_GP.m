clear;clc;close all;
load('data/input.mat');load('nn.mat');load('data/f0');load('data/sen_index');

data_amount = 3000;                     sentence_id = 16091;
train_x = input(1:data_amount,:);
train_y = f0(1:data_amount,:);
% train_y = train_y(:,1);

sen_i = find(sen_index(:,2)==sentence_id);
test_initial = sen_index(sen_i,1);
test_final = sen_index(sen_i+1,1)-1;
test_x = input(test_initial:test_final,:);
test_y = f0(test_initial:test_final,:);
% test_y = test_y(:,1);
%traning data with NN
for i=1:size(train_x,1)
     train_nn = train_x(i,:);
     train_nn = [1 train_nn];
     train_nn = sigm(train_nn * nn.W{1,1}');
     train_nn = [1 train_nn];
     train_nn = sigm(train_nn * nn.W{1,2}');
     train_nn = [1 train_nn];
     train_nn_x(i,:) = sigm(train_nn * nn.W{1,3}');
end
%test data with NN
for i=1:size(test_x,1)
     train_nn = test_x(i,:);
     train_nn = [1 train_nn];
     train_nn = sigm(train_nn * nn.W{1,1}');
     train_nn = [1 train_nn];
     train_nn = sigm(train_nn * nn.W{1,2}');
     train_nn = [1 train_nn];
     test_nn_x(i,:) = sigm(train_nn * nn.W{1,3}');
end
%%
covfunc = @covSEiso; 
nu = fix(size(train_nn_x,1)/2); iu = randperm(size(train_nn_x,1)); iu = iu(1:nu); u = train_nn_x(iu,:);
covfuncF = {@covFITC, {covfunc},u};
likfunc = @likGauss; 
sn = 0.1; 
hyp2.cov = [0 ; 0];    
hyp2.lik = log(sn);

% hyp2.mean =zeros(257,1);meanfunc = {@meanSum, {@meanLinear, @meanConst}};
predicted_f0=[];
for i=1:size(train_y,2)
    hyp2 = minimize(hyp2, @gp, -100, @infFITC, [], covfuncF, likfunc, train_nn_x, train_y(:,i));
    exp(hyp2.lik)
    nlml2 = gp(hyp2, @infFITC, [], covfuncF, likfunc, train_nn_x, train_y(:,i))
    [m s2] = gp(hyp2, @infFITC, [], covfuncF, likfunc, train_nn_x, train_y(:,i), test_nn_x);
    predicted_f0=[predicted_f0 m];
end

%%
p = reshape(predicted_f0',1,[]);
o = reshape(test_y',1,[]);
% o = o * 399.9851;
% F0 = p * 399.9851;
F0 = p;
% F0=medfilt1(p,7);
F0(F0<0) = 0;

plot(F0);hold on;plot(o,'r')

Error = rmse(F0,o);