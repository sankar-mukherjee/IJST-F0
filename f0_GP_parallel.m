clear;clc;close all;
run('./gpml-matlab-v3.2-2013-01-15/startup.m');
matlabpool ('open',6);
load('data/input.mat');load('nn.mat');load('data/f0');
data_amount = 5000;
train_x = input(1:data_amount,:);
train_y = f0(1:data_amount,:);
% train_y = train_y(:,1);
% 
% train_x = gpuArray(input(1:data_amount,:));
% train_y = gpuArray(f0(1:data_amount,:));

test_initial = 1;
test_final = 99;
test_x = input(test_initial:test_final,:);
test_y = f0(test_initial:test_final,:);
% test_y = test_y(:,1);

% test_x = gpuArray(input(test_initial:test_final,:));
% test_y = gpuArray(f0(test_initial:test_final,:));

%traning data with NN
train_nn_x = zeros(size(train_x,1),256);
parfor i=1:size(train_x,1)
     train_nn = [1 train_x(i,:)];
     train_nn = sigm(train_nn * nn.W{1,1}');
     train_nn = [1 train_nn];
     train_nn = sigm(train_nn * nn.W{1,2}');
     train_nn = [1 train_nn];
     train_nn_x(i,:) = sigm(train_nn * nn.W{1,3}');
end
%test data with NN
parfor i=1:size(test_x,1)
     train_nn = [1 test_x(i,:)];
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


% hyp2.mean =zeros(257,1);meanfunc = {@meanSum, {@meanLinear, @meanConst}};
% predicted_f0=[];
predicted_f0 = zeros(size(test_x,1),size(train_y,2));
parfor i=1:3
    hyp2_i.cov = [0 ; 0];
    hyp2_i.lik = log(sn);
    hyp2_i = minimize(hyp2_i, @gp, -100, @infFITC, [], covfuncF, likfunc, train_nn_x, train_y(:,i));
%     exp(hyp2.lik)
%     nlml2 = gp(hyp2, @infFITC, [], covfuncF, likfunc, train_nn_x, train_y(:,i))
    [m_i s2] = gp(hyp2_i, @infFITC, [], covfuncF, likfunc, train_nn_x, train_y(:,i), test_nn_x);
    predicted_f0(i)=m_i;
end

%%
p = reshape(predicted_f0',1,[]);
o = reshape(test_y',1,[]);
plot(p);hold on;plot(o,'r')
rmse(p,o)
matlabpool close;  