function [IK OK P] = TGP(train_x,train_y,nn)
%% TGP


for i=1:size(train_x,1)
    train_nn = [1 train_x(i,:)];
    train_nn = sigm(train_nn * nn.W{1,1}');
    train_nn = [1 train_nn];
    train_nn = sigm(train_nn * nn.W{1,2}');
    train_nn = [1 train_nn];
    train_nn_x(i,:) = sigm(train_nn * nn.W{1,3}');
end
%% Twin Gaussian Processes
Param.kparam1 = 0.2;
Param.kparam2 = 2*1e-6;
Param.kparam3 = Param.kparam2;
Param.lambda = 1e-3;
Param.knn = 100;
% [InvIK, InvOK] = TGPTrain(train_nn_x, train_y, Param);
[InvIK, InvOK] = TGPTrain(train_x, train_y, Param); %RAW features

 IK=InvIK;
 OK=InvOK;
 P=Param;