%% NN tranning
run('dbn_text');
% run('SAE_f0');
save('data/nn.mat','nn');
%% GP tranning
run('f0_TGP');
p = reshape(F0',size(predicted_f0,2),size(predicted_f0,1))';
save('data/predicted_f0.mat','p');
%% sound file generation
run('synthesis');
