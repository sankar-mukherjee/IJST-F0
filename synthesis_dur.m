clear;close all;
load('data/duration.mat');
load('data/f0.mat');
load('data/sen_index.mat');
load('data/sp_duration.mat');load('data/word_index.mat');load('data/initial_sil_duration.mat');
load('data/input.mat');load('data/nn.mat');load('data/nn_dur.mat');

sentence_id = 21419;

[sen_i, ~] = find(sen_index(:,2)==sentence_id);
sen_dur = sen_index(sen_i:sen_i+1,1);
[w_i1,~] = find(word_index==sen_dur(1));
[w_i2,~] = find(word_index==sen_dur(2));
word_i = word_index(w_i1:w_i2-1);
sen_dur(2) = sen_dur(2)-1;

%%
% target_f0 = f0(sen_dur(1):sen_dur(2),:);          %original F0

% NN predicted f0
input_text=input(sen_dur(1):sen_dur(2),:);  
target_f0 = [];
for i=1:size(input_text,1)
     a1 = input_text(i,:);
     a1 = [1 a1];
     a1 = sigm(a1 * nn.W{1,1}');
     a1 = [1 a1];
     a1 = sigm(a1 * nn.W{1,2}');
     a1 = [1 a1];
     a1 = sigm(a1 * nn.W{1,3}');
     a1 = [1 a1];
     a1 = sigm(a1 * nn.W{1,4}');
     target_f0 = [target_f0;a1];
end
target_f0 = target_f0.*399.9851;                   %predicted F0
%%
% load('data/predicted_f0.mat');
% target_f0 = p;

%%

% NN predicted f0
input_text=input(sen_dur(1):sen_dur(2),:);  
target_dur = [];
for i=1:size(input_text,1)
     a1 = input_text(i,:);
     a1 = [1 a1];
     a1 = sigm(a1 * nn_dur.W{1,1}');
     a1 = [1 a1];
     a1 = sigm(a1 * nn_dur.W{1,2}');
     a1 = [1 a1];
     a1 = sigm(a1 * nn_dur.W{1,3}');
     a1 = [1 a1];
     a1 = sigm(a1 * nn_dur.W{1,4}');
     target_dur = [target_dur;a1];
end
target_dur = target_dur.*52;                   %predicted F0
target_f0_dur = ceil(target_dur);

%%
sp_dur = sp_duration(w_i1:w_i2-1);
% target_f0_dur = duration(sen_dur(1):sen_dur(2),:);

a1 = sen_dur(1):sen_dur(2);

F0 = [];
for i=1:size(target_f0_dur,1)
                   
        ph_dur = 1:target_f0_dur(i);
        ph = NaN(1,target_f0_dur(i));
        D = target_f0_dur(i);
        
        ph(floor(median(1:D/3)))=target_f0(i,1);
        ph(floor(median(D/3:2*D/3)))=target_f0(i,2);
        ph(floor(median(2*D/3:D)))=target_f0(i,3);
        
        F0=[F0; spline(ph_dur',ph,ph_dur')];
        
    if(a1(i)==word_i(1))
        F0=[F0; zeros(sp_dur(1),1)];
        word_i= circshift(word_i,-1);
        sp_dur= circshift(sp_dur,-1);
    end
    
end
sil_dur = initial_sil_duration(sen_i:sen_i+1);
F0 = [zeros(sil_dur(1),1);F0];
% F0(F0>250) = 250;
F0(F0<0) = 0;
% target_f0 = reshape(target_f0',1,[]);
F0(F0>450)=450;
F0=medfilt1(F0,7);
path = ['pitch/' num2str(sentence_id) '.txt'];
a=load(path , '-ascii');
% a = a(find(a,1,'first'):find(a,1,'last')); 
plot(F0,'r')
hold on;
plot(a)
% rmse(a,F0)

% path = ['C:\Users\Sank\Desktop\exp\Deep_learning\data\wav\' num2str(sentence_id) '.wav'];
path = ['/home/sank/Desktop/DeepLearning/audio/data/trimwav/' num2str(sentence_id) '.wav'];
[x,fs]=wavread(path);
[f0raw,ap,~]=exstraightsource(x,fs);
[n3sgram,~]=exstraightspec(x,F0,fs);
[sy,~] = exstraightsynth(F0,n3sgram,ap,fs);
sy = sy./max(sy);
wavwrite(sy,fs,'predicted.wav');
wavwrite(x,fs,'original.wav');