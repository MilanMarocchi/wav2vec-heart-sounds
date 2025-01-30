% This m file implements the training of the CNN-based classifier based on the
% paper of "Ensemble of Feature-based and Deep learning-based Classifiers
% for Detection of Abnormal Heart Sounds" by Cristhian Potes et al.
% We added both ECG and PCG.

% By Yue Rong Jan. 2023.

clear all
close all

load('Springer_B_matrix.mat');
load('Springer_pi_vector.mat');
load('Springer_total_obs_distribution.mat');

%% Filters
N=60; sr = 1000; 
Wn = 45*2/sr; 
b1 = fir1(N,Wn,'low',hamming(N+1));
Wn = [45*2/sr, 80*2/sr];
b2 = fir1(N,Wn,hamming(N+1));
Wn = [80*2/sr, 200*2/sr];
b3 = fir1(N,Wn,hamming(N+1));
Wn = 200*2/sr;
b4 = fir1(N,Wn,'high',hamming(N+1));

% Database path
% db = ['a','b','c','d','e','f'];
db = ['a'];
%d_dir= '/Users/milanm/Dropbox/Uni/Year5/Thesis/Competitions/classification-of-heart-sound-recordings-the-physionet-computing-in-cardiology-challenge-2016-1.0-2.0/training-';
d_dir= '/home/mmaro/Dropbox/Uni/Year5/Thesis/Competitions/classification-of-heart-sound-recordings-the-physionet-computing-in-cardiology-challenge-2016-1.0-2.0/training-';

nfb = 4;            % number of frequency bands
ncycleNorm = 10;    % number of norminal heart beat cycles in each file
nsamp = 2500;       % number of samples in each heart beat cycle.
Maxfile = 1000;       % maximal number of files in each database in training.
trainingRatio= 1; % ratio of training and verification.
adj = 0.0;         % adjustment factor in segmentation.
    
% generate training data
nTrainEntry= 0;

for ndb = db
    data_dir= strcat(d_dir,ndb,'/',ndb);
    Label= readmatrix(strcat(data_dir, '.csv'));    % read the labels of data file
    Nfile = size(Label,1);  % number of files in each training database
    Nfile = min([Maxfile, Nfile]);
    file_o= randperm(Nfile);
    %file_o= randperm(Nfile);
    file_o = 1:Nfile;
    file_order =setdiff(file_o, [41,117,220,233],'stable'); % exclude files having no ecg data.
    Nfile = floor(length(file_order)*0.8);
    Nfile = 40; % for debugging

    for id = 1:Nfile
        if strcmp(ndb,'e');
            recordName= strcat(data_dir, num2str(file_order(id),'%05d'));
        else
            recordName= strcat(data_dir, num2str(file_order(id),'%04d'));
        end
        [PCG,Fs1] = audioread([recordName '.wav']);  % load PCG data
        
        fileName= strcat(recordName, '.dat');  % load ECG data
        fid = fopen(fileName);
        ecg = fread(fid,inf,'int16')/1000;
        
        if length(PCG)>60*Fs1
            PCG = PCG(1:60*Fs1);
            ecg = ecg(1:60*Fs1);
        end
        
        % resample to 1000 Hz
        Fs = 1000;
        PCG_resampled = resample(PCG,Fs,Fs1); % resample to Fs (1000 Hz)
        ecg_resampled = resample(ecg,Fs,Fs1);
        % filter the signal between 25 to 400 Hz
        PCG_resampled = butterworth_low_pass_filter(PCG_resampled,2,400,Fs, false);
        PCG_resampled = butterworth_high_pass_filter(PCG_resampled,2,25,Fs);
        % remove spikes
        PCG_resampled = schmidt_spike_removal(PCG_resampled,Fs);
        
        ecg_resampled = butterworth_low_pass_filter(ecg_resampled,2,60,Fs, false);
        ecg_resampled = butterworth_high_pass_filter(ecg_resampled,2,2,Fs);
        
%         figure(1)
%         plot(PCG_resampled);
               
        %% Running runSpringerSegmentationAlgorithm.m to obtain the assigned_states
        assigned_states = runSpringerSegmentationAlgorithm(PCG_resampled, Fs, Springer_B_matrix, Springer_pi_vector, Springer_total_obs_distribution, false);
        
        % get states
        idx_states = get_states(assigned_states);
        
        % filter signal in 4 different frequency bands,
        % [0,45],[45-80],[80-200],[200-400]
        clear PCG
        PCG(:,1) = filtfilt(b1,1,PCG_resampled);
        PCG(:,2) = filtfilt(b2,1,PCG_resampled);
        PCG(:,3) = filtfilt(b3,1,PCG_resampled);
        PCG(:,4) = filtfilt(b4,1,PCG_resampled);
        
        ncc = size(idx_states,1)-1;
        ncycle = min([ncc, ncycleNorm]);
        for row=1:ncycle
            X = nan(nsamp,nfb+1);
            for fb=1:nfb
                tmp = PCG(idx_states(row,1):idx_states(row+1,1),fb);                
                sampl_shift= min([floor(length(tmp)*adj), idx_states(row,1)-1]);   % number of samples shift in segmentation.
                N = nsamp-length(tmp);
                % adjust the segmentation.
                tmp = PCG(idx_states(row,1)-sampl_shift:idx_states(row+1,1)-sampl_shift,fb);
                % append zeros at the end of cardiac cycle
                X(:,fb) = [tmp; zeros(N,1)];
            end
            % Include ecg signal.
            tmp = ecg_resampled(idx_states(row,1):idx_states(row+1,1));
            N = nsamp-length(tmp);
            X(:,nfb+1) = [tmp; zeros(N,1)];
            
            nTrainEntry= nTrainEntry+1;
            Xall(:,:,1,nTrainEntry)= X;
            Yall(nTrainEntry,1) = Label(file_order(id),2);
        end
    end
end

Yall = sign(Yall+1);
size(Xall)

% load trainingdata
TrainSize = floor(size(Xall,4)*trainingRatio);
Xtrain = Xall(:,:,:,1:TrainSize);
Ytrain = categorical(Yall(1:TrainSize));
Xvalid = Xall(:,:,:,TrainSize+1:end);
Yvalid = categorical(Yall(TrainSize+1:end));

ValidSize = size(Xvalid,4);

%% Convert trianing data into cell arrays
imgCells1 = mat2cell(Xtrain(:,1,:,:),nsamp,1,1,ones(TrainSize,1));
imgCells2 = mat2cell(Xtrain(:,2,:,:),nsamp,1,1,ones(TrainSize,1));
imgCells3 = mat2cell(Xtrain(:,3,:,:),nsamp,1,1,ones(TrainSize,1));
imgCells4 = mat2cell(Xtrain(:,4,:,:),nsamp,1,1,ones(TrainSize,1));
imgCells5 = mat2cell(Xtrain(:,5,:,:),nsamp,1,1,ones(TrainSize,1));

imgCells1 = reshape(imgCells1,[TrainSize 1 1]);
imgCells2 = reshape(imgCells2,[TrainSize 1 1]);
imgCells3 = reshape(imgCells3,[TrainSize 1 1]);
imgCells4 = reshape(imgCells4,[TrainSize 1 1]);
imgCells5 = reshape(imgCells5,[TrainSize 1 1]);

labelCells = arrayfun(@(x)x,Ytrain,'UniformOutput',false);
combinedCells = [imgCells1 imgCells2 imgCells3 imgCells4 imgCells5 labelCells];
%% Save the converted data so that it can be loaded by filedatastore
save('trainData_ecg1.mat','combinedCells','-v7.3');

%load trainData400Epochs1024minBatchB.mat;
filedatastore = fileDatastore('trainData_ecg1.mat','ReadFcn',@load);
trainingDatastore = transform(filedatastore,@rearrangeData);

% %% Convert validation data into cell arrays
% ValidCells1 = mat2cell(Xvalid(:,1,:,:),nsamp,1,1,ones(ValidSize,1));
% ValidCells2 = mat2cell(Xvalid(:,2,:,:),nsamp,1,1,ones(ValidSize,1));
% ValidCells3 = mat2cell(Xvalid(:,3,:,:),nsamp,1,1,ones(ValidSize,1));
% ValidCells4 = mat2cell(Xvalid(:,4,:,:),nsamp,1,1,ones(ValidSize,1));
% ValidCells1 = reshape(ValidCells1,[ValidSize 1 1]);
% ValidCells2 = reshape(ValidCells2,[ValidSize 1 1]);
% ValidCells3 = reshape(ValidCells3,[ValidSize 1 1]);
% ValidCells4 = reshape(ValidCells4,[ValidSize 1 1]);
% VlabelCells = arrayfun(@(x)x,Yvalid,'UniformOutput',false);
% combinedvalidCells = [ValidCells1 ValidCells2 ValidCells3 ValidCells4 VlabelCells];
% %% Save the converted data so that it can be loaded by filedatastore
% save('ValidData1.mat','combinedvalidCells','-v7.3');
% 
% 
% Vdatastore = fileDatastore('validData.mat','ReadFcn',@load);
% ValidDatastore = transform(Vdatastore,@VrearrangeData);


%% Form DNN layers
% Frequency band 1
Layer1 = [imageInputLayer([nsamp 1],'Name','input1')
    convolution2dLayer([5 1],8,'stride',1,'Name','conv11')
    reluLayer('Name','relu11')
    maxPooling2dLayer([2 1],'Name','pooling11','stride',[2 1])
    convolution2dLayer([5 1],4,'stride',1,'Name','conv12')
    reluLayer('Name','relu12')
    maxPooling2dLayer([2 1],'Name','pooling12','stride',[2 1]);
    resize3dLayer('OutputSize',[2488 1 1],'Name','drop1')];

% Frequency band 2
Layer2 = [imageInputLayer([nsamp 1],'Name','input2')
    convolution2dLayer([5 1],8,'stride',1,'Name','conv21')
    reluLayer('Name','relu21')
    maxPooling2dLayer([2 1],'Name','pooling21','stride',[2 1])
    convolution2dLayer([5 1],4,'stride',1,'Name','conv22')
    reluLayer('Name','relu22')
    maxPooling2dLayer([2 1],'Name','pooling22','stride',[2 1]);
    resize3dLayer('OutputSize',[2488 1 1],'Name','drop2')];

% Frequency band 3
Layer3 = [imageInputLayer([nsamp 1],'Name','input3')
    convolution2dLayer([5 1],8,'stride',1,'Name','conv31')
    reluLayer('Name','relu31')
    maxPooling2dLayer([2 1],'Name','pooling31','stride',[2 1])
    convolution2dLayer([5 1],4,'stride',1,'Name','conv32')
    reluLayer('Name','relu32')
    maxPooling2dLayer([2 1],'Name','pooling32','stride',[2 1]);
    resize3dLayer('OutputSize',[2488 1 1],'Name','drop3')];

% Frequency band 4
Layer4 = [imageInputLayer([nsamp 1],'Name','input4')
    convolution2dLayer([5 1],8,'stride',1,'Name','conv41')
    reluLayer('Name','relu41')
    maxPooling2dLayer([2 1],'Name','pooling41','stride',[2 1])
    convolution2dLayer([5 1],4,'stride',1,'Name','conv42')
    reluLayer('Name','relu42')
    maxPooling2dLayer([2 1],'Name','pooling42','stride',[2 1]);
    resize3dLayer('OutputSize',[2488 1 1],'Name','drop4')];

% ECG layer
Layer5 = [imageInputLayer([nsamp 1],'Name','input5')
    convolution2dLayer([5 1],8,'stride',1,'Name','conv51')
    reluLayer('Name','relu51')
    maxPooling2dLayer([2 1],'Name','pooling51','stride',[2 1])
    convolution2dLayer([5 1],4,'stride',1,'Name','conv52')
    reluLayer('Name','relu52')
    maxPooling2dLayer([2 1],'Name','pooling52','stride',[2 1]);
    resize3dLayer('OutputSize',[2488 1 1],'Name','drop5')];

concat = concatenationLayer(1,5,'Name','concat');
lgraph = layerGraph();
lgraph = addLayers(lgraph, Layer1);
lgraph = addLayers(lgraph, Layer2);
lgraph = addLayers(lgraph, Layer3);
lgraph = addLayers(lgraph, Layer4);
lgraph = addLayers(lgraph, Layer5);
lgraph = addLayers(lgraph, concat);

% connecting the four frequency bands
lgraph = connectLayers(lgraph,'drop1','concat/in1');
lgraph = connectLayers(lgraph,'drop2','concat/in2');
lgraph = connectLayers(lgraph,'drop3','concat/in3');
lgraph = connectLayers(lgraph,'drop4','concat/in4');
lgraph = connectLayers(lgraph,'drop5','concat/in5');


% Two fully connected layers
LayerLast= [fullyConnectedLayer(20,'name','fc1')
            reluLayer('Name','relula')
            fullyConnectedLayer(2,'name','fc2')
            softmaxLayer('Name','softmx')
            classificationLayer('Name','cl')];

lgraph = addLayers(lgraph, LayerLast);
lgraph = connectLayers(lgraph,'concat','fc1');

% figure
% plot(lgraph)
 
MaxEpochs = 400;
% MaxEpochs = 40;
MiniBatchSize = 1024;
analyzeNetwork(lgraph)

%% Define trainig options 
Options = trainingOptions('adam',...%'ValidationData',ValidDatastore,...
    'InitialLearnRate',0.0007,...
    'ExecutionEnvironment','auto', ...
    'GradientThreshold',Inf, ...
    'LearnRateSchedule','none', ...
    'LearnRateDropFactor',0.1,...
    'Shuffle','every-epoch',...
    'MaxEpochs',MaxEpochs, ...
    'MiniBatchSize',MiniBatchSize, ...
    'Verbose',0,...
    'Plots','training-progress');
%% Train DNN
Net = trainNetwork(trainingDatastore,lgraph,Options);

Ytest = classify(Net,trainingDatastore);
figure
confusionchart(Ytrain,Ytest)
accuracy = mean(Ytrain==Ytest);


% Get the weights of the first CNN layer
W11=squeeze(Net.Layers(2,1).Weights);
B11=squeeze(Net.Layers(2,1).Bias);
W21=squeeze(Net.Layers(10,1).Weights);
B21=squeeze(Net.Layers(10,1).Bias);
W31=squeeze(Net.Layers(18,1).Weights);
B31=squeeze(Net.Layers(18,1).Bias);
W41=squeeze(Net.Layers(26,1).Weights);
B41=squeeze(Net.Layers(26,1).Bias);
W51=squeeze(Net.Layers(34,1).Weights);
B51=squeeze(Net.Layers(34,1).Bias);

% Get the weights of the second CNN layer
W12=squeeze(Net.Layers(5,1).Weights);
B12=squeeze(Net.Layers(5,1).Bias);
W22=squeeze(Net.Layers(13,1).Weights);
B22=squeeze(Net.Layers(13,1).Bias);
W32=squeeze(Net.Layers(21,1).Weights);
B32=squeeze(Net.Layers(21,1).Bias);
W42=squeeze(Net.Layers(29,1).Weights);
B42=squeeze(Net.Layers(29,1).Bias);
W52=squeeze(Net.Layers(37,1).Weights);
B52=squeeze(Net.Layers(37,1).Bias);


parms.H1(1,:,:)= W11.';
parms.H1(2,:,:)= W21.';
parms.H1(3,:,:)= W31.';
parms.H1(4,:,:)= W41.';
parms.H1(5,:,:)= W51.';
parms.b1 = [B11, B21, B31, B41, B51].';

parms.H2(1,:,:,:)=reshape(W12,[4,8,5]);
parms.H2(2,:,:,:)=reshape(W22,[4,8,5]);
parms.H2(3,:,:,:)=reshape(W32,[4,8,5]);
parms.H2(4,:,:,:)=reshape(W42,[4,8,5]);
parms.H2(5,:,:,:)=reshape(W52,[4,8,5]);
parms.b2 = [B12, B22, B32, B42, B52].';

% Weight of the first fully connected layer.
parms.W1= (squeeze(Net.Layers(42,1).Weights)).';
parms.bias1= (squeeze(Net.Layers(42,1).Bias)).';

% Weight of the second fully connected layer.
parms.W2= (squeeze(Net.Layers(44,1).Weights)).';
parms.bias2= (squeeze(Net.Layers(44,1).Bias)).';

save('parms_cnn_Rong_A_ecg1.mat','parms','Net','file_order');


%% function to be used to transform the filedatastore 
%to ensure the read(datastore) returns M-by-3 cell array ie., (numInputs+1) columns
function out = rearrangeData(ds)
out = ds.combinedCells;
end

function out = VrearrangeData(ds)
out = ds.combinedvalidCells;
end


function idx_states = get_states(assigned_states)
    indx = find(abs(diff(assigned_states))>0); % find the locations with changed states

    if assigned_states(1)>0   % for some recordings, there are state zeros at the beginning of assigned_states
        switch assigned_states(1)
            case 4
                K=1;
            case 3
                K=2;
            case 2
                K=3;
            case 1
                K=4;
        end
    else
        switch assigned_states(indx(1)+1)
            case 4
                K=1;
            case 3
                K=2;
            case 2
                K=3;
            case 1
                K=0;
        end
        K=K+1;
    end

    indx2                = indx(K:end);
    rem                  = mod(length(indx2),4);
    indx2(end-rem+1:end) = [];
    length(indx2);
    idx_states           = reshape(indx2,4,length(indx2)/4)';
end