%% Classify PCG signals directly using Matlab classify function
%% Combined ecg and pcg signal classification
clear all

load('Springer_B_matrix.mat');
load('Springer_pi_vector.mat');
load('Springer_total_obs_distribution.mat');
load parms_cnn_Rong_A_ecg.mat;
parms.maxpooling = 2;

N=60; sr = 1000; 
Wn = 45*2/sr; 
b1 = fir1(N,Wn,'low',hamming(N+1));
Wn = [45*2/sr, 80*2/sr];
b2 = fir1(N,Wn,hamming(N+1));
Wn = [80*2/sr, 200*2/sr];
b3 = fir1(N,Wn,hamming(N+1));
Wn = 200*2/sr;
b4 = fir1(N,Wn,'high',hamming(N+1));

adj = 0.00;         % adjustment factor in segmentation.

%% Load data and resample data
springer_options   = default_Springer_HSMM_options;
springer_options.use_mex = 1;
%[PCG, Fs1, nbits1] = wavread([recordName '.wav']);  % load data

d_dir= '/Users/milanm/Dropbox/Uni/Year5/Thesis/Competitions/classification-of-heart-sound-recordings-the-physionet-computing-in-cardiology-challenge-2016-1.0-2.0/training-';
ndb = ['a'];
data_dir= strcat(d_dir,ndb,'/',ndb);
Label= readmatrix(strcat(data_dir, '.csv'));
ncycleNorm = 10;    % number of norminal heart beat cycles in each file
Maxfile = 1000;       % maximal number of files in each database in training.
% Nfile = min([Maxfile,size(Label,1)]);  % number of files in each training database
Nfile = 405;
%file_order= randperm(Nfile);
% file_order = 1:1:Nfile;
Errcl=0;

% filedatastore = fileDatastore('trainData_A.mat','ReadFcn',@load);
% testDatastore = transform(filedatastore,@rearrangeData);
% Y = classify(Net,testDatastore);
Y2 = categorical([]);


% last 20% of each database for testing.
num_test = 0;
for id = floor(Nfile*0.8)+1:Nfile
% for id = 42:101
    if strcmp(ndb,'e');
        recordName= strcat(data_dir, num2str(file_order(id),'%05d'));
    else
        recordName= strcat(data_dir, num2str(file_order(id),'%04d'));
    end   
    [PCG,Fs1] = audioread([recordName '.wav']);  % load data
    fileName= strcat(recordName, '.dat');  % load ECG data
    fid = fopen(fileName);
    ecg = fread(fid,inf,'int16')/1000;
    
    if length(PCG)>60*Fs1
        PCG = PCG(1:60*Fs1);
        ecg = ecg(1:60*Fs1);
    end
    
    % resample to 1000 Hz
    PCG_resampled = resample(PCG,springer_options.audio_Fs,Fs1); % resample to springer_options.audio_Fs (1000 Hz)
    ecg_resampled = resample(ecg,springer_options.audio_Fs,Fs1);

    % filter the signal between 25 to 400 Hz
    PCG_resampled = butterworth_low_pass_filter(PCG_resampled,2,400,springer_options.audio_Fs, false);
    PCG_resampled = butterworth_high_pass_filter(PCG_resampled,2,25,springer_options.audio_Fs);
    % remove spikes
    PCG_resampled = schmidt_spike_removal(PCG_resampled,springer_options.audio_Fs);
    
    ecg_resampled = butterworth_low_pass_filter(ecg_resampled,2,60,springer_options.audio_Fs, false);
    ecg_resampled = butterworth_high_pass_filter(ecg_resampled,2,2,springer_options.audio_Fs);

    %% Running runSpringerSegmentationAlgorithm.m to obtain the assigned_states
    assigned_states = runSpringerSegmentationAlgorithm(PCG_resampled,...
        springer_options.audio_Fs,...
        Springer_B_matrix, Springer_pi_vector,...
        Springer_total_obs_distribution, false);
    
    % get states
    idx_states = get_states(assigned_states);
    
    
    % filter signal in 4 different frequency bands,
    % [0,45],[45-80],[80-200],[200-400]
    clear PCG
    PCG(:,1) = filtfilt(b1,1,PCG_resampled);
    PCG(:,2) = filtfilt(b2,1,PCG_resampled);
    PCG(:,3) = filtfilt(b3,1,PCG_resampled);
    PCG(:,4) = filtfilt(b4,1,PCG_resampled);
    
    nfb = 4;
    nsamp = 2500;
    ncc = size(idx_states,1)-1;
    ncc = min([ncc, ncycleNorm]);
    X = nan(ncc,nsamp,nfb+1);
    for row=1:ncc
        for fb=1:nfb
            tmp = PCG(idx_states(row,1):idx_states(row+1,1),fb);
            sampl_shift= min([floor(length(tmp)*adj), idx_states(row,1)-1]);    % number of samples shift in segmentation.
            N = nsamp-length(tmp);
            % adjust the segmentation.
            tmp = PCG(idx_states(row,1)-sampl_shift:idx_states(row+1,1)-sampl_shift,fb);
            % append zeros at the end of cardiac cycle
            X(row,:,fb) = [tmp; zeros(N,1)];
        end
        tmp = ecg_resampled(idx_states(row,1):idx_states(row+1,1));
        N = nsamp-length(tmp);
        X(row,:,nfb+1) = [tmp; zeros(N,1)];
    end
    
    % run the classifier
    res_num = zeros(size(X,1),1);
    res = categorical([]);
    for sample = 1:size(X,1)
        s = squeeze(X(sample,:,:));
        res(sample) = classify(Net,s(:,1),s(:,2),s(:,3),s(:,4),s(:,5));
        if res(sample)==categorical([0])
            res_num(sample) = 0;
        else
            res_num(sample) = 1;
        end
    end
    Y2 = [Y2; res.'];
    prb_cnn = mean(res_num);
    
    if prb_cnn > 0.4
        classifyResult = 1;
    else
        classifyResult = -1;
    end
    
    % [id, classifyResult]
    if classifyResult ~= Label(file_order(id),2)
        Errcl = Errcl+1;
    end
    num_test= num_test+1;
    classAll(num_test)= classifyResult;
    LabelAll(num_test)= Label(file_order(id),2);
end

figure
cm= confusionchart(LabelAll,classAll);
cm2= cm.NormalizedValues;
spci =cm2(1,1)/sum(cm2(1,:))
sens = cm2(2,2)/sum(cm2(2,:))
accu = sum(diag(cm2))/sum(cm2,'all')

1-Errcl/(Nfile-floor(Nfile*0.8))
%1-Errcl/60
%%

%% function to be used to transform the filedatastore 
%to ensure the read(datastore) returns M-by-3 cell array ie., (numInputs+1) columns
function out = rearrangeData(ds)
out = ds.combinedCells;
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
    idx_states           = reshape(indx2,4,length(indx2)/4)';
end

%% 

