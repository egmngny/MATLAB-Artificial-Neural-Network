clc;clear;close all;
%% Path of training data set and separate the validation
TrainDatasetPath='PATH';
Train=imageDatastore(TrainDatasetPath,'IncludeSubfolders',true,'LabelSource','foldername');
TestDatasetPath='PATH';
TestSeti=imageDatastore(TestDatasetPath,'IncludeSubfolders',true,'LabelSource','foldername');
TrainDatasetRatio=0.80; %Ratio of Validation is %20
[TrainDataset,Validation]= splitEachLabel(Train,TrainDatasetRatio,'randomize');
resim=readimage(Train,1); % Tuning the First Layer
size=size(resim);
%% Building the Network
layers=[
    imageInputLayer(size)
    maxPooling2dLayer(2,'Stride',2)
    convolution2dLayer(3,16,'Padding','same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2,'Stride',2)
    convolution2dLayer(3,32,'Padding','same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2,'Stride',2)
    convolution2dLayer(3,64,'Padding','same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2,'Stride',2)
    convolution2dLayer(3,128,'Padding','same')
    batchNormalizationLayer
    reluLayer
    fullyConnectedLayer(7) % 7 possibility for this project  
    softmaxLayer
    classificationLayer];
options= trainingOptions('sgdm',...
                        'LearnRateSchedule','piecewise', ...
                        'L2Regularization',0.001,... %Overfloating 
                        'Moment',0.9,... %old processes weight usage 
                        'MaxEpochs',20,... 
                        'MiniBatchSize',100,... 
                        'Shuffle','every-epoch',... 
                        'ValidationData',Validation,... %validation data 
                        'ValidationFrequency',30,... 
                        'Verbose',false,...
                        'Plots','training-progres',... %Creating Plots
                        'ExecutionEnvironment','gpu'); %GPU choosen for the process
%% Training Network
Network=trainNetwork(TrainDataset,layers,options); 
%% Testing Network
guess = classify(Network,TestSeti); 
answer = TestSeti.Labels;
accuracy = sum(guess==answer)/numel(answer)