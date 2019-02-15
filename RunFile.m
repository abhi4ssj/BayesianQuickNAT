clc
clear
close all
%% Directory
HomeDir = '/home/deeplearning/Abhijit/nas_drive/Abhijit/UncertaintyExperiments/DeploymentCodeForGitHub/SampleData/'; % Enter the Directory Path where the target MRI data exists
DataID = 'orig_test.mgz';
addpath('Models');

%% Read the Data
DataVol = MRIread([HomeDir,DataID]);
NumFrames = 52;
NumMCsamples = 10;

%% Run the Prediction Code
out_dir = [HomeDir,DataID(1:end-4)];
mkdir(out_dir);
[Predictions, cv_s, iou_s, SegTime] = SegmentVol_EstimateUncertainty(DataVol,NumFrames, NumMCsamples, out_dir);
save([out_dir, '/Uncertainties.mat'], 'cv_s', 'iou_s');
disp(['----Processing Over. Segmentation Time is ',num2str(SegTime), ' seconds.']);



