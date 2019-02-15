function [Predictions, cv_s, iou_s, SegTime] = SegmentVol_EstimateUncertainty(DataVol,NumFrames, NumMCsamples, out_dir)

% The segmentation is done 2D slice-wise. Speed is dependent on number of frames you can push 'NumFrames'. This is dependent on GPU size. Please 
% try different values to optimize this for your GPU. In Titan Xp 12GB, 70 slices were pushed giving segmentation time of 20secs.

warning('off', 'all');
% Load the Trained Models
disp('Loading the Trained Models and Data...')
load('CoronalModel.mat'); % CoronalNet
fnet = dagnn.DagNN.loadobj(net);
load('AxialModel.mat'); % AxialNet
fnet2 = dagnn.DagNN.loadobj(net);
load('SagittalModel.mat') % SagittalNet
fnet3 = dagnn.DagNN.loadobj(net);
NumCls = 34;
% Prepare the data for deployment in QuickNAT
Vol = DataVol.vol;
sz = size(Vol);
DataVol_Ax = permute(Vol, [3,2,1]);
DataVol_Sag = permute(Vol, [3,1,2]);
DataSelect = single(reshape(mat2gray(Vol(:,:,:)),[sz(1), sz(2), 1, sz(3)]));
DataSelect_Ax = single(reshape(mat2gray(DataVol_Ax(:,:,:)),[sz(1), sz(2), 1, sz(3)]));
DataSelect_Sag = single(reshape(mat2gray(DataVol_Sag(:,:,:)),[sz(1), sz(2), 1, sz(3)]));

Final_Seg = zeros(256, 256, NumCls, 256, NumMCsamples, 'single');

% ---- start of segmentation
disp('Starting the process...')
tic
% 70 slices in one pass restricted by GPU space
for pp = 1:NumMCsamples
    Predictions_MC_Cor = [];
    Predictions_MC_Ax = [];
    Predictions_MC_Sag = [];
    for j= 1:NumFrames:256
        if(j>(256-NumFrames+1))
            k=256;
        else
            k=j+NumFrames-1;
        end
        fnet3.move('cpu'); % GPU-CPU handshaking for space
        fnet.mode = 'test'; fnet.move('gpu');
        %     fnet.conserveMemory = 0;
        fnet.eval({'input', gpuArray(DataSelect(:,:,:,j:k))});
        reconstruction = fnet.vars(fnet.getVarIndex('prob')).value;
        reconstruction = gather(reconstruction);
        Predictions1 = squeeze(reconstruction);
        Predictions_MC_Cor = cat(4,Predictions_MC_Cor, Predictions1);

        fnet.move('cpu');
        fnet2.mode = 'test'; fnet2.move('gpu');
        %     fnet2.conserveMemory = 0;
        fnet2.eval({'input', gpuArray(DataSelect_Ax(:,:,:,j:k))});
        reconstruction = fnet2.vars(fnet2.getVarIndex('prob')).value;
        reconstruction = gather(reconstruction);
        Predictions1 = squeeze(reconstruction);
        Predictions_MC_Ax = cat(4,Predictions_MC_Ax, Predictions1);

        fnet2.move('cpu');
        fnet3.mode = 'test'; fnet3.move('gpu');
        %     fnet3.conserveMemory = 0;
        fnet3.eval({'input', gpuArray(DataSelect_Sag(:,:,:,j:k))});
        reconstruction = fnet3.vars(fnet3.getVarIndex('prob')).value;
        reconstruction = gather(reconstruction);
        Predictions1 = squeeze(reconstruction);
        Predictions_MC_Sag = cat(4,Predictions_MC_Sag, Predictions1);

    end
    Predictions_MC_Ax = permute(Predictions_MC_Ax, [4,2,3,1]);
    Predictions_MC_Sag = permute(Predictions_MC_Sag, [2,4,3,1]);
    Predictions_MC_Sag = ReMapSagProbMap(Predictions_MC_Sag, NumCls);
    PredictionsFinal = (0.4*Predictions_MC_Ax + 0.4*Predictions_MC_Cor + 0.2*Predictions_MC_Sag);
    Final_Seg(:,:,:,:,pp) = PredictionsFinal; 
    % Arg Max Stage for dense labelling
    [~, Predictions] = max(PredictionsFinal,[],3);
    Predictions = squeeze(Predictions);
    Pred = DataVol;
    Pred.vol = Predictions-1;
    err = MRIwrite(Pred,[out_dir,'/MonteCarloSample_',num2str(pp),'.mgz']);   
    disp(['Number of MC passes completed ',num2str(pp),' out of ',num2str(NumMCsamples)]);
end

[~, Final_Seg] = max(mean(Final_Seg, 5),[],3);
Final_Seg = Final_Seg - 1;
Final = DataVol; Final.vol = Final_Seg;
err = MRIwrite(Final,[out_dir,'/Final_Segmentation.mgz']);

disp('Estimating Structure-wise Uncertainty...');

iou_s = zeros(NumCls-1,1);
VolumeEstimate = zeros(NumCls-1, NumMCsamples);

% Estimate the IoU Uncertainty
for j = 1:NumMCsamples
    S{j} = MRIread([out_dir,'/MonteCarloSample_',num2str(j),'.mgz']);
    S{j}.vol = S{j}.vol;
end


for k = 1:NumCls-1
    % IoU
    Inter = S{1}.vol==k;
    Union = S{1}.vol==k;
    for j = 2:NumMCsamples
        Inter = Inter.*(S{j}.vol==k);
        Union = (Union + (S{j}.vol==k))>0;
    end
    iou_s(k, 1) = sum(Inter(:))/sum(double(Union(:))+eps);
end
disp('IoU Uncertainty done');
for j = 1:NumMCsamples
    for k = 1:NumCls-1
        temp = S{j}.vol==k;
        VolumeEstimate(k,j) = sum(temp(:));
    end
end
cv_s = squeeze((std(VolumeEstimate,[],2)./mean(VolumeEstimate,2))*100);
disp('Volume Uncertainty done');


SegTime = toc;
%---- end of Segmentation