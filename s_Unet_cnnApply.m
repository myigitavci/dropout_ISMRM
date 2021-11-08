% s_Unet_cnnApply.m
%
%
%
% QT,MYA 2021

clear, clc, close all
dpRoot = rootpath();

%%
subjects = dir("C:\Users\yigitavci\Desktop\codes\mwu*");
subjects={subjects.name}

for ii = 1 :1
    sj = subjects{ii}; % subject
    disp(['***** ' sj ' *****']);

    dpSub = fullfile(dpRoot, sj); % subject path
    dpBind = fullfile(dpSub, 'bind');
    dpPred = fullfile(dpSub, 'pred');
    dpPred = fullfile(dpPred, '1-subjects-cnn-drop-02'); %%%%%%%%%%%%%%%%%%%%%%
    mkdir(dpPred);
    
    fpBmask = fullfile(dpSub, 'data', [sj '_diff_mask.nii.gz']);        
    fpBind = fullfile(dpBind, [sj '_bind_b64.mat']);
    
    fpMetrics = fullfile(dpSub, 'metrics-norm', [sj '_metrics.nii.gz']);
    metrics = niftiread(fpMetrics);
    
    bmask = niftiread(fpBmask) > 0.5;
    load(fpBind);

    preds = listfile(fullfile(dpPred, '*unet_1_subj_drop_02_img_block_pred*.mat'), 1);
    
    img_allavgs = [];
    
    for jj = 1 : 100
        disp(jj)
        pred = preds{jj};
        [~, fnPred] = fileparts(pred);
        name = fnPred(1 : strfind(fnPred, 'img_block') - 2);
                
        fpImg = fullfile(dpPred, pred);
        load(fpImg);
        
        count_map = zeros(size(metrics));
        imgnorm_unblock = zeros(size(metrics));
        
        for mm = 1 : size(bind, 1)            
            ind  = bind(mm, :);
            ind_new = zeros(size(ind));
            
            for tt = 1 : 3
                vmin = min(min(bind(:, (tt - 1)*2 + 1 : tt*2)));
                vmax = max(max(bind(:, (tt - 1)*2 + 1 : tt*2)));
                vall = unique(bind(:, (tt - 1)*2 + 1 : tt*2));
                
                ind2 = ind(:, (tt - 1)*2 + 1 : tt*2);
                ind2_new = zeros(size(ind2));
                
                if ind2(1) == vmin
                    ind2_new(1) = ind2(1);
                else
                    idx1 = ind2(1);
                    tmp = find(vall > idx1);
                    idx2 = vall(tmp(1));
                    idx = (idx1 + idx2) / 2;
                    ind2_new(1) = ceil(idx + 0.1);
                end
                if ind2(2) == vmax
                    ind2_new(2) = ind2(2);
                else
                    idx1 = ind2(2);
                    tmp = find(vall < idx1);
                    idx2 = vall(tmp(end));
                    idx = (idx1 + idx2) / 2;
                    ind2_new(2) = floor(idx + 0.1);
                end
                ind_new(:, (tt - 1)*2 + 1 : tt*2) = ind2_new;
            end
                        
            img_tmp = zeros(size(metrics));
            img_tmp(ind(1):ind(2), ind(3):ind(4), ind(5):ind(6), :) = squeeze(img_block_pred(mm, :, :, :, :));
            
            mask_tmp = zeros(size(metrics));
            mask_tmp(ind_new(1):ind_new(2), ind_new(3):ind_new(4), ind_new(5):ind_new(6), :) = 1;
            
            img_tmp = img_tmp .* mask_tmp;
                        
            imgnorm_unblock = imgnorm_unblock + img_tmp;
            count_map = count_map + mask_tmp;
        end
        
        count_map(count_map == 0 ) = 1;
        imgnorm_pred = imgnorm_unblock ./ count_map .* bmask;
        
        if jj == 1 
            fpPred = fullfile(dpPred, [name '_imgnorm_single.nii.gz']);
            save=make_nii(imgnorm_pred .* bmask)
            save_nii( save,fpPred);
        end
        
        img_pred = zeros(size(imgnorm_pred));
        
        for tt = 1 : size(metrics, 4)
            tmp = metrics(:, :, :, tt);
            img_pred(:, :, :, tt) = imgnorm_pred(:, :, :, tt) * std(tmp(bmask)) + mean(tmp(bmask));
        end
        
        if jj == 1
            fpPred = fullfile(dpPred, [name '_img_single.nii.gz']);
            save=make_nii(img_pred .* bmask)
            save_nii( save,fpPred);
        end
           
        img_allavgs = cat(5, img_allavgs, img_pred);
    end    
    
    img_avg = mean(img_allavgs, 5);
    fpAvg = fullfile(dpPred, [name  '_img_avg.nii.gz']);
    save=make_nii(img_avg .* bmask)
    save_nii( save,fpAvg);
    
    
     
    if any(strfind(name, '_drop_'))
        img_std = std(img_allavgs, [], 5);
        img_std = img_std ./ img_avg;
        img_avg(repmat(~bmask, 1, 1, 1, 2)) = 0;

        fpStd = fullfile(dpPred, [name '_img_std.nii.gz']);
        save=make_nii(img_std .* bmask)
         save_nii( save,fpStd);
    end
    
end
