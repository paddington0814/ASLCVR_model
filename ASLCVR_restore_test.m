clear all;
close all;
clc
addpath(genpath('C:\Users\David\Documents\MATLAB\Tools\'));
cd 'D:\Project\O-15\ASLCVR\ASLCVR\final\result_final\aug16_58_2\'
path = ['D:\Project\O-15\ASLCVR\ASLCVR\final\result_final\aug16_58_2\'];
num_fold = 6;
%% test list 
% test = {'HC025','HC028','HC033','HC034','HC035','HC036','HC037','HC038'};
% test = {'MM042','MM047','MM048','MM054'}; % 1
% test = {'MM049','MM039','MM051','MM052'}; % 2
% test = {'MM045','MM043','MM037','MM041'}; % 3
% test = {'MM044','MM050','MM034','MM040'}; % 4
% test = {'MM031','MM032','MM038','MM035'}; % 5

% test = {'HC025','HC028','HC033','HC034','HC035','HC036','HC037','HC038'};
% test = {'HC038','HC039','MM042','MM047','MM048','MM054'}; % 1
% test = {'HC036','HC037','MM049','MM039','MM051','MM052'}; % 2
% test = {'HC034','HC035','MM045','MM043','MM037','MM041'}; % 3
% test = {'HC030','HC033','MM044','MM050','MM034','MM040'}; % 4
% test = {'HC025','HC028','MM031','MM032','MM038','MM035'}; % 5

% 6fold_testcase
% test = {'HC043_1','HC044_1','MM057_1','MM058_1','MM059_1'}; % 1
% test = {'HC043_2','HC044_2','MM057_2','MM058_2','MM059_2'}; % 2
% test = {'HC043_3','HC044_3','MM057_3','MM058_3','MM059_3'}; % 3
% test = {'HC043_4','HC044_4','MM057_4','MM058_4','MM059_4'}; % 4
% test = {'HC043_5','HC044_5','MM057_5','MM058_5','MM059_5'}; % 5
% test = {'HC043_6','HC044_6','MM057_6','MM058_6','MM059_6'}; % 6

% 7fold1
% test = {'HC038','HC039','MM059','MM051','MM048','MM054'}; % 1
% test = {'HC036','HC037','MM049','MM039','MM029','MM052'}; % 2
% test = {'HC034','HC035','MM045','MM043','MM037','MM041'}; % 3
% test = {'HC030','HC033','MM044','MM050','MM046','MM040'}; % 4
% test = {'HC025','HC028','MM031','MM032','MM038','MM035'}; % 5
% test = {'HC040','HC042','MM028','MM042','MM034','MM056'}; % 6
% test = {'HC043','HC044','MM045','MM057','MM059','MM047'}; % 7

% 6fold final
test_1 = {'HC040','HC042','MM044','MM051','MM058','MM059'}; % 1
test_2 = {'HC038','HC039','MM042','MM037','MM041','MM053'}; % 2
test_3 = {'HC036','HC037','MM054','MM035','MM038','MM048'}; % 3
test_4 = {'HC043','HC035','MM043','MM032','MM049','MM046'}; % 4
test_5 = {'HC030','HC033','MM040','MM056','MM045','MM052'}; % 5
test_6 = {'HC025','HC028','MM039','MM031','MM034','MM050'}; % 6

test = {{'HC040','HC042','MM044','MM051','MM058','MM059'},...
        {'HC038','HC039','MM042','MM037','MM041','MM057'},...
        {'HC036','HC037','MM054','MM035','MM038','MM048'},...
        {'HC043','HC035','MM043','MM032','MM049','MM046'},...
        {'HC030','HC033','MM040','MM056','MM045','MM052'},...
        {'HC025','HC028','MM039','MM031','MM034','MM050'}}; % 6

% MM057
% test = {{'MM057_1'},...
%         {'MM057_2'},...
%         {'MM057_3'},...
%         {'MM057_4'},...
%         {'MM057_5'},...
%         {'MM057_6'}}; % 6

% 6fold1
% test = {'HC038','HC039','MM042','MM029','MM048','MM054'}; % 1
% test = {'HC036','HC037','MM049','MM039','MM051','MM052'}; % 2
% test = {'HC034','HC035','MM045','MM043','MM037','MM041'}; % 3
% test = {'HC030','HC033','MM044','MM050','MM034','MM040'}; % 4
% test = {'HC025','HC028','MM031','MM032','MM038','MM035'}; % 5
% test = {'HC040','HC042','MM028','MM047','MM046','MM056'}; % 6

% 6fold2
% test = {'HC038','HC039','MM046','MM029','MM032','MM031'}; % 1
% test = {'HC036','HC037','MM038','MM034','MM035','MM047'}; % 2
% test = {'HC034','HC035','MM050','MM028','MM044','MM040'}; % 3
% test = {'HC030','HC033','MM041','MM045','MM042','MM043'}; % 4
% test = {'HC025','HC028','MM052','MM051','MM049','MM039'}; % 5
% test = {'HC025','HC028','MM048','MM056','MM037','MM054'}; % 6


% test = {'MM054','MM051','MM052'}; % 1
% test = {'MM041','MM039','MM053'}; % 2
% test = {'MM043','MM044','MM049'}; % 3
% test = {'MM034','MM040','MM045'}; % 4
% test = {'MM035','MM037','MM038'}; % 5
% test = {'MM031','MM032','MM050'}; % 6

% test = {'MM051','MM052'}; % 1
% test = {'MM041','MM039'}; % 2
% test = {'MM044','MM049'}; % 3
% test = {'MM045','MM043',}; % 4
% test = {'MM034','MM040'}; % 5
% test = {'MM037','MM038'}; % 6
% test = {'MM050','MM035'}; % 7
% test = {'MM031','MM032'}; % 8

% test = {'MM039','MM051','MM052','MM047','MM054'}; % 1
% test = {'MM043','MM044','MM049','MM041','MM042'}; % 2
% test = {'MM038','MM034','MM040','MM048','MM045'}; % 3   
% test = {'MM031','MM032','MM050','MM035','MM037'}; % 4

% test = {'MM035','MM051','MM052','MM054'}; % 1
% test = {'MM045','MM047','MM043','MM049'}; % 2
% test = {'MM041','MM042','MM048','MM044'}; % 3   
% test = {'MM037','MM038','MM039','MM040'}; % 4
% test = {'MM031','MM032','MM034','MM050'}; % 5

% test = {'MM052','MM051'}; % 1
% test = {'MM045','MM037'}; % 2
% test = {'MM043','MM049'}; % 3
% test = {'MM035','MM040'}; % 4
% test = {'MM044','MM039'}; % 5
% test = {'MM038','MM041'}; % 6
% test = {'MM050','MM034'}; % 7
% test = {'MM032','MM031'}; % 8
% test = {'MM034','MM050'}; % 9
% test = {'MM031','MM032'}; % 10

% test = {'MM029', 'VA003', 'VA017', 'VA019'}; % 1
% test = {'MM022', 'MM025', 'MM027', 'MM028'}; % 2
% test = {'MM010', 'MM013', 'MM014', 'MM016'}; % 3
% test = {'MM004', 'MM007', 'MM008', 'MM009'}; % 4

%%  
for i = 1:num_fold
    path_file = [path 'result_final_init_' num2str(i) '_resmat.mat'];
    data = load(path_file);
    data = data.data_test_output;
    data = permute(data, [2 3 1]);
    
    for j = 1:6
        name_test = char(test{i}{j});
        slice = 1+70*(j-1):70*j;
        output = zeros([91 109 91]);
        output(:, 7:102, 8:77) = data(4:94, :, slice);

        nii = make_nii(output);
        nii.hdr.dime.dim = [3,91,109,91,1,1,1,1];
        nii.hdr.dime.pixdim = [1,2,2,2,0,0,0,0];
        nii.hdr.dime.xyzt_units = 2;
        nii.hdr.dime.datatype = 16;
        nii.hdr.dime.bitpix = 32; 
        nii.hdr.hist.sform_code = 1;
        nii.hdr.hist.qform_code = 0;
        nii.hdr.hist.qoffset_x = 90;
        nii.hdr.hist.qoffset_y = -126;
        nii.hdr.hist.qoffset_z = -72;
        nii.hdr.hist.srow_x = [-2,0,0,90];
        nii.hdr.hist.srow_y = [0,2,0,-126];
        nii.hdr.hist.srow_z = [0,0,2,-72];
        nii.hdr.hist.originator = [46,64,37,0,-32768];
        nii.hdr.hist.rot_orient = [1,2,3];
        nii.hdr.hist.flip_orient = [3,0,0];
        save_nii(nii, [name_test '_synth_pet_cvr_abs_init.nii']);
    end
    
end
    
% output = zeros([2 70 96 96]);
% 
% % rescale synth CBF to ml/100g/min
% for i = 1:2
%     for j = 1:70
%         output(i,j,:,:) = Data.data_test_output((i-1)*70+j,:,:);  
%     end
% end

% 
% % restore individual CBF, CVR
%     for j = 1:2:7
%         cbf_pre = flipud(squeeze(synth_cbf(i,:,:,((j-1)*70+1):j*70)));
%         cbf_post = flipud(squeeze(synth_cbf(i,:,:,(j*70+1):(j+1)*70)));
%         cvr_abs = (cbf_post - cbf_pre);
%         cvr_ratio = 100*(cbf_post - cbf_pre)./cbf_pre;
%     
%         subname = char(List.list((j+1)/2,i));
%         fname_cbf_pre = [subname '_cbf_synth_pre.nii'];
%         fname_cbf_post = [subname '_cbf_synth_post.nii'];
%         fname_cvr_abs = [subname '_cvr_synth_abs.nii'];
%         fname_cvr_ratio = [subname '_cvr_synth_ratio.nii'];
%     
%         nii = make_nii(cbf_pre); save_nii(nii, fname_cbf_pre);
%         nii = make_nii(cbf_post); save_nii(nii, fname_cbf_post);
%         nii = make_nii(cvr_abs); save_nii(nii, fname_cvr_abs);
%         nii = make_nii(cvr_ratio); save_nii(nii, fname_cvr_ratio);
%     end
