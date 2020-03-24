clear all;
close all;
clc

cd '[result_directory]'
path = '[result_directory]';
num_fold = 1;

test = {{'newcase'}}; 


%%  
for i = 1:num_fold
    path_file = [path 'result_final_init_' num2str(i) '_resmat.mat'];
    data = load(path_file);
    data = data.data_test_output;
    data = permute(data, [2 3 1]);
    
    for j = 1
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
        save_nii(nii, [name_test '_syn_cvr.nii']);
    end
    
end
