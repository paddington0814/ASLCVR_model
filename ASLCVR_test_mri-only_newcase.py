from scipy import io as sio
import numpy as np
import os
import nibabel as nib
import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from keras.utils import to_categorical
from skimage.measure import compare_mse, compare_nrmse, compare_psnr, compare_ssim
import json

from my_cafndl_fileio import *
from cafndl_utils import *
from my_cafndl_network_convres_input_weight import *
from cafndl_metrics import *


### specify the name of the case to be inferenced here ###

case_name = "newcase"


###

if not os.path.exists('/[model_directory]/{0}'.format(case_name)):
    os.makedirs('/[model_directory]/{0}'.format(case_name))


data_dir = '/[data_directory]/'
filename_mask = '/[MNI_brain_mask_filled.nii]/'

M_model = ['/[model_directory]/model_ASLCVR_fine_1.ckpt',]
    
Res_dir = ['/[result_directory]/{0}'.format(case_name)]

Res_name = ['syn_cvr']	
					
chnn_list = [range(0,11)]

scale_channels = [
					0,
					0,
					3,
					0,
					0,
					0,
					0,
					0,
					0,
					0,
					0,
				]



test_subs = [
		['{0}'.format(case_name)],
            ]


'''
use linear activation, with coordinates as input and MNI brain mask applied
'''
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
keras_memory = 0.6
batch_size = 96
with_batch_norm = True
ylim = np.array([-2, 2]) # activation function [-0.5,0.5] tan; [0.1, 0.5] sig; [-2, 2] linear

savefig = 0
subtest = 'subs'
slice2use = range(5,75) #range(5,75) #range(30,40)+range(50,60) #range(20,65)

test_runs = [
                [[1],],
			]
											
N_pool = [3]
N_conv = [3]
pool_meth = ['max']

Ch_pow_base = [2]


for ii in range(1):
    '''
    dataset
    '''
    filename_checkpoint = M_model[ii]
    channel2use = chnn_list[ii]
    num_poolings = N_pool[ii]
    num_conv_per_pooling = N_conv[ii]
    pooling_method = pool_meth[ii]
    chnl_power_base = Ch_pow_base[ii]
    if not os.path.exists(Res_dir[ii]):
        os.makedirs(Res_dir[ii])
    filename_results = Res_dir[ii] + '/' + Res_name[ii]

    list_dataset_test = gen_input_list(data_dir, test_subs[ii], test_runs[ii], slice2use=slice2use, channel2use=channel2use)
    print(list_dataset_test)

    num_dataset_test = len(list_dataset_test)                
    print('process {0} data description'.format(num_dataset_test))

    '''
    augmentation
    '''
    list_augments = []

    '''
    generate test data
    '''
    list_test_input = []
    index_sample_total = 0
    for index_data in range(num_dataset_test):
        # directory
        list_data_test_input = []
        # subset of slices
        try:
            slices = np.array(list_dataset_test[index_data]['slices'])
        except:
            slices = None
        # subset of channels
        try:
            channels = np.array(list_dataset_test[index_data]['channels'])
        except:
            channels = None

        # get different contrasts
        count_ch = 0
        scale_CBF = 100
        scale_PET = 50
        for path_test_input in list_dataset_test[index_data]['inputs']:					
            # get scaling factor for channels			
            scale_factor = scale_channels[count_ch]
            #print(count_ch,scale_factor)
            if scale_factor == 0:
                scale_by_mean = True
                scale_by_norm = False
                scale_factor = 0.1
            elif scale_factor == 1: #cbf
                scale_by_mean = False
                scale_by_norm = False
                scale_factor = 0.01  
            elif scale_factor == 2: #cvr
                scale_by_mean = False
                scale_by_norm = False
                scale_factor = 0.02 
            elif scale_factor == 3: #ATT
                scale_by_mean = False
                scale_by_norm = False
                scale_factor = 0.000333
            elif scale_factor == 4: #Tmax
                scale_by_mean = False
                scale_by_norm = False
                scale_factor = 0.12
            else:
                scale_by_mean = False

            # load data
            if count_ch == 16:
                data_test_input = my_prepare_data_from_nifti_25D(
                    path_test_input, list_augments,
                    scale_by_mean=scale_by_mean, scale_by_norm=scale_by_norm, scale_factor = scale_factor,
                    slices = slices, maskfile = filename_mask, thickness=7)

                list_data_test_input.append(data_test_input)
                count_ch += 1

            else:
                data_test_input = my_prepare_data_from_nifti(
                    path_test_input, list_augments,
                    scale_by_mean=scale_by_mean, scale_by_norm=scale_by_norm, scale_factor = scale_factor,
                    slices = slices, maskfile = filename_mask)

                list_data_test_input.append(data_test_input)
                count_ch += 1


        data_test_input = np.concatenate(list_data_test_input, axis=-1)


        # append
        list_test_input.append(data_test_input)


    # generate test dataset
    data_test_input = np.concatenate(list_test_input, axis = 0)



    # get mask, GJ
    nib_mask = nib.load(filename_mask)
    im_maskall = nib_mask.get_data()
    if np.ndim(im_maskall)==3:
        im_maskall = im_maskall[:,:,:,np.newaxis]	
    im_maskall = np.transpose(im_maskall, [2,0,1,3])
    im_maskall = im_maskall[slice2use,:,:,:]
    print(im_maskall.shape)
    im_maskall = np.tile(im_maskall,[num_dataset_test,1,1,1])


    # bigger ROI for more possible layers 96x96
    data_test_input_new = np.zeros([data_test_input.shape[0],96,96,data_test_input.shape[-1]])
    data_test_input_new[:,3:94,:,:] = data_test_input[:,:,7:103,:]
    data_test_input = data_test_input_new
    data_test_input_new = None


    im_maskall_new = np.zeros([im_maskall.shape[0],96,96,im_maskall.shape[-1]])
    im_maskall_new[:,3:94,:,:] = im_maskall[:,:,7:103,:]
    im_maskall = im_maskall_new
    im_maskall_new = None

    print('crop to power of 2:', data_test_input.shape)

    '''
    save input figure
    ''' 
    ## display
    #slice2show = 35
    #nch2use = data_test_input.shape[-1]
    #im_inputall = np.squeeze(data_test_input[slice2show,:,:,0]).T
    #for ich in range(1,nch2use):
        #im_inputall = np.concatenate((im_inputall, np.squeeze(data_test_input[slice2show,:,:,ich]).T),axis=1)

    ## translate
    #im_inputall = np.flipud(im_inputall)
    #fig = plt.figure(figsize=[2*nch2use,2])
    #ax = plt.imshow(im_inputall, clim=[-0.05,1.], cmap='jet')
    #im_title = '{0} input, channels: {1}'.format(test_subs[0][0], channel2use)
    #plt.title(im_title)
    #fig.colorbar(ax)
    #path_figure = filename_results+'_insl{0}.png'.format(slice2show,)
    #plt.savefig(path_figure)
    '''
    end save input figure
    '''  

    '''
    setup parameters
    '''
    keras_backend = 'tf'
    num_channel_input = data_test_input.shape[-1]
    img_rows = data_test_input.shape[1]
    img_cols = data_test_input.shape[1]
    print('setup parameters')


    '''
    define model
    '''
    setKerasMemory(keras_memory)
    model = my_deepEncoderDecoder(num_channel_input = num_channel_input,
                            num_channel_output = 1,
                            img_rows = img_rows,
                            img_cols = img_cols,
                            #lr_init = lr_init, 
                            num_poolings = num_poolings, 
                            num_conv_per_pooling = num_conv_per_pooling, 
                            with_bn = with_batch_norm, verbose=1,
                            pooling_method = pooling_method,
                            chnl_power_base = chnl_power_base,
                            y=ylim)


    print('trained model:', filename_checkpoint)
    print('num_channel_input:', num_channel_input)
    print('num_channel_output:', 1)
    print('img_rows:', img_rows)
    print('img_cols:', img_cols)
    print('num_poolings:', num_poolings)
    print('num_conv_per_pooling:', num_conv_per_pooling)
    print('pooling_method:', pooling_method)
    print('chnl_power_base:', chnl_power_base)
    print('parameter count:', model.count_params())
    print('channel used:', channel2use)
    print('slice used:', slice2use)

    '''
    load network
    '''
    model.load_weights(filename_checkpoint)
    print('model load from' + filename_checkpoint)        

    '''
    apply model
    '''
    t_start_pred = datetime.datetime.now()
    data_test_output = model.predict(data_test_input, batch_size=batch_size)
    # clamp
    clamp_min = -0.05
    clamp_max = 5
    data_test_output = np.maximum(np.minimum(data_test_output, clamp_max), clamp_min)

    # scale back
    for i in range(70):
        data_test_output[i,:,:,:] *= 2.0

    t_end_pred = datetime.datetime.now()
    print('predict on data size {0} using time {1}'.format(
        data_test_output.shape, t_end_pred - t_start_pred))

    # save predict
    path_matresults = filename_results+'_resmat.mat'
    sio.savemat(path_matresults, {
                    'data_test_output':data_test_output,
                    })
    print('results exported to .mat at: {0}'.format(path_matresults))
