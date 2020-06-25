import tensorflow
import nibabel as nib
import numpy as np
from glob import glob
import os
import cv2

def dice_loss(y_true, y_pred):
  numerator = 2 * tensorflow.reduce_sum(y_true * y_pred, axis=-1)
  denominator = tensorflow.reduce_sum(y_true + y_pred, axis=-1)
  return 1 - (numerator + 1) / (denominator + 1)

def execute():
    # TODO read in your parameters from the JSON file
    input_folder = 'data_eval'
    model_input_size = (64,64)
    data_input_size = (128,128)
    
    print('load model')
    model = tensorflow.keras.models.load_model('model.h5',custom_objects={'dice_loss': dice_loss})
    # optionally consider loading best weights here as well
    
    print('searching for data')
    inputFiles = glob(os.path.join(input_folder,'*.nii.gz'),recursive=True) + glob(os.path.join(input_folder,'*.nii'),recursive=True)
    
    print('evaluating model')
    for f in inputFiles:
        print('processing {}'.format(f))
        
        # load data
        nii = nib.load(f)
        nii_data = nii.get_fdata()
        out_data = np.zeros_like(nii_data)
        
        # this code assumes a single 2D input slice input and single 2D output slice
        # TODO fix this limitation
        
        # loop through slices in the z direction
        for z in range(0,nii_data.shape[2]):
            # resize input data
            curr_input = cv2.resize( nii_data[:,:,z], dsize=model_input_size, interpolation = cv2.INTER_CUBIC)
            curr_input = curr_input[ np.newaxis, ..., np.newaxis]
            curr_output = model.predict( curr_input )
            
            # convert from channel-encoding to integer encoding
            for c in range(0,curr_output.shape[3]):
                out_data[:,:,z] += cv2.resize( curr_output[0,:,:,c] * (c+1), dsize=data_input_size, interpolation = cv2.INTER_NEAREST )
        
        # save file
        # first create new Nifti object based on the input
        new_nii = nib.Nifti1Image(out_data, nii.affine, nii.header)
        # next determine the filename
        drive, filepath = os.path.splitdrive( f )
        path, filename = os.path.split( filepath )
        new_base = filename.replace('.nii','_seg.nii')
        new_file = os.path.join(drive,path,new_base)
        # now write to disk
        nib.save(new_nii, new_file)
    
    print('done')

if __name__ == "__main__": 
    execute()