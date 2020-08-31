import json

from lovasz_losses_tf import *
import tensorflow
import nibabel as nib
import numpy as np
from glob import glob
import os
import cv2

f = open('MedML.json')
MedML = json.load(f)

g = open('OptimizerParameters.json')
OptParam = json.load(g)

def dice_loss(y_true, y_pred):
  numerator = 2 * tensorflow.reduce_sum(y_true * y_pred, axis=-1)
  denominator = tensorflow.reduce_sum(y_true + y_pred, axis=-1)
  return 1 - (numerator + 1) / (denominator + 1)

def balanced_cross_entropy(beta):
  def convert_to_logits(y_pred):
      # see https://github.com/tensorflow/tensorflow/blob/r1.10/tensorflow/python/keras/backend.py#L3525
      y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())

      return tf.log(y_pred / (1 - y_pred))

  def loss(y_true, y_pred):
    y_pred = convert_to_logits(y_pred)
    pos_weight = beta / (1 - beta)
    loss = tf.nn.weighted_cross_entropy_with_logits(logits=y_pred, targets=y_true, pos_weight=pos_weight)

    # or reduce_sum and/or axis=-1
    return tf.reduce_mean(loss * (1 - beta))

  return loss

def weighted_cross_entropy(beta):
  def convert_to_logits(y_pred):
      # see https://github.com/tensorflow/tensorflow/blob/r1.10/tensorflow/python/keras/backend.py#L3525
      y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())

      return tf.log(y_pred / (1 - y_pred))

  def loss(y_true, y_pred):
    y_pred = convert_to_logits(y_pred)
    loss = tf.nn.weighted_cross_entropy_with_logits(logits=y_pred, targets=y_true, pos_weight=beta)

    # or reduce_sum and/or axis=-1
    return tf.reduce_mean(loss)

  return loss

def intersection_over_union(num_classes):
    return tf.keras.metrics.MeanIoU(num_classes=num_classes)

def tversky_loss(beta):
    def loss(y_true, y_pred):
        numerator = tensorflow.reduce_sum(y_true * y_pred, axis=-1)
        denominator = y_true * y_pred + beta * (1 - y_true) * y_pred + (1 - beta) * y_true * (1 - y_pred)

        return 1 - (numerator + 1) / (tensorflow.reduce_sum(denominator, axis=-1) + 1)

    return loss

def lovasz_softmax(y_true, y_pred):
  return lovasz_hinge(labels=y_true, logits=y_pred)


def execute():
    # TODO read in your parameters from the JSON file
    X = MedML.get('X')
    Y = MedML.get('Y')
    sliceSamples = int(MedML.get('Slice samples'))
    #This will be the name of the data eval folder
    input_folder = 'data_eval'
    model_input_size = (int(X), int(Y))


    lossDict = MedML.get('Loss function')

    # check and call the selected loss and insert parameter
    if (lossDict == "dice_loss"):
        lossVal = dice_loss
    if (lossDict == "balanced_cross_entropy"):
        lossVal = balanced_cross_entropy
    if (lossDict == "weighted_cross_entropy"):
        lossVal = weighted_cross_entropy
    if (lossDict == "intersection_over_union"):
        lossVal = intersection_over_union(MedML.get('Number of segmentation classes'))
    if (lossDict == "tversky_loss"):
        lossVal = tversky_loss
    if (lossDict == "lovasz_softmax"):
        lossVal = lovasz_softmax
    
    print('load model')
    model = tensorflow.keras.models.load_model('model.h5',custom_objects={lossDict: lossVal})
    # optionally consider loading best weights here as well
    
    print('searching for data')
    inputFiles = glob(os.path.join(input_folder,'*.nii.gz'),recursive=True) + glob(os.path.join(input_folder,'*.nii'),recursive=True)
    
    print('evaluating model')
    for f in inputFiles:
        print('processing {}'.format(f))
        
        # load data
        nii = nib.load(f)
        nii_data = nii.get_fdata()
        data_input_size = (nii_data.shape[0], nii_data.shape[1])
        out_data = np.zeros_like(nii_data)
        
        # loop through slices
        for i in range(0,nii_data.shape[2]-sliceSamples):

            # determine sampling range
            if sliceSamples == 1:
                z = i
                seg_z = z + sliceSamples//2
            else:
                # warning, sliceSamples is assumed to be 1,3,5,7,9
                z = range(i,i+sliceSamples)
                
                # output segmentation will be the center slice
                seg_z = z[0] + sliceSamples//2

            # resize input data to fit model
            curr_input = cv2.resize( nii_data[:,:,z], dsize=model_input_size, interpolation = cv2.INTER_CUBIC)
            if sliceSamples == 1:
                curr_input = curr_input[ np.newaxis, ..., np.newaxis] # make input size= [1,X,Y,1]
            else:
                curr_input = curr_input[np.newaxis, ...] # make input size= [1,X,Y,sliceSamples]

            # predict segmentation
            curr_output = model.predict( curr_input )
            
            # convert from channel-encoding to integer encoding
            curr_output_flat = np.zeros( (curr_output.shape[1], curr_output.shape[2]) )
            for c in range(0,curr_output.shape[3]):
                curr_output_flat += np.squeeze(curr_output[0,:,:,c])
                
            # reshape segmentation to match input data size
            curr_output_flat = cv2.resize( curr_output_flat, dsize=data_input_size, interpolation = cv2.INTER_NEAREST )
                
            # store in output array
            out_data[:,:,seg_z] = curr_output_flat

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