import json

from sklearn.tests.test_base import K

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

# def intersection_over_union(y_true, y_pred, smooth=1):
#     intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
#     union = K.sum(y_true, -1) + K.sum(y_pred, -1) - intersection
#     iou = (intersection + smooth) / (union + smooth)
#     return iou

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
    input_folder = MedML.get('Folder name')
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