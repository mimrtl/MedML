import json
import os

from sklearn.tests.test_base import K

os.environ['TF_KERAS'] = '1'
os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1'
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import tensorflow
from keras_radam import RAdam
from tensorflow.keras.callbacks import History, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.optimizers import Adagrad
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.optimizers import Nadam
from tensorflow.keras.optimizers import Adamax
from tensorflow.keras import losses
import Unet
from lovasz_losses_tf import *
from NiftiGenerator import NiftiGenerator

tensorflow.keras.backend.set_image_data_format('channels_last')

f = open('MedML.json')
MedML = json.load(f)

g = open('OptimizerParameters.json')
OptParam = json.load(g)


    #call Adam
def callAdam(learnRate, beta1, beta2, epsilon, amsgrad):
    return Adam(lr=learnRate, beta_1=beta1, beta_2=beta2, epsilon=epsilon, amsgrad=amsgrad)

    #call Rectified Adam
def callRectifiedAdam(learnRate, beta1, beta2, epsilon, decay, weightDecay, amsgrad, totalSteps, warmUpProp, minLr):
    return RAdam(lr=learnRate,beta_1=beta1,beta_2=beta2,epsilon=epsilon,decay=decay, weight_decay=weightDecay, amsgrad=amsgrad,total_steps=totalSteps, warmup_proportion=warmUpProp, min_lr=minLr)

def callRMSprop(learnRate, rho, momentum, epsilon, centered):
    return RMSprop(lr=learnRate, rho=rho, momentum=momentum, epsilon=epsilon,centered=centered)

def callAdagrad(learnRate, initialAccumVal, epsilon):
    return Adagrad(lr=learnRate, initial_accumulator_value=initialAccumVal, epsilon=epsilon)

def callSGD(learnRate, momentum, nesterov):
    return SGD(lr=learnRate, momentum=momentum, nesterov=nesterov)

def callNadam(learnRate, beta1, beta2, epsilon):
    return Nadam(lr=learnRate, beta_1=beta1, beta_2=beta2, epsilon=epsilon)

def callAdamax(learnRate, beta1, beta2, epsilon):
    return Adamax(lr=learnRate, beta_1=beta1, beta_2=beta2, epsilon=epsilon)

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

def intersection_over_union(y_true, y_pred, smooth=1):
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    union = K.sum(y_true, -1) + K.sum(y_pred, -1) - intersection
    iou = (intersection + smooth) / (union + smooth)
    return iou

def tversky_loss(beta):
    def loss(y_true, y_pred):
        numerator = tensorflow.reduce_sum(y_true * y_pred, axis=-1)
        denominator = y_true * y_pred + beta * (1 - y_true) * y_pred + (1 - beta) * y_true * (1 - y_pred)

        return 1 - (numerator + 1) / (tensorflow.reduce_sum(denominator, axis=-1) + 1)

    return loss

def lovasz_softmax(y_true, y_pred):
  return lovasz_hinge(labels=y_true, logits=y_pred)


def execute():
    # read in your parameters from the JSON file
    # TODO
    incRate = MedML.get('Filter increasing rate')
    activFunc = MedML.get('Activation function')
    dropRate = MedML.get('Dropout rate')
    batchNorm = MedML.get('Batch Normalization')
    optimizerDict = MedML.get('Optimizer')
    lossDict = MedML.get('LossFunction')
    lrDict = MedML.get('Learning rate')
    Out_Ch = MedML.get('Out_ch')
    Start_Ch = MedML.get('Start_ch')
    Depth = MedML.get('Depth')
    Maxpool = MedML.get('Maxpool')
    Upconv = MedML.get('Upconv')
    Residual = MedML.get('Residual')

    print('creating model')
    model = Unet.UNet([64,64,1],out_ch=int(Out_Ch),start_ch=int(Start_Ch),depth=int(Depth),inc_rate=float(incRate),activation=activFunc,dropout=float(dropRate),batchnorm=batchNorm,maxpool=Maxpool,upconv=Upconv,residual=Residual)

    #check and call the selected optimizer and insert parameter
    if(optimizerDict == "Adam"):
        if(OptParam.get('Beta1') == "(float)") or (OptParam.get('Beta1') == "True/False") or (OptParam.get('Beta1') == "") or not (os.path.exists('OptimizerParameters.json')):
            Beta1 = 0.9
        else:
            Beta1 = OptParam.get('Beta1')

        if (OptParam.get('Beta2') == "(float)") or (OptParam.get('Beta2') == "True/False") or (
                    OptParam.get('Beta2') == "") or not (os.path.exists('OptimizerParameters.json')):
            Beta2 = 0.999
        else:
            Beta2 = OptParam.get('Beta2')

        if (OptParam.get('Epsilon') == "(float)") or (OptParam.get('Epsilon') == "True/False") or (
                    OptParam.get('Epsilon') == "") or not (os.path.exists('OptimizerParameters.json')):
            Epsilon = 1e-7
        else:
            Epsilon = OptParam.get('Epsilon')

        if (OptParam.get('Amsgrad') == "(float)") or (OptParam.get('Amsgrad') == "True/False") or (
                    OptParam.get('Amsgrad') == "") or not (os.path.exists('OptimizerParameters.json')):
            Amsgrad = False
        else:
            Amsgrad = OptParam.get('Amsgrad')

        optimizerVal = callAdam(float(lrDict), float(Beta1), float(Beta2), float(Epsilon), Amsgrad)

    if(optimizerDict == "RectifiedAdam"):
        if (OptParam.get('Beta1') == "(float)") or (OptParam.get('Beta1') == "True/False") or (
                OptParam.get('Beta1') == "") or not (os.path.exists('OptimizerParameters.json')):
            Beta1 = 0.9
        else:
            Beta1 = OptParam.get('Beta1')
        if (OptParam.get('Beta2') == "(float)") or (OptParam.get('Beta2') == "True/False") or (
                    OptParam.get('Beta2') == "") or not (os.path.exists('OptimizerParameters.json')):
            Beta2 =0.999
        else:
            Beta2 = OptParam.get('Beta2')
        if (OptParam.get('Epsilon') == "(Any)") or (OptParam.get('Epsilon') == "True/False") or (
                OptParam.get('Epsilon') == "") or not (os.path.exists('OptimizerParameters.json')):
            Epsilon = None
        else:
            Epsilon = OptParam.get('Epsilon')
        if (OptParam.get('Decay') == "(float)") or (OptParam.get('Decay') == "True/False") or (
                OptParam.get('Decay') == "") or not (os.path.exists('OptimizerParameters.json')):
            Decay = 0.
        else:
            Decay = OptParam.get('Decay')
        if (OptParam.get('WeightDecay') == "(float)") or (OptParam.get('WeightDecay') == "True/False") or (
                OptParam.get('WeightDecay') == "") or not (os.path.exists('OptimizerParameters.json')):
            WeightDecay = 0.
        else:
            WeightDecay = OptParam.get('WeightDecay')
        if (OptParam.get('Amsgrad') == "(float)") or (OptParam.get('Amsgrad') == "True/False") or (
                OptParam.get('Amsgrad') == "") or not (os.path.exists('OptimizerParameters.json')):
            Amsgrad = False
        else:
            Amsgrad = OptParam.get('Amsgrad')
        if (OptParam.get('TotalSteps') == "(int)") or (OptParam.get('TotalSteps') == "True/False") or (
                OptParam.get('TotalSteps') == "") or not (os.path.exists('OptimizerParameters.json')):
            TotalSteps = 0
        else:
            TotalSteps = OptParam.get('TotalSteps')
        if (OptParam.get('WarmUpProportion') == "(float)") or (OptParam.get('WarmUpProportion') == "True/False") or (
                OptParam.get('WarmUpProportion') == "") or not (os.path.exists('OptimizerParameters.json')):
            WarmUpProportion = 0.1
        else:
            WarmUpProportion = OptParam.get('WarmUpProportion')
        if (OptParam.get('MinLr') == "(float)") or (OptParam.get('MinLr') == "True/False") or (
                OptParam.get('MinLr') == "") or not (os.path.exists('OptimizerParameters.json')):
            MinLr = 0.
        else:
            MinLr = OptParam.get('MinLr')

        optimizerVal = callRectifiedAdam(float(lrDict), float(Beta1), float(Beta2), Epsilon, float(Decay), float(WeightDecay), Amsgrad, int(TotalSteps), float(WarmUpProportion), float(MinLr))

    if(optimizerDict == "RMSprop"):
        if (OptParam.get('Rho') == "(float)") or (OptParam.get('Rho') == "True/False") or (
                OptParam.get('Rho') == "") or not (os.path.exists('OptimizerParameters.json')):
            Rho = 0.9
        else:
            Rho = OptParam.get('Rho')

        if (OptParam.get('Momentum') == "(float)") or (OptParam.get('Momentum') == "True/False") or (
                OptParam.get('Momentum') == "") or not (os.path.exists('OptimizerParameters.json')):
            Momentum = 0.0
        else:
            Momentum = OptParam.get('Momentum')
        if (OptParam.get('Epsilon') == "(float)") or (OptParam.get('Epsilon') == "True/False") or (
                OptParam.get('Epsilon') == "") or not (os.path.exists('OptimizerParameters.json')):
            Epsilon = 1e-07
        else:
            Epsilon = OptParam.get('Epsilon')
        if (OptParam.get('Centered') == "(float)") or (OptParam.get('Centered') == "True/False") or (
                OptParam.get('Centered') == "") or not (os.path.exists('OptimizerParameters.json')):
            Centered = False
        else:
            Centered = OptParam.get('Centered')

        optimizerVal = callRMSprop(float(lrDict), float(Rho), float(Momentum), float(Epsilon), Centered)

    if(optimizerDict == "Adagrad"):
        if (OptParam.get('InitialAccumVal') == "(float)") or (OptParam.get('InitialAccumVal') == "True/False") or (
                OptParam.get('InitialAccumVal') == "") or not (os.path.exists('OptimizerParameters.json')):
            InitialAccumVal = 0.1
        else:
            InitialAccumVal = OptParam.get('InitialAccumVal')
        if (OptParam.get('Epsilon') == "(float)") or (OptParam.get('Epsilon') == "True/False") or (
                OptParam.get('Epsilon') == "") or not (os.path.exists('OptimizerParameters.json')):
            Epsilon = 1e-7
        else:
            Epsilon = OptParam.get('Epsilon')

        optimizerVal = callAdagrad(float(lrDict), float(InitialAccumVal), float(Epsilon))

    if(optimizerDict == "SGD"):
        if (OptParam.get('Momentum') == "(float)") or (OptParam.get('Momentum') == "True/False") or (
                OptParam.get('Momentum') == "") or not (os.path.exists('OptimizerParameters.json')):
            Momentum = 0.0
        else:
            Momentum = OptParam.get('Momentum')
        if (OptParam.get('Nesterov') == "(float)") or (OptParam.get('Nesterov') == "True/False") or (
                OptParam.get('Nesterov') == "") or not (os.path.exists('OptimizerParameters.json')):
            Nesterov = False
        else:
            Nesterov = OptParam.get('Nesterov')
        optimizerVal = callSGD(float(lrDict), float(Momentum), Nesterov)

    if(optimizerDict == "Nadam"):
        if (OptParam.get('Beta1') == "(float)") or (OptParam.get('Beta1') == "True/False") or (
                OptParam.get('Beta1') == "") or not (os.path.exists('OptimizerParameters.json')):
            Beta1 = 0.9
        else:
            Beta1 = OptParam.get('Beta1')
        if (OptParam.get('Beta2') == "(float)") or (OptParam.get('Beta2') == "True/False") or (
                OptParam.get('Beta2') == "") or not (os.path.exists('OptimizerParameters.json')):
            Beta2 = 0.999
        else:
            Beta2 = OptParam.get('Beta2')
        if (OptParam.get('Epsilon') == "(float)") or (OptParam.get('Epsilon') == "True/False") or (
                OptParam.get('Epsilon') == "") or not (os.path.exists('OptimizerParameters.json')):
            Epsilon = 1e-7
        else:
            Epsilon = OptParam.get('Epsilon')
        optimizerVal = callNadam(float(lrDict), float(Beta1), float(Beta2), float(Epsilon))

    if(optimizerDict == "Adamax"):
        if (OptParam.get('Beta1') == "(float)") or (OptParam.get('Beta1') == "True/False") or (
                OptParam.get('Beta1') == "") or not (os.path.exists('OptimizerParameters.json')):
            Beta1 = 0.9
        else:
            Beta1 = OptParam.get('Beta1')
        if (OptParam.get('Beta2') == "(float)") or (OptParam.get('Beta2') == "True/False") or (
                OptParam.get('Beta2') == "") or not (os.path.exists('OptimizerParameters.json')):
            Beta2 = 0.999
        else:
            Beta2 = OptParam.get('Beta2')
        if (OptParam.get('Epsilon') == "(float)") or (OptParam.get('Epsilon') == "True/False") or (
                OptParam.get('Epsilon') == "") or not (os.path.exists('OptimizerParameters.json')):
            Epsilon = 1e-7
        else:
            Epsilon = OptParam.get('Epsilon')
        optimizerVal = callAdamax(float(lrDict), float(Beta1), float(Beta2), float(Epsilon))

    #check and call the selected lss and insert parameter
    if(lossDict == "dice_loss"):
        lossVal = dice_loss
    if(lossDict == "balanced_cross_entropy"):
        lossVal = balanced_cross_entropy
    if(lossDict == "weighted_cross_entropy"):
        lossVal = weighted_cross_entropy
    if(lossDict == "intersection_over_union"):
        lossVal = intersection_over_union
    if(lossDict == "tversky_loss"):
        lossVal = tversky_loss
    if(lossDict == "lovasz_softmax"):
        lossVal = lovasz_softmax

    model.compile(optimizer=optimizerVal, loss=lossVal, metrics=[lossVal,losses.binary_crossentropy])
    model.summary()

    print('creating data generators')
    ng = NiftiGenerator()
    ng.initialize('data_train')

    print('creating callbacks')
    history = History()
    modelCheckpoint = ModelCheckpoint('best_weights.h5', monitor='loss', save_best_only=True)
    #tblogdir = 'tblogs/{}'.format(time())
    #tensorboard = TensorBoard(log_dir=tblogdir)

    print('fitting model')

    epochs = MedML.get('Epochs')
    batch_size = MedML.get('Batch size')
    #steps = MedML.get('Steps')
    sliceSamples = MedML.get('Slice samples')
    numOfClasses = MedML.get('NumOfSegClasses')
    useMultiProcessing = MedML.get('Use Multiprocessing')
    Workers = MedML.get('Workers')
    MaxQueueSize = MedML.get('Max queue size')

    steps_per_epoch = 1000 // batch_size # should really be number of total samples in the dataset divided by batch size
    model.fit( ng.generate(img_size=(64,64),slice_samples=int(sliceSamples),batch_size=batch_size,num_classes=int(numOfClasses)),
               epochs=epochs, steps_per_epoch=steps_per_epoch,
               use_multiprocessing=useMultiProcessing, workers=int(Workers), max_queue_size=int(MaxQueueSize),
               callbacks=[history, modelCheckpoint] )

    model.save('model.h5')

    print('done')

if __name__ == "__main__": 
    execute()