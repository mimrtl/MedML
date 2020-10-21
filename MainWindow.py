#!/usr/bin/env Python3
import json
import ntpath
import os
from glob import glob

import PySimpleGUI as sg

sg.ChangeLookAndFeel('DarkTeal3')

tab1_layout = [
    [sg.T('Validation Split:'), sg.InputText('(float) 0.0-1.0', key='IN1', size=(25, 1), enable_events=True)],

    [sg.T('Augmentation:'), sg.Checkbox('Option 1', size=(10, 1), key='AugOption1'),
     sg.Checkbox('Option 2', default=True, key='AugOption2')],
    [sg.T('Cross Validation Runs:'),
     sg.InputText('Input integer', key='CrossValidationRuns', size=(25, 1), enable_events=True)],
    [sg.T('Number of segmentation classes:'),
     sg.Spin([i for i in range(1, 1000)], initial_value=1, size=(5, 1), key='NumOfSegClasses')],
    [sg.T('Slice samples:'),
        sg.DropDown(('1', '3', '5', '7', '9'), default_value='1', key='slice_samples', size=(5, 1))]
]

tab2_layout = [[sg.T('UNet Type:'), sg.DropDown(('2D', '2.5D', '3D'), default_value='2D', change_submits=True,
                                                key='UNet', size=(5, 1))],
               [sg.T('X:'), sg.InputText('Input integer', key='x', size=(25, 1), enable_events=True), sg.T('Y:'), sg.InputText('Input integer', key='y', size=(25, 1), enable_events=True)],
               [sg.T('Starting filters:'),
                sg.Spin([i for i in range(8, 512)], initial_value=32, size=(5, 1), key='StartFilter')],

               [sg.T('Filter increasing rate:'),
                sg.InputText('(float) 1.0-5.0', key='IN2', size=(25, 1), enable_events=True)],
               [sg.T('Depth:'), sg.InputText('Input integer', key='depth', size=(25, 1), enable_events=True)]
               ]
tab3_layout = [[sg.T('Epochs:'), sg.Spin([i for i in range(1, 5000)], initial_value=1, size=(5, 1), key='Epochs')],
               [sg.T('Batch size:'),
                sg.Spin([i for i in range(1, 500)], initial_value=1, size=(5, 1), key='BatchSize')],
               [sg.T('Optimizer:'),
                sg.DropDown(('Adam', 'RectifiedAdam', 'RMSprop', 'Adagrad', 'SGD', 'Nadam', 'Adamax'),
                            default_value='Adam', size=(20, 1), key='Optimizer'), sg.Button('Set Optimizer Parameters', key='OptimizerParams')],
               [sg.T('Learning rate:'),
                sg.InputText('(float) 0.000000001-1.0', key='IN4', size=(25, 1), enable_events=True)],
               [sg.T('Loss function:'), sg.DropDown(('dice_loss', 'balanced_cross_entropy', 'weighted_cross_entropy',
                                                     'intersection_over_union', 'tversky_loss', 'lovasz_softmax'),
                                                    default_value='dice_loss', size=(23, 1),
                                                    key='LossFunction')],
               [sg.T('Use Tensorboard:'), sg.Checkbox('On/Off', size=(10, 1), key='TensorOption')]
               ]

tab4_layout = [[sg.T('Activation function:'),
                sg.DropDown(('relu', 'leaky relu', 'sigmoid'), default_value='relu', size=(20, 1),
                            key='ActivationFunction')],
               [sg.T('Workers:', visible=False), sg.InputText(default_text=4, key='workers', size=(25, 1), visible=False, enable_events=True)],
               [sg.T('Max queue size:', visible=False), sg.InputText(default_text=8, visible=False, key='max_queue_size', size=(25, 1), enable_events=True)],
               [sg.T('Dropout rate:'), sg.Checkbox('On:0.5, Off:0', key='IN3', size=(10, 1), enable_events=True)],
               [sg.T('Batch Normalization:'), sg.Checkbox('On/Off', size=(10, 1), key='BatchNormalization')],
               [sg.T('Maxpool:'), sg.Checkbox('On/Off', size=(10, 1), key='maxpool')],
               [sg.T('Upconv:'), sg.Checkbox('On/Off', size=(10, 1), key='upconv')],
               [sg.T('Residual:'), sg.Checkbox('On/Off', size=(10, 1), key='residual')],
               [sg.T('Use multiprocessing:'), sg.Checkbox('On/Off', size=(10, 1), key='use_multiprocessing')]
               ]

frame_layout = [
    [sg.TabGroup([[sg.Tab('Data Set', tab1_layout), sg.Tab('Model Parameters', tab2_layout),
                   sg.Tab('Model Options', tab4_layout), sg.Tab('Training', tab3_layout)]])]
]

layout = [
    [sg.Text('Select Folder', size=(35, 1))],
    [sg.Text('Your Folder', size=(15, 1), auto_size_text=False, justification='right'),
     sg.InputText('Default Folder', key='inputFolder', enable_events=True), sg.FolderBrowse(key='inputFolder2')],
    [sg.Button('Evaluate', key='Evaluate'), sg.Button('Clear', key='Clear')],

    [sg.Output(size=(57, 5), key='-OUTPUT-')],

    [sg.Text('_' * 80)],

    [sg.Frame('Select Parameters', frame_layout, font='Any 12', title_color='black')],

    [sg.Submit(key='Submit'), sg.Cancel()]
]

window = sg.Window('MedML', layout, default_element_size=(40, 1), resizable=True, finalize=True)

while True:
    event, values = window.read()


    # write to JSON
    def writeToJSONFile(path, fileName, data):
        filePathNameWExt = './' + path + '/' + fileName + '.json'
        with open(filePathNameWExt, 'w') as fp:
            json.dump(data, fp)

    #extract filename from path
    def path_leaf(path):
        head, tail = ntpath.split(path)
        return tail or ntpath.basename(head)

    pathJSON = './'
    fileName = 'MedML'
    fileNameOpt = 'OptimizerParameters'

    # JSON tags
    data = {'Validation Split': values['IN1'], 'Cross Validation Runs': values['CrossValidationRuns'],
            'Unet Type': values['UNet'],'Starting filter': values['StartFilter'],
            'Filter increasing rate': values['IN2'],
            'Dropout rate': values['IN3'], 'Activation function': values['ActivationFunction'],
            'Number of segmentation classes': values['NumOfSegClasses'],
            'Epochs': values['Epochs'], 'Batch size': values['BatchSize'], 'Optimizer': values['Optimizer'],
            'Loss function': values['LossFunction'], 'Learning rate': values['IN4'],
            'Depth': values['depth'], 'Slice samples': values['slice_samples'],
            'Workers': values['workers'], 'Max queue size': values['max_queue_size'], 'X': values['x'], 'Y': values['y']}

    # print(event, values)
    if event is None:  # always,  always give a way out!
        break

    #Integer boxes handeling
    # CrossValidation integer input box
    if event == 'CrossValidationRuns' and values['CrossValidationRuns'] and values['CrossValidationRuns'][-1] not in (
            '0123456789'):
        window['CrossValidationRuns'].update(values['CrossValidationRuns'][:-1])

    #SECTION 2 integer boxes of Model
    # x integer input box
    if event == 'x' and values['x'] and values['x'][-1] not in ('0123456789'):
        window['x'].update(values['x'][:-1])

    # y integer input box
    if event == 'y' and values['y'] and values['y'][-1] not in ('0123456789'):
        window['y'].update(values['y'][:-1])

    # out_ch integer input box
    if event == 'out_ch' and values['out_ch'] and values['out_ch'][-1] not in ('0123456789'):
        window['out_ch'].update(values['out_ch'][:-1])

    # depth integer input box
    if event == 'depth' and values['depth'] and values['depth'][-1] not in ('0123456789'):
        window['depth'].update(values['depth'][:-1])

    #SECTION3 integer boxes of Model

    if event == 'slice_samples' and values['slice_samples'] and values['slice_samples'][-1] not in ('0123456789'):
        window['slice_samples'].update(values['slice_samples'][:-1])

    if event == 'workers' and values['workers'] and values['workers'][-1] not in ('0123456789'):
        window['workers'].update(values['workers'][:-1])

    if event == 'max_queue_size' and values['max_queue_size'] and values['max_queue_size'][-1] not in ('0123456789'):
        window['max_queue_size'].update(values['max_queue_size'][:-1])

    # Float input boxes
    # if last character in input element is invalid, remove it
    if event == 'IN1' and values['IN1']:
        try:
            in_as_float = float(values['IN1'])
        except:
            if len(values['IN1']) == 1 and values['IN1'][0] == '-':
                continue
            window['IN1'].update(values['IN1'][:-1])

    # if last character in input element is invalid, remove it
    if event == 'IN2' and values['IN2']:
        try:
            in_as_float = float(values['IN2'])
        except:
            if len(values['IN2']) == 1 and values['IN2'][0] == '-':
                continue
            window['IN2'].update(values['IN2'][:-1])

    # if last character in input element is invalid, remove it
    if event == 'IN3' and values['IN3']:
        try:
            in_as_float = float(values['IN3'])
        except:
            if len(values['IN3']) == 1 and values['IN3'][0] == '-':
                continue
            window['IN3'].update(values['IN3'][:-1])

    # if last character in input element is invalid, remove it
    if event == 'IN4' and values['IN4']:
        try:
            in_as_float = float(values['IN4'])
        except:
            if len(values['IN4']) == 1 and values['IN4'][0] == '-':
                continue
            window['IN4'].update(values['IN4'][:-1])

    # Folder input handeling

    if event == 'Evaluate':
        label_files = glob(os.path.join(values['inputFolder'], '*_labels.nii.gz'), recursive=True) + glob(
            os.path.join(values['inputFolder'], '*_labels.nii'), recursive=True)
        for f in label_files:
            drive, filepath = os.path.splitdrive(f)
            path, filename = os.path.split(filepath)
            match_base = filename.replace('_labels.nii', '.nii')
            match_file = os.path.join(drive, path, match_base)

            # False -> No match
            if not os.path.exists(match_file):
                print('Label file ' + f + ' does not have a matching image file named ' + match_file)


    if event == 'Clear':
        window.FindElement('-OUTPUT-').Update('')

    #Create OptimizerParameters.json or load it
    #SetOptimizerParameter button click launch window event
    #Adam Parameter Window
    if (event == 'OptimizerParams') and (values['Optimizer'] == 'Adam'):
        optimizerData = {}

        #Check for existing submissions to populate fields with latest entries submitted
        if(os.path.exists('OptimizerParameters.json')):
            OptParam = json.load(open('OptimizerParameters.json'))
            optimizerData['Beta1'] = OptParam.get('Beta1')
            optimizerData['Beta2'] = OptParam.get('Beta2')
            optimizerData['Epsilon'] = OptParam.get('Epsilon')
            optimizerData['Amsgrad'] = OptParam.get('Amsgrad')

        else:
            optimizerData['Beta1'] = 0.9
            optimizerData['Beta2'] = 0.999
            optimizerData['Epsilon'] = 1e-7
            optimizerData['Amsgrad'] = False

        event, values = sg.Window('Set Optimizer Parameters',
                  [
                      [sg.T('beta_1:', key='beta1', visible=True), sg.In(default_text=optimizerData['Beta1'], size=(25, 1), key='IN5')],
                      [sg.T('beta_2:', key='beta2', visible=True),
                       sg.In(default_text=optimizerData['Beta2'], size=(25, 1), key='IN6')],
                      [sg.T('epsilon:', key='epsilon0', visible=True),
                       sg.In(default_text=optimizerData['Epsilon'], size=(25, 1), key='IN7')],
                      [sg.T('amsgrad:', key='amsgrad0', visible=True),
                       sg.In(default_text=optimizerData['Amsgrad'], size=(10, 1), key='Amsgrad0')],

                  [sg.B('Submit', key='OptSubmit')]]).read(close=False)


        #Submit and create json with the values
        if event == 'OptSubmit':
            optimizerData['Beta1'] = values['IN5']
            optimizerData['Beta2'] = values['IN6']
            optimizerData['Epsilon'] = values['IN7']
            optimizerData['Amsgrad'] = values['Amsgrad0']
            writeToJSONFile(pathJSON, fileNameOpt, optimizerData)


        #login_id = values['-ID-']
        #create dictionary to store the values
        #put write to json here

    #RectifiedAdam Parameter Window
    if (event == 'OptimizerParams') and (values['Optimizer'] == 'RectifiedAdam'):
        optimizerData = {}

        if(os.path.exists('OptimizerParameters.json')):
            OptParam = json.load(open('OptimizerParameters.json'))
            optimizerData['Beta1'] = OptParam.get('Beta1')
            optimizerData['Beta2'] = OptParam.get('Beta2')
            optimizerData['Epsilon'] = OptParam.get('Epsilon')
            optimizerData['Decay'] = OptParam.get('Decay')
            optimizerData['WeightDecay'] = OptParam.get('WeightDecay')
            optimizerData['Amsgrad'] = OptParam.get('Amsgrad')
            optimizerData['TotalSteps'] = OptParam.get('TotalSteps')
            optimizerData['WarmUpProportion'] = OptParam.get('WarmUpProportion')
            optimizerData['MinLr'] = OptParam.get('MinLr')

        else:
            optimizerData['Beta1'] = 0.9
            optimizerData['Beta2'] = 0.999
            optimizerData['Epsilon'] = None
            optimizerData['Decay'] = 0.
            optimizerData['WeightDecay'] = 0.
            optimizerData['Amsgrad'] = False
            optimizerData['TotalSteps'] = 0
            optimizerData['WarmUpProportion'] = 0.1
            optimizerData['MinLr'] = 0.


        event, values = sg.Window('Set Optimizer Parameters',
                  [
                      [sg.T('beta_1:', key='beta1', visible=True),
                       sg.In(default_text=optimizerData['Beta1'], size=(25, 1), key='IN5')],
                      [sg.T('beta_2:', key='beta2', visible=True),
                       sg.In(default_text=optimizerData['Beta2'], size=(25, 1), key='IN6')],
                      [sg.T('epsilon:', key='epsilon0', visible=True),
                       sg.In(default_text=optimizerData['Epsilon'], size=(25, 1), key='IN7')],
                      [sg.T('decay:', key='decay0', visible=True),
                       sg.In(default_text=optimizerData['Decay'], size=(25, 1), key='Decay0')],
                      [sg.T('weight_decay:', key='weightDecay', visible=True),
                       sg.In(default_text=optimizerData['WeightDecay'], size=(25, 1), key='IN8')],
                      [sg.T('amsgrad:', key='amsgrad0', visible=True),
                       sg.In(default_text=optimizerData['Amsgrad'], size=(10, 1), key='Amsgrad0')],
                      [sg.T('total_steps:', key='totalSteps', visible=True),
                       sg.In(default_text=optimizerData['TotalSteps'], size=(25, 1), key='IN11')],
                      [sg.T('warmup_proportion:', key='warmupProp', visible=True),
                       sg.In(default_text=optimizerData['WarmUpProportion'], size=(25, 1), key='IN12')],
                      [sg.T('min_lr:', key='minLr', visible=True),
                       sg.In(default_text=optimizerData['MinLr'], size=(25, 1), key='IN13')],

                      [sg.B('Submit', key='OptSubmit')]]).read(close=False)

        # Submit and create json with the values
        if event == 'OptSubmit':
            optimizerData['Beta1'] = values['IN5']
            optimizerData['Beta2'] = values['IN6']
            optimizerData['Epsilon'] = values['IN7']
            optimizerData['Decay'] = values['Decay0']
            optimizerData['WeightDecay'] = values['IN8']
            optimizerData['Amsgrad'] = values['Amsgrad0']
            optimizerData['TotalSteps'] = values['IN11']
            optimizerData['WarmUpProportion'] = values['IN12']
            optimizerData['MinLr'] = values['IN13']
            writeToJSONFile(pathJSON, fileNameOpt, optimizerData)

        #login_id = values['-ID-']
        #create dictionary to store the values
        #put write to json here

    #RMSprop Parameter Window
    if (event == 'OptimizerParams') and (values['Optimizer'] == 'RMSprop'):
        optimizerData = {}

        if(os.path.exists('OptimizerParameters.json')):
            OptParam = json.load(open('OptimizerParameters.json'))
            optimizerData['Rho'] = OptParam.get('Rho')
            optimizerData['Momentum'] = OptParam.get('Momentum')
            optimizerData['Epsilon'] = OptParam.get('Epsilon')
            optimizerData['Centered'] = OptParam.get('Centered')

        else:
            optimizerData['Rho'] = 0.9
            optimizerData['Momentum'] = 0.0
            optimizerData['Epsilon'] = 1e-07
            optimizerData['Centered'] = False

        event, values = sg.Window('Set Optimizer Parameters',
                [
                    [sg.T('rho:', key='rho0', visible=True),
                     sg.In(default_text=optimizerData['Rho'], size=(25, 1), key='IN16')],
                    [sg.T('momentum:', key='momentum0', visible=True),
                     sg.In(default_text=optimizerData['Momentum'], key='IN14', size=(25, 1))],
                    [sg.T('epsilon:', key='epsilon0', visible=True),
                     sg.In(default_text=optimizerData['Epsilon'], key='IN7', size=(25, 1))],
                    [sg.T('centered:', key='centered0', visible=True),
                     sg.In(default_text=optimizerData['Centered'], size=(10, 1), key='Centered0')],

                    [sg.B('Submit', key='OptSubmit')]]).read(close=False)

        # Submit and create json with the values
        if event == 'OptSubmit':
            optimizerData['Rho'] = values['IN16']
            optimizerData['Momentum'] = values['IN14']
            optimizerData['Epsilon'] = values['IN7']
            optimizerData['Centered'] = values['Centered0']
            writeToJSONFile(pathJSON, fileNameOpt, optimizerData)

            # login_id = values['-ID-']
            # create dictionary to store the values
            # put write to json here

    #Adagrad Parameter Window
    if (event == 'OptimizerParams') and (values['Optimizer'] == 'Adagrad'):
        optimizerData = {}

        if(os.path.exists('OptimizerParameters.json')):
            OptParam = json.load(open('OptimizerParameters.json'))
            optimizerData['InitialAccumVal'] = OptParam.get('InitialAccumVal')
            optimizerData['Epsilon'] = OptParam.get('Epsilon')

        else:
            optimizerData['InitialAccumVal'] = 0.1
            optimizerData['Epsilon'] = 1e-7

        event, values = sg.Window('Set Optimizer Parameters',
                [
                    [sg.T('initial_accumulator_value:', key='initialAccumVal', visible=True),
                     sg.In(default_text=optimizerData['InitialAccumVal'], key='IN15', size=(25, 1))],
                    [sg.T('epsilon:', key='epsilon0', visible=True),
                     sg.In(default_text=optimizerData['Epsilon'], key='IN7', size=(25, 1))],

                    [sg.B('Submit', key='OptSubmit')]]).read(close=False)

        # Submit and create json with the values
        if event == 'OptSubmit':
            optimizerData['InitialAccumVal'] = values['IN15']
            optimizerData['Epsilon'] = values['IN7']
            writeToJSONFile(pathJSON, fileNameOpt, optimizerData)

            # login_id = values['-ID-']
            # create dictionary to store the values
            # put write to json here

    #SGD Parameter Window
    if (event == 'OptimizerParams') and (values['Optimizer'] == 'SGD'):
        optimizerData = {}

        if(os.path.exists('OptimizerParameters.json')):
            OptParam = json.load(open('OptimizerParameters.json'))
            optimizerData['Momentum'] = OptParam.get('Momentum')
            optimizerData['Nesterov'] = OptParam.get('Nesterov')

        else:
            optimizerData['Momentum'] = 0.0
            optimizerData['Nesterov'] = False

        event, values = sg.Window('Set Optimizer Parameters',
                [
                    [sg.T('momentum:', key='momentum0', visible=True),
                     sg.In(default_text=optimizerData['Momentum'], key='IN14', size=(25, 1))],
                    [sg.T('nesterov:', key='nesterov0', visible=True),
                     sg.In(default_text=optimizerData['Nesterov'], size=(10, 1), key='Nesterov0')],

                    [sg.B('Submit', key='OptSubmit')]]).read(close=False)

        # Submit and create json with the values
        if event == 'OptSubmit':
            optimizerData['Momentum'] = values['IN14']
            optimizerData['Nesterov'] = values['Nesterov0']
            writeToJSONFile(pathJSON, fileNameOpt, optimizerData)

            # login_id = values['-ID-']
            # create dictionary to store the values
            # put write to json here

    #Nadam Parameter Window
    if (event == 'OptimizerParams') and (values['Optimizer'] == 'Nadam'):
        optimizerData = {}

        if(os.path.exists('OptimizerParameters.json')):
            OptParam = json.load(open('OptimizerParameters.json'))
            optimizerData['Beta1'] = OptParam.get('Beta1')
            optimizerData['Beta2'] = OptParam.get('Beta2')
            optimizerData['Epsilon'] = OptParam.get('Epsilon')

        else:
            optimizerData['Beta1'] = 0.9
            optimizerData['Beta2'] = 0.999
            optimizerData['Epsilon'] = 1e-7

        event, values = sg.Window('Set Optimizer Parameters',
                [
                    [sg.T('beta_1:', key='beta1', visible=True),
                     sg.In(default_text=optimizerData['Beta1'], key='IN5', size=(25, 1))],
                    [sg.T('beta_2:', key='beta2', visible=True),
                     sg.In(default_text=optimizerData['Beta2'], key='IN6', size=(25, 1))],
                    [sg.T('epsilon:', key='epsilon0', visible=True),
                     sg.In(default_text=optimizerData['Epsilon'], key='IN7', size=(25, 1))],

                    [sg.B('Submit', key='OptSubmit')]]).read(close=False)

        # Submit and create json with the values
        if event == 'OptSubmit':
            optimizerData['Beta1'] = values['IN5']
            optimizerData['Beta2'] = values['IN6']
            optimizerData['Epsilon'] = values['IN7']
            writeToJSONFile(pathJSON, fileNameOpt, optimizerData)

            # login_id = values['-ID-']
            # create dictionary to store the values
            # put write to json here

    #Adamax Parameter Window
    if (event == 'OptimizerParams') and (values['Optimizer'] == 'Adamax'):
        optimizerData = {}

        if(os.path.exists('OptimizerParameters.json')):
            OptParam = json.load(open('OptimizerParameters.json'))
            optimizerData['Beta1'] = OptParam.get('Beta1')
            optimizerData['Beta2'] = OptParam.get('Beta2')
            optimizerData['Epsilon'] = OptParam.get('Epsilon')

        else:
            optimizerData['Beta1'] = 0.9
            optimizerData['Beta2'] = 0.999
            optimizerData['Epsilon'] = 1e-7

        event, values = sg.Window('Set Optimizer Parameters',
                                  [
                                      [sg.T('beta_1:', key='beta1', visible=True),
                                       sg.In(default_text=optimizerData['Beta1'], key='IN5', size=(25, 1))],
                                      [sg.T('beta_2:', key='beta2', visible=True),
                                       sg.In(default_text=optimizerData['Beta2'], key='IN6', size=(25, 1))],
                                      [sg.T('epsilon:', key='epsilon0', visible=True),
                                       sg.In(default_text=optimizerData['Epsilon'], key='IN7', size=(25, 1))],

                                      [sg.B('Submit', key='OptSubmit')]]).read(close=False)

        # Submit and create json with the values
        if event == 'OptSubmit':
            optimizerData['Beta1'] = values['IN5']
            optimizerData['Beta2'] = values['IN6']
            optimizerData['Epsilon'] = values['IN7']
            writeToJSONFile(pathJSON, fileNameOpt, optimizerData)

            # login_id = values['-ID-']
            # create dictionary to store the values
            # put write to json here

    if event == 'Submit':
        #store extracted data_train file name from path
        data['Folder name'] = path_leaf(values['inputFolder'])

        #check dropout rate
        if(values['IN3']==True):
            data['Dropout rate'] = 0.5
        else:
            data['Dropout rate'] = 0

        #check for learning rate
        if(values['IN4'] == '(float) 0.000000001-1.0') or (values['IN4'] == ""):
            if(values['Optimizer'] == "SGD"):
                data['Learning rate'] = 0.01
            else:
                data['Learning rate'] = 0.001
        # Add checkbox options to the data dictionary
        data['Augmentation: AugOpt1'] = values['AugOption1']
        data['Augmentation: AugOpt2'] = values['AugOption2']
        data['Batch Normalization'] = values['BatchNormalization']
        data['Use Tensorboard'] = values['TensorOption']

        #Model Section 2
        data['Maxpool'] = values['maxpool']
        data['Upconv'] = values['upconv']
        data['Residual'] = values['residual']

        #Model Section 3
        data['Use Multiprocessing'] = values['use_multiprocessing']

        # Write to JSON file, Create MedML.json
        writeToJSONFile(pathJSON, fileName, data)
