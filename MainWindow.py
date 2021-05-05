#!/usr/bin/env Python3
import json
import ntpath
import os
from glob import glob

import PySimpleGUI as sg

sg.ChangeLookAndFeel('DarkTeal3')

# write to JSON
def writeToJSONFile(path, fileName, data):
    filePathNameWExt = './' + path + '/' + fileName + '.json'
    with open(filePathNameWExt, 'w') as fp:
        json.dump(data, fp)

def collapse(layout, key):
    return sg.pin(sg.Column(layout, key=key))

# Try to open existing json file to display UI parameter input
try:
    f = open('MedML.json')
    MedML = json.load(f)

# otherwise instantiate a parameters json file to have default parameters to display in UI
except:
    pathJSON = './'
    fileName = 'MedML'

    medMLdata = {}

    medMLdata["Validation Split"] = "(float) 0.0-1.0"
    medMLdata["Cross Validation Runs"] = "Input integer"
    medMLdata["Unet Type"] = "2D"
    medMLdata["Starting filter"] = 32
    medMLdata["Filter increasing rate"] = "2"
    medMLdata["Dropout rate"] = 0.5
    medMLdata["Activation function"] = "relu"
    medMLdata["Number of segmentation classes"] = 1
    medMLdata["Epochs"] = 5
    medMLdata["Batch size"] = 16
    medMLdata["Optimizer"] = "Adam"
    medMLdata["Loss function"] = "dice_loss"
    medMLdata["Learning rate"] = "0.001"
    medMLdata["Depth"] = "4"
    medMLdata["Slice samples"] = "1"
    medMLdata["Workers"] = "4"
    medMLdata["Max queue size"] = "8"
    medMLdata["X"] = "64"
    medMLdata["Y"] = "64"

    medMLdata["Folder name"] = "Default Folder"
    medMLdata["Image Folder name"] = "Image Folder"
    medMLdata["Label Folder name"] = "Label Folder"

    medMLdata["Dropout rate boolean"] = False
    medMLdata["Batch Normalization"] = True
    medMLdata["Use Tensorboard"] = False
    medMLdata["Maxpool"] = True
    medMLdata["Upconv"] = True
    medMLdata["Residual"] = False
    medMLdata["Use Multiprocessing"] = True

    medMLdata['Augmode'] = 'reflect'
    medMLdata['Augseed'] = 813
    medMLdata['Addnoise'] = 0
    medMLdata['Hflips'] = True
    medMLdata['Vflips'] = True
    medMLdata['Rotations'] = 0
    medMLdata['Scalings'] = 0
    medMLdata['Shears'] = 0
    medMLdata['Translations'] = 0

    #write to file MedML.json and then open it
    writeToJSONFile(pathJSON, fileName, medMLdata)
    f = open('MedML.json')
    MedML = json.load(f)


#Tab UI element component layouts

tab1_layout = [
    [sg.T('Validation Split:'), sg.InputText(default_text=MedML.get("Validation Split"), key='IN1', size=(25, 1), enable_events=True)],
    [sg.T('Cross Validation Runs:'),
     sg.InputText(default_text=MedML.get("Cross Validation Runs"), key='CrossValidationRuns', size=(25, 1), enable_events=True)],
    [sg.T('Number of segmentation classes:'),
     sg.Spin([i for i in range(1, 1000)], initial_value=MedML.get("Number of segmentation classes"), size=(5, 1), key='NumOfSegClasses')],
    [sg.T('Slice samples:'),
        sg.DropDown(('1', '3', '5', '7', '9'), default_value=MedML.get("Slice samples"), key='slice_samples', size=(5, 1))]
]

tab2_layout = [[sg.T('UNet Type:'), sg.DropDown(('2D', '2.5D', '3D'), default_value=MedML.get("Unet Type"), enable_events=True,
                                                key='UNet', size=(5, 1))],
               [sg.T('X:'), sg.InputText(default_text=MedML.get("X"), key='x', size=(25, 1), enable_events=True), sg.T('Y:'), sg.InputText(default_text=MedML.get("Y"), key='y', size=(25, 1), enable_events=True)],
               [sg.T('Starting filters:'),
                sg.Spin([i for i in range(8, 512)], initial_value=MedML.get("Starting filter"), size=(5, 1), key='StartFilter')],

               [sg.T('Filter increasing rate:'),
                sg.InputText(default_text=MedML.get("Filter increasing rate"), key='IN2', size=(25, 1), enable_events=True)],
               [sg.T('Depth:'), sg.InputText(default_text=MedML.get("Depth"), key='depth', size=(25, 1), enable_events=True)]
               ]
tab3_layout = [[sg.T('Epochs:'), sg.Spin([i for i in range(1, 5000)], initial_value=MedML.get("Epochs"), size=(5, 1), key='Epochs')],
               [sg.T('Batch size:'),
                sg.Spin([i for i in range(1, 500)], initial_value=MedML.get("Batch size"), size=(5, 1), key='BatchSize')],
               [sg.T('Optimizer:', enable_events=True),
                sg.DropDown(('Adam', 'RectifiedAdam', 'RMSprop', 'Adagrad', 'SGD', 'Nadam', 'Adamax'),
                            default_value=MedML.get("Optimizer"), size=(20, 1), key='Optimizer', enable_events=True), sg.Button('Set Optimizer Parameters', key='OptimizerParams')],
               [sg.T('Learning rate:'),
                sg.InputText(default_text=MedML.get("Learning rate"), key='IN4', size=(25, 1), enable_events=True)],
               [sg.T('Loss function:'), sg.DropDown(('dice_loss', 'balanced_cross_entropy', 'weighted_cross_entropy',
                                                     'intersection_over_union', 'tversky_loss', 'lovasz_softmax'),
                                                    default_value=MedML.get("Loss function"), size=(23, 1),
                                                    key='LossFunction')],
               [sg.T('Use Tensorboard:'), sg.Checkbox('On/Off', default=MedML.get("Use Tensorboard"), size=(10, 1), key='TensorOption')]
               ]

tab4_layout = [[sg.T('Activation function:'),
                sg.DropDown(('relu', 'leaky relu', 'sigmoid'), default_value=MedML.get("Activation function"), size=(20, 1),
                            key='ActivationFunction')],
               [sg.T('Workers:', visible=False), sg.InputText(default_text=4, key='workers', size=(25, 1), visible=False, enable_events=True)],
               [sg.T('Max queue size:', visible=False), sg.InputText(default_text=8, visible=False, key='max_queue_size', size=(25, 1), enable_events=True)],
               [sg.T('Dropout rate:'), sg.Checkbox('On:0.5, Off:0', default=MedML.get("Dropout rate boolean"), key='IN3', size=(10, 1), enable_events=True)],
               [sg.T('Batch Normalization:'), sg.Checkbox('On/Off', default=MedML.get("Batch Normalization"), size=(10, 1), key='BatchNormalization')],
               [sg.T('Maxpool:'), sg.Checkbox('On/Off', default=MedML.get("Maxpool"), size=(10, 1), key='maxpool')],
               [sg.T('Upconv:'), sg.Checkbox('On/Off', default=MedML.get("Upconv"), size=(10, 1), key='upconv')],
               [sg.T('Residual:'), sg.Checkbox('On/Off', default=MedML.get("Residual"), size=(10, 1), key='residual')],
               [sg.T('Use multiprocessing:'), sg.Checkbox('On/Off', default=MedML.get("Use Multiprocessing"), size=(10, 1), key='use_multiprocessing')]
               ]

tab5_layout = [[sg.T('Aug Mode:'),
                sg.DropDown(('mirror', 'nearest', 'reflect', 'wrap'), default_value=MedML.get("Augmode"), size=(20, 1),
                            key='Augmode')],
               [sg.T('Aug Seed:'), sg.InputText(default_text=MedML.get("Augseed"), key='Augseed', size=(25, 1), enable_events=True),
               sg.T('Add Noise:'), sg.InputText(default_text=MedML.get("Addnoise"), key='Addnoise', size=(25, 1), enable_events=True)],

               [sg.T('Random Horizontal Flips:'), sg.Checkbox('On/Off', default=MedML.get("Hflips"), size=(10, 1), key='hflips')],
               [sg.T('Random Vertical Flips:'), sg.Checkbox('On/Off', default=MedML.get("Vflips"), size=(10, 1), key='vflips')],

               [sg.T('Rotations Angle'), sg.InputText(default_text=MedML.get("Rotations"), key='rotations', size=(25, 1), enable_events=True),
               sg.T('Scalings Range'), sg.InputText(default_text=MedML.get("Scalings"), key='scalings', size=(25, 1), enable_events=True)],

               [sg.T('Shears Angle'), sg.InputText(default_text=MedML.get("Shears"), key='shears', size=(25, 1), enable_events=True),
               sg.T('Translations Pixels'), sg.InputText(default_text=MedML.get("Translations"), key='translations', size=(25, 1), enable_events=True)]

]

#Frame grouping for tabs
frame_layout = [
    [sg.TabGroup([[sg.Tab('Data Set', tab1_layout), sg.Tab('Augmentation Options', tab5_layout), sg.Tab('Model Parameters', tab2_layout),
                   sg.Tab('Model Options', tab4_layout), sg.Tab('Training', tab3_layout)]])]
]

inputOption1 = [
    [sg.Text('Training Data:', font='12')],
    [sg.Text('Input Folder (Images & Labels)', justification='right'),
     sg.InputText(default_text=MedML.get("Folder name"), key='inputFolder', enable_events=True), sg.FolderBrowse(target='inputFolder')],]

inputOption2 = [
    [sg.Text('Images Folder', size=(15, 1), auto_size_text=False, justification='right'),
     sg.InputText(default_text="Images Folder", key='inputFolder3', enable_events=True),
     sg.FolderBrowse(key='inputFolder4')],
    [sg.Text('Labels Folder', size=(15, 1), auto_size_text=False, justification='right'),
     sg.InputText(default_text="Labels Folder", key='inputFolder5', enable_events=True),
     sg.FolderBrowse(key='inputFolder6')],
    [sg.Text(" " * 40), sg.Button('Load input folders', key='LoadInputFolders')],
]

layout = [
    #Hide show single or double image/label folder input options
    [sg.Checkbox('Hide Single Folder Input', enable_events=True, default=False, key='-OPEN SEC1-CHECKBOX'), sg.Checkbox('Hide Multi Folder Input', enable_events=True, default=False, key='-OPEN SEC2-CHECKBOX')],
    [collapse(inputOption1, '-SEC1-')],
    [collapse(inputOption2, '-SEC2-')],

    [sg.Text('_' * 80)],

    [sg.Frame('Select Parameters', frame_layout, font='Any 12', title_color='black')],
    [sg.Input(key='_FILEBROWSE_', enable_events=True, visible=False)],
    [sg.Input(key='_FILESAVEAS_', enable_events=True, visible=False)],
     [sg.Button('Save', key='Save'), sg.FileSaveAs('Save As', key='Save As', target='_FILESAVEAS_'), sg.FileBrowse('Load Paramaters', file_types=(("Json Files", "*.json"),), target='_FILEBROWSE_')]
]

window = sg.Window('MedML', layout, default_element_size=(40, 1), resizable=True, finalize=True)

opened1, opened2 = True, True

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
            'Workers': values['workers'], 'Max queue size': values['max_queue_size'], 'X': values['x'], 'Y': values['y'],
            'Augmode': values['Augmode'], 'Augseed': values['Augseed'], 'Addnoise': values['Addnoise'], 'Hflips': values['hflips'],
            'Vflips': values['vflips'], 'Rotations': values['rotations'], 'Scalings': values['scalings'], 'Shears': values['shears'],
            'Translations': values['translations']}

    # print(event, values)
    if event is None:  # always,  always give a way out!
        break

    #manage inputOption, training data, section
    if event.startswith('-OPEN SEC1-'):
        opened1 = not opened1
        window['-OPEN SEC2-CHECKBOX'].update(not opened1)
        window['-SEC1-'].update(visible=opened1)

    if event.startswith('-OPEN SEC2-'):
        opened2 = not opened2
        window['-OPEN SEC2-CHECKBOX'].update(not opened2)
        window['-SEC2-'].update(visible=opened2)

    #load parameters
    # load values into the window input spaces

    if event == '_FILEBROWSE_':

        #Any loading is dependent on selection made which will then be in path text in a hidden text box
        #check for any user selection of load file in the hidden text path box
        while(values['_FILEBROWSE_'] == ""):
            pass

        if(values['_FILEBROWSE_'] != ""):

            #once have the load params file path load
            p = open(path_leaf(values['_FILEBROWSE_']))
            loadedParamsFile = json.load(p)


            #update the window with new loaded values to MedML
            window['IN1'].update(loadedParamsFile.get("Validation Split"))
            window['CrossValidationRuns'].update(loadedParamsFile.get("Cross Validation Runs"))
            window['UNet'].update(loadedParamsFile.get("Unet Type"))
            window['StartFilter'].update(loadedParamsFile.get("Starting filter"))
            window['IN2'].update(loadedParamsFile.get("Filter increasing rate"))

            #This is determined by changed dropout rate boolean
            # window["Dropout rate"].update(loadedParamsFile.get("Dropout rate"))

            window['ActivationFunction'].update(loadedParamsFile.get("Activation function"))
            window['NumOfSegClasses'].update(loadedParamsFile.get("Number of segmentation classes"))
            window['Epochs'].update(loadedParamsFile.get("Epochs"))
            window['BatchSize'].update(loadedParamsFile.get("Batch size"))
            window['Optimizer'].update(loadedParamsFile.get("Optimizer"))
            window['LossFunction'].update(loadedParamsFile.get("Loss function"))
            window['IN4'].update(loadedParamsFile.get("Learning rate"))
            window['depth'].update(loadedParamsFile.get("Depth"))
            window['slice_samples'].update(loadedParamsFile.get("Slice samples"))
            window['workers'].update(loadedParamsFile.get("Workers"))
            window['max_queue_size'].update(loadedParamsFile.get("Max queue size"))
            window['x'].update(loadedParamsFile.get("X"))
            window['y'].update(loadedParamsFile.get("Y"))
            window['_FILEBROWSE_'].update(loadedParamsFile.get("Folder name"))
            window['IN3'].update(loadedParamsFile.get("Dropout rate boolean"))
            window['BatchNormalization'].update(loadedParamsFile.get("Batch Normalization"))
            window['TensorOption'].update(loadedParamsFile.get("Use Tensorboard"))
            window['maxpool'].update(loadedParamsFile.get("Maxpool"))
            window['upconv'].update(loadedParamsFile.get("Upconv"))
            window['residual'].update(loadedParamsFile.get("Residual"))
            window['use_multiprocessing'].update(loadedParamsFile.get("Use Multiprocessing"))

            sg.popup_ok('Parameters Loaded')


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


    #Set mode for single folder with image and labels
    if event == 'inputFolder':

        data["inputFolderMode"] = "one"

        writeToJSONFile(pathJSON, fileName, data)

    # Set mode and Store folder input for split image and label folders

    if event == 'LoadInputFolders':

        # Set mode for two folder input
        data["inputFolderMode"] = "two"

        writeToJSONFile(pathJSON, fileName, data)

        sg.popup_ok('Image and Label folders set')


    #Create OptimizerParameters.json or load it
    #SetOptimizerParameter button click launch window event
    #Adam Parameter Window
    if (event == 'OptimizerParams') and (values['Optimizer'] == 'Adam'):
        optimizerData = {}

        optimizerData['Optimizer'] = 'Adam'

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

            sg.popup_ok('Optimizer Parameters set (Adam)')


        #login_id = values['-ID-']
        #create dictionary to store the values
        #put write to json here

    #RectifiedAdam Parameter Window
    if (event == 'OptimizerParams') and (values['Optimizer'] == 'RectifiedAdam'):
        optimizerData = {}

        optimizerData['Optimizer'] = 'RectifiedAdam'

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

            sg.popup_ok('Optimizer Parameters set (RectifiedAdam)')

        #login_id = values['-ID-']
        #create dictionary to store the values
        #put write to json here

    #RMSprop Parameter Window
    if (event == 'OptimizerParams') and (values['Optimizer'] == 'RMSprop'):
        optimizerData = {}

        optimizerData['Optimizer'] = 'RMSprop'

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

            sg.popup_ok('Optimizer Parameters set (RMSprop)')

            # login_id = values['-ID-']
            # create dictionary to store the values
            # put write to json here

    #Adagrad Parameter Window
    if (event == 'OptimizerParams') and (values['Optimizer'] == 'Adagrad'):
        optimizerData = {}

        optimizerData['Optimizer'] = 'Adagrad'

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

            sg.popup_ok('Optimizer Parameters set (Adagrad)')

            # login_id = values['-ID-']
            # create dictionary to store the values
            # put write to json here

    #SGD Parameter Window
    if (event == 'OptimizerParams') and (values['Optimizer'] == 'SGD'):
        optimizerData = {}

        optimizerData['Optimizer'] = 'SGD'

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

            sg.popup_ok('Optimizer Parameters set (SGD)')

            # login_id = values['-ID-']
            # create dictionary to store the values
            # put write to json here

    #Nadam Parameter Window
    if (event == 'OptimizerParams') and (values['Optimizer'] == 'Nadam'):
        optimizerData = {}

        optimizerData['Optimizer'] = 'Nadam'

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

            sg.popup_ok('Optimizer Parameters set (Nadam)')

            # login_id = values['-ID-']
            # create dictionary to store the values
            # put write to json here

    #Adamax Parameter Window
    if (event == 'OptimizerParams') and (values['Optimizer'] == 'Adamax'):
        optimizerData = {}

        optimizerData['Optimizer'] = 'Adamax'

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

            sg.popup_ok('Optimizer Parameters set (Adamax)')

            # login_id = values['-ID-']
            # create dictionary to store the values
            # put write to json here

    # SAVE

    if event == 'Save':
        #reload MedML.json to check for folder mode status
        f = open('MedML.json')
        MedML = json.load(f)

        #Based on folder mode status read, set the folder, or images/label folders
        if MedML.get("inputFolderMode") == "one":

            data["inputFolderMode"] = "one"

            # store extracted data_train file name from path
            data['Folder name'] = path_leaf(values['inputFolder'])

        if MedML.get("inputFolderMode") == "two":

            data["inputFolderMode"] = "two"

            # Images Folder
            data["Image Folder name"] = path_leaf(values['inputFolder3'])

            # Labels Folder
            data["Label Folder name"] = path_leaf(values['inputFolder5'])

        #check dropout rate
        if(values['IN3']==True):
            data['Dropout rate'] = 0.5
            data['Dropout rate boolean'] = True
        else:
            data['Dropout rate'] = 0
            data['Dropout rate boolean'] = False

        #check for learning rate
        if(values['IN4'] == '(float) 0.000000001-1.0') or (values['IN4'] == ""):
            if(values['Optimizer'] == "SGD"):
                data['Learning rate'] = 0.01
            else:
                data['Learning rate'] = 0.001

        # Augmentation Options
        data['Augmode'] = values['Augmode']
        data['Augseed'] = values['Augseed']
        data['Addnoise'] = values['Addnoise']
        data['Hflips'] = values['hflips']
        data['Vflips'] = values['vflips']
        data['Rotations'] = values['rotations']
        data['Scalings'] = values['scalings']
        data['Shears'] = values['shears']
        data['Translations'] = values['translations']

        # Add checkbox options to the data dictionary
        # Model Options
        data['Batch Normalization'] = values['BatchNormalization']
        data['Use Tensorboard'] = values['TensorOption']
        data['Maxpool'] = values['maxpool']
        data['Upconv'] = values['upconv']
        data['Residual'] = values['residual']
        data['Use Multiprocessing'] = values['use_multiprocessing']

        # Add on or create OptimizerParameters.json and put inputs dictionary into MedML

        # try to read OptimizerParameters.json if it was not created create it and put in filler values
        try:

            g = open('OptimizerParameters.json')
            OptParam = json.load(g)
            data['OptParam'] = OptParam

            # optimizer value changed so we need to provide values consisting from new optimizer selected from gui
            # otherwise when reading from the MedML.json file it will be missing values from the un-updated optimizer dict

            if(values['Optimizer'] != OptParam.get("Optimizer")):
                pathJSON = './'
                fileNameOpt = 'OptimizerParameters'

                # create empty dict of opt data
                optimizerData = {}

                # fill with filler values relative to the Optimizer selection from MedML.json
                if (values['Optimizer'] == "Adam"):
                    optimizerData['Optimizer'] = "Adam"

                    optimizerData['Beta1'] = 0.9
                    optimizerData['Beta2'] = 0.999
                    optimizerData['Epsilon'] = 1e-7
                    optimizerData['Amsgrad'] = False

                if (values['Optimizer'] == "RectifiedAdam"):
                    optimizerData['Optimizer'] = "RectifiedAdam"

                    optimizerData['Beta1'] = 0.9
                    optimizerData['Beta2'] = 0.999
                    optimizerData['Epsilon'] = None
                    optimizerData['Decay'] = 0.
                    optimizerData['WeightDecay'] = 0.
                    optimizerData['Amsgrad'] = False
                    optimizerData['TotalSteps'] = 0
                    optimizerData['WarmUpProportion'] = 0.1
                    optimizerData['MinLr'] = 0.

                if (values['Optimizer'] == "RMSprop"):
                    optimizerData['Optimizer'] = "RMSprop"

                    optimizerData['Rho'] = 0.9
                    optimizerData['Momentum'] = 0.0
                    optimizerData['Epsilon'] = 1e-07
                    optimizerData['Centered'] = False

                if (values['Optimizer'] == "Adagrad"):
                    optimizerData['Optimizer'] = "Adagrad"

                    optimizerData['InitialAccumVal'] = 0.1
                    optimizerData['Epsilon'] = 1e-7

                if (values['Optimizer'] == "SGD"):
                    optimizerData['Optimizer'] = "SGD"

                    optimizerData['Momentum'] = 0.0
                    optimizerData['Nesterov'] = False

                if (values['Optimizer'] == "Nadam"):
                    optimizerData['Optimizer'] = "Nadam"

                    optimizerData['Beta1'] = 0.9
                    optimizerData['Beta2'] = 0.999
                    optimizerData['Epsilon'] = 1e-7

                if (values['Optimizer'] == "Adamax"):
                    optimizerData['Optimizer'] = "Adamax"

                    optimizerData['Beta1'] = 0.9
                    optimizerData['Beta2'] = 0.999
                    optimizerData['Epsilon'] = 1e-7

                # write to file OptimizerParameters.json and then open it
                writeToJSONFile(pathJSON, fileNameOpt, optimizerData)
                g = open('OptimizerParameters.json')
                OptParam = json.load(g)

                # Take loaded OptParam and put in data dictionary to be in MedML.json
                data['OptParam'] = OptParam


        except:

            pathJSON = './'
            fileNameOpt = 'OptimizerParameters'

            # create empty dict of opt data
            optimizerData = {}

            # fill with filler values relative to the Optimizer selection from MedML.json
            if (values['Optimizer'] == "Adam"):
                optimizerData['Optimizer'] = "Adam"

                optimizerData['Beta1'] = 0.9
                optimizerData['Beta2'] = 0.999
                optimizerData['Epsilon'] = 1e-7
                optimizerData['Amsgrad'] = False

            if (values['Optimizer'] == "RectifiedAdam"):
                optimizerData['Optimizer'] = "RectifiedAdam"

                optimizerData['Beta1'] = 0.9
                optimizerData['Beta2'] = 0.999
                optimizerData['Epsilon'] = None
                optimizerData['Decay'] = 0.
                optimizerData['WeightDecay'] = 0.
                optimizerData['Amsgrad'] = False
                optimizerData['TotalSteps'] = 0
                optimizerData['WarmUpProportion'] = 0.1
                optimizerData['MinLr'] = 0.

            if (values['Optimizer'] == "RMSprop"):
                optimizerData['Optimizer'] = "RMSprop"

                optimizerData['Rho'] = 0.9
                optimizerData['Momentum'] = 0.0
                optimizerData['Epsilon'] = 1e-07
                optimizerData['Centered'] = False

            if (values['Optimizer'] == "Adagrad"):
                optimizerData['Optimizer'] = "Adagrad"

                optimizerData['InitialAccumVal'] = 0.1
                optimizerData['Epsilon'] = 1e-7

            if (values['Optimizer'] == "SGD"):
                optimizerData['Optimizer'] = "SGD"

                optimizerData['Momentum'] = 0.0
                optimizerData['Nesterov'] = False

            if (values['Optimizer'] == "Nadam"):
                optimizerData['Optimizer'] = "Nadam"

                optimizerData['Beta1'] = 0.9
                optimizerData['Beta2'] = 0.999
                optimizerData['Epsilon'] = 1e-7

            if (values['Optimizer'] == "Adamax"):
                optimizerData['Optimizer'] = "Adamax"

                optimizerData['Beta1'] = 0.9
                optimizerData['Beta2'] = 0.999
                optimizerData['Epsilon'] = 1e-7

            # write to file OptimizerParameters.json and then open it
            writeToJSONFile(pathJSON, fileNameOpt, optimizerData)
            g = open('OptimizerParameters.json')
            OptParam = json.load(g)

            # Take loaded OptParam and put in data dictionary to be in MedML.json
            data['OptParam'] = OptParam


        # Write to JSON file, Create MedML.json
        writeToJSONFile(pathJSON, fileName, data)

        sg.popup_ok('MedML parameters saved')

    # SAVE AS

    if event == '_FILESAVEAS_':

        # reload MedML.json to check for folder mode status
        f = open('MedML.json')
        MedML = json.load(f)

        # Based on folder mode status read, set the folder, or images/label folders
        if MedML.get("inputFolderMode") == "one":
            data["inputFolderMode"] = "one"

            # store extracted data_train file name from path
            data['Folder name'] = path_leaf(values['inputFolder'])

        if MedML.get("inputFolderMode") == "two":
            data["inputFolderMode"] = "two"

            # Images Folder
            data["Image Folder name"] = path_leaf(values['inputFolder3'])

            # Labels Folder
            data["Label Folder name"] = path_leaf(values['inputFolder5'])

        #check dropout rate
        if(values['IN3']==True):
            data['Dropout rate'] = 0.5
            data['Dropout rate boolean'] = True
        else:
            data['Dropout rate'] = 0
            data['Dropout rate boolean'] = False

        #check for learning rate
        if(values['IN4'] == '(float) 0.000000001-1.0') or (values['IN4'] == ""):
            if(values['Optimizer'] == "SGD"):
                data['Learning rate'] = 0.01
            else:
                data['Learning rate'] = 0.001

        # Augmentation Options
        data['Augmode'] = values['Augmode']
        data['Augseed'] = values['Augseed']
        data['Addnoise'] = values['Addnoise']
        data['Hflips'] = values['hflips']
        data['Vflips'] = values['vflips']
        data['Rotations'] = values['rotations']
        data['Scalings'] = values['scalings']
        data['Shears'] = values['shears']
        data['Translations'] = values['translations']

        # Add checkbox options to the data dictionary
        # Model Options
        data['Batch Normalization'] = values['BatchNormalization']
        data['Use Tensorboard'] = values['TensorOption']
        data['Maxpool'] = values['maxpool']
        data['Upconv'] = values['upconv']
        data['Residual'] = values['residual']
        data['Use Multiprocessing'] = values['use_multiprocessing']

        # Add on or create OptimizerParameters.json and put inputs dictionary into MedML

        # try to read OptimizerParameters.json if it was not created create it and put in filler values
        try:

            g = open('OptimizerParameters.json')
            OptParam = json.load(g)
            data['OptParam'] = OptParam

            # optimizer value changed so we need to provide values consisting from new optimizer selected from gui
            # otherwise when reading from the MedML.json file it will be missing values from the un-updated optimizer dict

            if(values['Optimizer'] != OptParam.get("Optimizer")):
                pathJSON = './'
                fileNameOpt = 'OptimizerParameters'

                # create empty dict of opt data
                optimizerData = {}

                # fill with filler values relative to the Optimizer selection from MedML.json
                if (values['Optimizer'] == "Adam"):
                    optimizerData['Optimizer'] = "Adam"

                    optimizerData['Beta1'] = 0.9
                    optimizerData['Beta2'] = 0.999
                    optimizerData['Epsilon'] = 1e-7
                    optimizerData['Amsgrad'] = False

                if (values['Optimizer'] == "RectifiedAdam"):
                    optimizerData['Optimizer'] = "RectifiedAdam"

                    optimizerData['Beta1'] = 0.9
                    optimizerData['Beta2'] = 0.999
                    optimizerData['Epsilon'] = None
                    optimizerData['Decay'] = 0.
                    optimizerData['WeightDecay'] = 0.
                    optimizerData['Amsgrad'] = False
                    optimizerData['TotalSteps'] = 0
                    optimizerData['WarmUpProportion'] = 0.1
                    optimizerData['MinLr'] = 0.

                if (values['Optimizer'] == "RMSprop"):
                    optimizerData['Optimizer'] = "RMSprop"

                    optimizerData['Rho'] = 0.9
                    optimizerData['Momentum'] = 0.0
                    optimizerData['Epsilon'] = 1e-07
                    optimizerData['Centered'] = False

                if (values['Optimizer'] == "Adagrad"):
                    optimizerData['Optimizer'] = "Adagrad"

                    optimizerData['InitialAccumVal'] = 0.1
                    optimizerData['Epsilon'] = 1e-7

                if (values['Optimizer'] == "SGD"):
                    optimizerData['Optimizer'] = "SGD"

                    optimizerData['Momentum'] = 0.0
                    optimizerData['Nesterov'] = False

                if (values['Optimizer'] == "Nadam"):
                    optimizerData['Optimizer'] = "Nadam"

                    optimizerData['Beta1'] = 0.9
                    optimizerData['Beta2'] = 0.999
                    optimizerData['Epsilon'] = 1e-7

                if (values['Optimizer'] == "Adamax"):
                    optimizerData['Optimizer'] = "Adamax"

                    optimizerData['Beta1'] = 0.9
                    optimizerData['Beta2'] = 0.999
                    optimizerData['Epsilon'] = 1e-7

                # write to file OptimizerParameters.json and then open it
                writeToJSONFile(pathJSON, fileNameOpt, optimizerData)
                g = open('OptimizerParameters.json')
                OptParam = json.load(g)

                # Take loaded OptParam and put in data dictionary to be in MedML.json
                data['OptParam'] = OptParam


        except:

            pathJSON = './'
            fileNameOpt = 'OptimizerParameters'

            # create empty dict of opt data
            optimizerData = {}

            # fill with filler values relative to the Optimizer selection from MedML.json
            if (values['Optimizer'] == "Adam"):
                optimizerData['Optimizer'] = "Adam"

                optimizerData['Beta1'] = 0.9
                optimizerData['Beta2'] = 0.999
                optimizerData['Epsilon'] = 1e-7
                optimizerData['Amsgrad'] = False

            if (values['Optimizer'] == "RectifiedAdam"):
                optimizerData['Optimizer'] = "RectifiedAdam"

                optimizerData['Beta1'] = 0.9
                optimizerData['Beta2'] = 0.999
                optimizerData['Epsilon'] = None
                optimizerData['Decay'] = 0.
                optimizerData['WeightDecay'] = 0.
                optimizerData['Amsgrad'] = False
                optimizerData['TotalSteps'] = 0
                optimizerData['WarmUpProportion'] = 0.1
                optimizerData['MinLr'] = 0.

            if (values['Optimizer'] == "RMSprop"):
                optimizerData['Optimizer'] = "RMSprop"

                optimizerData['Rho'] = 0.9
                optimizerData['Momentum'] = 0.0
                optimizerData['Epsilon'] = 1e-07
                optimizerData['Centered'] = False

            if (values['Optimizer'] == "Adagrad"):
                optimizerData['Optimizer'] = "Adagrad"

                optimizerData['InitialAccumVal'] = 0.1
                optimizerData['Epsilon'] = 1e-7

            if (values['Optimizer'] == "SGD"):
                optimizerData['Optimizer'] = "SGD"

                optimizerData['Momentum'] = 0.0
                optimizerData['Nesterov'] = False

            if (values['Optimizer'] == "Nadam"):
                optimizerData['Optimizer'] = "Nadam"

                optimizerData['Beta1'] = 0.9
                optimizerData['Beta2'] = 0.999
                optimizerData['Epsilon'] = 1e-7

            if (values['Optimizer'] == "Adamax"):
                optimizerData['Optimizer'] = "Adamax"

                optimizerData['Beta1'] = 0.9
                optimizerData['Beta2'] = 0.999
                optimizerData['Epsilon'] = 1e-7

            # write to file OptimizerParameters.json and then open it
            writeToJSONFile(pathJSON, fileNameOpt, optimizerData)
            g = open('OptimizerParameters.json')
            OptParam = json.load(g)

            # Take loaded OptParam and put in data dictionary to be in MedML.json
            data['OptParam'] = OptParam


        # Write to JSON file, Create MedML.json
        writeToJSONFile(pathJSON, path_leaf(values['_FILESAVEAS_']), data)

        sg.popup_ok('Parameters saved in file ' + path_leaf(values['_FILESAVEAS_']))
