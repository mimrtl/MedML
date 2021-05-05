"""NiftiGenerator is a tool to ingest Nifti images using Nibabel, apply basic augmentation, and utilize them as inputs to a deep learning model
Data is sampled as fixed-size chunks. The chunks can be as small as you would like or as large as an entire image.
Augmentation is currently only 2D (not performed on the 3rd dimension of the input images in consideration of non-isotropic input data).
Sampling of each chunk of data is performed after any augmentation.
# please see the source code for implementation details. Basic implementations are as follows:
# SingleNiftiGenerator -- To use for generating a single input into your model, do something like the following:
    # define the NiftiGenerator
    niftiGen = NiftiGenerator.SingleNiftiGenerator()
    # get augmentation options (see help for get_default_augOptions for more details! )
    niftiGen_augment_opts = NiftiGenerator.SingleNiftiGenerator.get_default_augOptions()
    niftiGen_augment_opts.hflips = True
    niftiGen_augment_opts.vflips = True
    # get normalization options ( see help for get_default_normOptions for more details! )
    niftiGen_norm_opts = NiftiGenerator.SingleNiftiGenerator.get_default_normOptions()
    niftiGen_norm_opts.normXtype = 'auto'
    # initialize the generator (where x_data_train is either a path to a single folder or a list of Nifti files)
    niftiGenTrain.initialize( x_data_train, augOptions=niftiGen_augment_opts, normOptions=niftiGen_norm_opts )
    ## in your training function you will then call something like:
    NiftiGenerator.generate_chunks( niftiGen, chunk_size=(128,128,5), batch_size=16 )
    ## to generate a batch of 16, 128x128x5 chunks of data
# PairedNiftiGenerator -- To use for generating paired inputs into your model, do something like the following:
    # define the NiftiGenerator
    niftiGen = NiftiGenerator.PairedNiftiGenerator()
    # get augmentation options (see help for get_default_augOptions for more details! )
    niftiGen_augment_opts = NiftiGenerator.PairedNiftiGenerator.get_default_augOptions()
    niftiGen_augment_opts.hflips = True
    niftiGen_augment_opts.vflips = True
    niftiGen_augment_opts.rotations = 5
    niftiGen_augment_opts.scalings = .1
    # get normalization options ( see help for get_default_normOptions for more details! )
    niftiGen_norm_opts = NiftiGenerator.PairedNiftiGenerator.get_default_normOptions()
    niftiGen_norm_opts.normXtype = 'auto'
    niftiGen_norm_opts.normYtype = 'fixed'
    niftiGen_norm_opts.normYoffset = 0
    niftiGen_norm_opts.normYscale = 50000
    # initialize the generator (where x_data_train and y_data_train are either a paths to a single folder or lists of Nifti files)
    niftiGen.initialize( x_data_train, y_data_train, augOptions=niftiGen_augment_opts, normOptions=niftiGen_norm_opts )
    ## in your training function you will then call something like:
    NiftiGenerator.generate_paired_chunks( niftiGen, chunk_size=(32,32,32), batch_size=64 )
    ## to generate a batch of 64, 32x32x32 chunks of paired data
# More advanced things:
    The NiftiGenerators are designed to allow flexible callbacks at various places to do more advanced things to the input data.
    Custom functions can be used in three different places:
        1. During augmentation using the augOptions.additionalFunction. This additional function will be called at the last step of the augmentation.
        2. During normalization using the normType ='function'. This function will be called to do the normalization of the input data.
                Note that this is slow because it requires loading the whole Nifti volume
        3. During the sampling of each batch by passing the additional function batchTransformFunction to the initialize call.
                This function will be called right before each batch is returned. The transform function should preserve the size of the batch.
"""

import nibabel as nib
import cv2
from glob import glob
import types
import os
import sys
import numpy as np
import logging
from scipy.ndimage import affine_transform

# set up logging
module_logger = logging.getLogger(__name__)
module_logger_handler = logging.StreamHandler()
module_logger_formatter = logging.Formatter('%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
module_logger_handler.setFormatter(module_logger_formatter)
module_logger.addHandler(module_logger_handler)
module_logger.setLevel(logging.INFO)

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
# We will utilize a clever approach to enable pre-fetching of the generator.
#   below we use code is from https://github.com/justheuristic/prefetch_generator/blob/master/prefetch_generator/__init__.py
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
import threading
import sys
import queue as Queue


class BackgroundGenerator(threading.Thread):
    def __init__(self, generator, max_prefetch=1):
        """
        This function transforms generator into a background-thead generator.
        :param generator: generator or genexp or any
        It can be used with any minibatch generator.
        It is quite lightweight, but not entirely weightless.
        Using global variables inside generator is not recommended (may rise GIL and zero-out the benefit of having a background thread.)
        The ideal use case is when everything it requires is store inside it and everything it outputs is passed through queue.
        There's no restriction on doing weird stuff, reading/writing files, retrieving URLs [or whatever] wlilst iterating.
        :param max_prefetch: defines, how many iterations (at most) can background generator keep stored at any moment of time.
        Whenever there's already max_prefetch batches stored in queue, the background process will halt until one of these batches is dequeued.
        !Default max_prefetch=1 is okay unless you deal with some weird file IO in your generator!
        Setting max_prefetch to -1 lets it store as many batches as it can, which will work slightly (if any) faster, but will require storing
        all batches in memory. If you use infinite generator with max_prefetch=-1, it will exceed the RAM size unless dequeued quickly enough.
        """
        threading.Thread.__init__(self)
        self.queue = Queue.Queue(max_prefetch)
        self.generator = generator
        self.daemon = True
        self.start()
        self.exhausted = False

    def run(self):
        for item in self.generator:
            self.queue.put(item)
        self.queue.put(None)

    def next(self):
        if self.exhausted:
            raise StopIteration
        else:
            next_item = self.queue.get()
            if next_item is None:
                raise StopIteration
            return next_item

    # Python 3 compatibility
    def __next__(self):
        return self.next()

    def __iter__(self):
        return self


# decorator
class background:
    def __init__(self, max_prefetch=1):
        self.max_prefetch = max_prefetch

    def __call__(self, gen):
        def bg_generator(*args, **kwargs):
            return BackgroundGenerator(gen(*args, **kwargs), max_prefetch=self.max_prefetch)

        return bg_generator


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
# end prefetch_generator
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
# utility functions to get multi-worker generators from SingleNiftiGenerators and PairedNiftiGenerators
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
def generate_chunks(single_niftigen, chunk_size=(32, 32, 32), batch_size=16, num_workers=4):
    """realizes a multi-threaded pre-fetching generator to generate data from a SingleNiftiGenerator.
    Arguments
        single_niftigen - an initialized PairedNiftiGenerator
        chunk_size - a length-3 tuple, e.g., (32,32,32) that indicates the size of the sampled chunks
        batch_size - the number of chunks to sample in a batch
        num_workers - the number of workers (threads)
    """

    # we simply make an array of generators and call each one with a batch_size of 1
    generators = []
    for i in range(num_workers):
        generators.append(single_niftigen.generate_chunks(chunk_size, batch_size=1))

    # store the batch of data
    batch_X = np.zeros([batch_size, chunk_size[0], chunk_size[1], chunk_size[2]])

    # loop through each pre-fetching generator to retrieve a full batch
    curr_generator = 0
    while True:
        for i in range(batch_size):
            curr_X = next(generators[curr_generator])

            batch_X[i, :, :, :] = curr_X

            curr_generator += 1
            if curr_generator == num_workers:
                curr_generator = 0

        yield (batch_X)


def generate_paired_chunks(paired_niftigen, chunk_size=(32, 32, 32), batch_size=16, num_workers=4):
    """realizes a multi-threaded pre-fetching generator to generate data from a PairedNiftiGenerator.
    Arguments
        paired_niftigen - an initialized PairedNiftiGenerator
        chunk_size - a length-3 tuple, e.g., (32,32,32) that indicates the size of the sampled chunks
        batch_size - the number of chunks to sample in a batch
        num_workers - the number of workers (threads)
    """

    # we simply make an array of generators and call each one with a batch_size of 1
    generators = []
    for i in range(num_workers):
        generators.append(paired_niftigen.generate_chunks(chunk_size, batch_size=1))

    # store the batches of data
    batch_X = np.zeros([batch_size, chunk_size[0], chunk_size[1], chunk_size[2]])
    batch_Y = np.zeros([batch_size, chunk_size[0], chunk_size[1], chunk_size[2]])

    # loop through each pre-fetching generator to retrieve a full batch
    curr_generator = 0
    while True:
        for i in range(batch_size):
            curr_X, curr_Y = next(generators[curr_generator])

            batch_X[i, :, :, :] = curr_X
            batch_Y[i, :, :, :] = curr_Y

            curr_generator += 1
            if curr_generator == num_workers:
                curr_generator = 0

        yield (batch_X, batch_Y)


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
# end utility functions
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
# SingleNiftiGenerator: data generator for a single (unpaired) set of nifti files
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
class SingleNiftiGenerator:
    """SingleNiftiGenerator
    A generator that provides slices or patches of data from a set of Nifti files
    """
    augOptions = types.SimpleNamespace()
    normOptions = types.SimpleNamespace()

    def initialize(self, inputX, augOptions=None, normOptions=None, batchTransformFunction=None):
        """Initialize a SingleNiftiGenerator object.
        Arguments
            inputX - a valid folder containing Nifti (.nii or .nii.gz files)
                        -or-
                     a list of Nifti (.nii or .nii.gz) files
            augOptions - a Augmentation Options namespace. See get_default_augOptions().
            normOptions - a Normalization Options namespace. See get_default_normOptions().
            batchTransformFunction - an optional user-specified function that takes a batch of data as its input and performs a
                                    transformation, and returns new data in the same shape.
        """

        # if input is a list, let's just use that
        # otherwise consider this input as a folder
        if isinstance(inputX, list):
            self.inputFilesX = inputX
        else:
            self.inputFilesX = glob(os.path.join(inputX, '*.nii.gz'), recursive=True) + glob(
                os.path.join(inputX, '*.nii'), recursive=True)
        num_Xfiles = len(self.inputFilesX)

        module_logger.info('{} datasets were found'.format(num_Xfiles))

        if augOptions is None:
            module_logger.warning('No augmentation options were specified.')
            self.augOptions = SingleNiftiGenerator.get_default_augOptions()
        else:
            self.augOptions = augOptions

        if normOptions is None:
            module_logger.warning('No normalization options were specified.')
            self.normOptions = SingleNiftiGenerator.get_default_normOptions(self)
        else:
            self.normOptions = normOptions

        # handle normalization
        if self.normOptions.normXtype == 'auto'.lower():
            module_logger.info('normalization is ''auto''. Computing normalizations now...')
            self.normXready = [False] * num_Xfiles
            self.normXoffset = [0] * num_Xfiles
            self.normXscale = [1] * num_Xfiles
            for i in range(num_Xfiles):
                Ximg = nib.load(self.inputFilesX[i])
                tmpX = Ximg.get_fdata()
                self.normXoffset[i] = np.mean(tmpX)
                self.normXscale[i] = np.std(tmpX)
                self.normXready[i] = True
        elif self.normOptions.normXtype == 'fixed'.lower():
            self.normXready = [True] * num_Xfiles
            self.normXoffset = [self.normOptions.normXoffset] * num_Xfiles
            self.normXscale = [self.normOptions.normXscale] * num_Xfiles
        elif self.normOptions.normXtype == 'function'.lower():
            self.normXready = [False] * num_Xfiles
        elif self.normOptions.normXtype == 'none'.lower():
            self.normXready = [True] * num_Xfiles
            self.normXoffset = [0] * num_Xfiles
            self.normXscale = [1] * num_Xfiles
        else:
            module_logger.error('Fatal Error: Normalization for X was specified as an unknown value.')
            sys.exit(1)

        # optional function to transform each batch
        if batchTransformFunction is not None:
            self.batchTransformFunction = batchTransformFunction

        # set random seed
        np.random.seed(augOptions.augseed)

        # we are now initialized
        self.initialized = True

    @background(max_prefetch=1)
    def generate_chunks(self, chunk_size=(32, 32, 32), batch_size=16):
        """Get a prefetching generator to yield chunks of data.
        Arguments
            chunk_size - a length-3 tuple, e.g., (32,32,32) that indicates the size of the sampled chunks
            batch_size - the number of chunks to sample in a batch
        Yields
            a batch of data from the NiftiGenerator
        """
        if not self.initialized:
            module_logger.error('This NiftiGenerator is not initialized. Make sure to run initialize(...) first!')
        while True:
            yield self.get_batch(chunk_size, batch_size)

    def get_batch(self, chunk_size=(32, 32, 32), batch_size=16):
        """Get a batch of samples as chunks (samples can be the full image size if desired)
        Arguments
            chunk_size - a length-3 tuple, e.g., (32,32,32) that indicates the size of the sampled chunks
            batch_size - the number of chunks to sample in a batch
        Returns
            a batch of paired data
        """
        # create empty variables for this batch
        batch_X = np.zeros([batch_size, chunk_size[0], chunk_size[1], chunk_size[2]])

        for i in range(batch_size):
            # get a random subject
            j = np.random.randint(0, len(self.inputFilesX))
            currImgFileX = self.inputFilesX[j]

            # load nifti header
            module_logger.debug('reading file {}'.format(currImgFileX))
            Ximg = nib.load(currImgFileX)

            XimgShape = Ximg.header.get_data_shape()

            # determine sampling location
            x = np.random.randint(0, XimgShape[0] - chunk_size[0] - 1)
            y = np.random.randint(0, XimgShape[1] - chunk_size[1] - 1)
            z = np.random.randint(0, XimgShape[2] - chunk_size[2] - 1)

            xe = x + chunk_size[0]
            ye = y + chunk_size[1]
            ze = z + chunk_size[2]

            # handle input data normalization and sampling
            if self.normOptions.normXtype == 'function'.lower():
                # normalization is performed via a specified function
                # get normalized data (and read whole volume)
                tmpX = self.normOptions.normXfunction(Ximg.get_fdata())
                # sample data
                XimgSlices = tmpX[:, :, z:ze]
            else:
                # type is none, auto, or fixed, no computation should be needed
                # sample data
                XimgSlices = Ximg.dataobj[:, :, z:ze]
                # do normalization
                XimgSlices = (XimgSlices - self.normXoffset[j]) / self.normXscale[j]

            # ensure 3D matrix if batch size is equal to 1
            if XimgSlices.ndim == 2:
                XimgSlices = XimgSlices[..., np.newaxis]

            # augmentation here
            M = self.get_augment_transform()
            XimgSlices = self.do_augment(XimgSlices, M)

            # sample chunk of data
            XimgChunk = XimgSlices[x:xe, y:ye, :]

            # put into data array for batch for this batch of samples
            batch_X[i, :, :, :] = XimgChunk

        # optional additional transform to the batch of data
        if self.batchTransformFunction:
            batch_X = self.batchTransformFunction(batch_X)

        return (batch_X)

    def get_default_normOptions(self):
        """Get a Normalization Options namespace for a SingleNiftiGenerator, with the default options
        Returns
            a normOptions namespace with the following structure:
                normOptions.normXType = 'none', where the options are 'none', 'auto', 'fixed', and 'function'
                                        for none, no normalization is done
                                        for auto, a Z-score normalization is done on the Nifti volume to make mean=0, stdev=1
                                        for fixed, a specified offset and scaling factor is applied (data-offset)/scale
                                        for function, a python function is passed that takes the input data and returns a normalized version
                normOptions.normXoffset = 0, where the value (floating point) is the offset applied to 'fixed' normalization, where the normalization is (data-normXoffset)/normXscale
                normOptions.normXscale = 0, where the value (floating point) is the scale applied to 'fixed' normalization, where the normalization is (data-normXoffset)/normXscale
                normOptions.normXfunction = None, where a user-specified function is specified that takes a single dataset as its input and performs a normalization and
                                            returns new data in the same shape.
                normOptions.normXinterp = cv2.INTER_CUBIC, which specifies the OpenCV interpolation method that is used when the augmentation transformation is applied.
                                          valid types are cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.LANCZOS4, cv2.INTER_LINEAR_EXACT,
                                              cv2.INTER_NEAREST_EXACT, cv2.INTER_MAX
                                          see https://docs.opencv.org/3.4/da/d54/group__imgproc__transform.html for more information
        """
        normOptions = types.SimpleNamespace()
        normOptions.normXtype = 'none'
        normOptions.normXoffset = 0
        normOptions.normXscale = 1
        normOptions.normXfunction = None
        normOptions.normXinterp = cv2.INTER_CUBIC

        return normOptions

    def get_default_augOptions(self):
        """Get an Augmentation Options namespace for a NiftiGenerator, with the default options
        Returns
            a normOptions namespace with the following structure:
                augOptions.augmode = 'reflect', where the options are 'mirror','nearest','reflect','wrap' that defines how augmented data is extended beyond its boundaries.
                                     scipy.ndimage documentation for more information
                augOptions.augseed = 813, where the value (integer) is the random seed to enable reproducible augmentation
                augOptions.addnoise = 0, where the value (floating point) is the sigma of mean zero Gaussian noise (see numpy.random.normal)
                augOptions.hflips = False, where the value (True, False) indicates whether to perform random horizontal flips
                augOptions.vflips = False, where the value (True, False) indicates whether to perform random vertical flips
                augOptions.rotations = 0, where the value (floating point) specifies the random amount of rotations in degrees within the range [-rotations,rotations]
                augOptions.scalings = 0, where the value (floating point) specifies the random amount of scaling within the range [(1-scale),(1+scale)]
                augOptions.shears = 0, where the value (floating point) specifies the random amount of shears in degrees within the range [-shears,shears]
                augOptions.translations = 0, where the value (integer) is the random amount of translations applied in the horizontal and vertical directions
                                          within the range [-translations,translations]
                augOptions.additionalFunction = None, where a user-specified function is specified that takes a single dataset as its input and performs augmentation
                                                and returns new data in the same shape.
        """
        augOptions = types.SimpleNamespace()
        augOptions.augmode = 'reflect'
        augOptions.augseed = 813
        augOptions.addnoise = 0
        augOptions.hflips = False
        augOptions.vflips = False
        augOptions.rotations = 0
        augOptions.scalings = 0
        augOptions.shears = 0
        augOptions.translations = 0
        augOptions.additionalFunction = None

        return augOptions

    def get_augment_transform(self):
        """Internal function used to calculate the augmentation transform based on augOptions
        Returns
            M, an augmentation transform
        """
        # use affine transformations as augmentation
        M = np.eye(3)
        # horizontal flips
        if self.augOptions.hflips:
            M_ = np.eye(3)
            M_[1][1] = 1 if np.random.random() < 0.5 else -1
            M = np.matmul(M, M_)
        # vertical flips
        if self.augOptions.vflips:
            M_ = np.eye(3)
            M_[0][0] = 1 if np.random.random() < 0.5 else -1
            M = np.matmul(M, M_)
        # rotations
        if np.abs(self.augOptions.rotations) > 1e-2:
            rot_angle = np.pi / 180.0 * np.random.randint(-np.abs(self.augOptions.rotations),
                                                          np.abs(self.augOptions.rotations))
            M_ = np.eye(3)
            M_[0][0] = np.cos(rot_angle)
            M_[0][1] = np.sin(rot_angle)
            M_[1][0] = -np.sin(rot_angle)
            M_[1][1] = np.cos(rot_angle)
            M = np.matmul(M, M_)
        # shears
        if np.abs(self.augOptions.shears) > 1e-2:
            rot_angle_x = np.pi / 180.0 * np.random.randint(-np.abs(self.augOptions.rotations),
                                                            np.abs(self.augOptions.rotations))
            rot_angle_y = np.pi / 180.0 * np.random.randint(-np.abs(self.augOptions.rotations),
                                                            np.abs(self.augOptions.rotations))
            M_ = np.eye(3)
            M_[0][1] = np.tan(rot_angle_x)
            M_[1][0] = np.tan(rot_angle_y)
            M = np.matmul(M, M_)
        # scaling (also apply specified resizing [--imsize] here)
        if np.abs(self.augOptions.scalings) > 1e-4:
            init_factor_x = 1
            init_factor_y = 1
            if np.abs(self.augOptions.scalings) > 1e-4:
                random_factor_x = np.random.randint(-np.abs(self.augOptions.scalings) * 10000,
                                                    np.abs(self.augOptions.scalings) * 10000) / 10000
                random_factor_y = np.random.randint(-np.abs(self.augOptions.scalings) * 10000,
                                                    np.abs(self.augOptions.scalings) * 10000) / 10000
            else:
                random_factor_x = 0
                random_factor_y = 0
            scale_factor_x = init_factor_x + random_factor_x
            scale_factor_y = init_factor_y + random_factor_y
            M_ = np.eye(3)
            M_[0][0] = scale_factor_x
            M_[1][1] = scale_factor_y
            M = np.matmul(M, M_)
        # translations
        if np.abs(self.augOptions.translations) > 0:
            translate_x = np.random.randint(-np.abs(self.augOptions.translations), np.abs(self.augOptions.translations))
            translate_y = np.random.randint(-np.abs(self.augOptions.translations), np.abs(self.augOptions.translations))
            M_ = np.eye(3)
            M_[0][2] = translate_x
            M_[1][2] = translate_y
            M = np.matmul(M, M_)

        return M

    def do_augment(self, X, M):
        """Internal function used to apply augmentation to a specified image
        Arguments
            X, the input image
            M, the transformation matrix, that applies a 2D transform slicewise to input X
        Returns
            the augmented image
        """
        # now apply the transform
        X_ = np.zeros_like(X)

        for k in range(X.shape[2]):
            X_[:, :, k] = affine_transform(X[:, :, k], M, output_shape=X[:, :, k].shape, mode=self.augOptions.augmode)

        # optionally add noise
        if np.abs(self.augOptions.addnoise) > 1e-10:
            noise_mean = 0
            noise_sigma = self.augOptions.addnoise
            noise = np.random.normal(noise_mean, noise_sigma,
                                     X_.shape)  # [:,:,k] for k=0,1,2. Which k? output_shape was undefined 3rd arg here
            X_ += noise

        # if an additional augmentation function is supplied, apply it here
        if self.augOptions.additionalFunction:
            X_ = self.augOptions.additionalFunction(X_)

        return X_


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
# end SingleNiftiGenerator
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
# PairedNiftiGenerator: data generator for paired sets of nifti files
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
class PairedNiftiGenerator(SingleNiftiGenerator):
    """PairedNiftiGenerator
    A generator that provides paired slices or patches of data from a set of paired Nifti files
    """

    def initialize(self, inputX, inputY, augOptions=None, normOptions=None, batchTransformFunction=None):
        """Initialize a DoubleNiftiGenerator object.
        Arguments
            inputX - a valid folder containing Nifti (.nii or .nii.gz files)
                        -or-
                     a list of Nifti (.nii or .nii.gz) files
            inputY - a valid folder containing Nifti (.nii or .nii.gz files)
                        -or-
                     a list of Nifti (.nii or .nii.gz) files
            augOptions - a Augmentation Options namespace. See get_default_augOptions().
            normOptions - a Normalization Options namespace. See get_default_normOptions().
            batchTransformFunction - an optional user-specified function that takes a batch of data as its input and performs a
                                    transformation, and returns new data in the same shape.
        """

        # if input is a list, let's just use that
        # otherwise consider this input as a folder
        if isinstance(inputX, list):
            self.inputFilesX = inputX
        else:
            self.inputFilesX = sorted(
                glob(os.path.join(inputX, '*.nii.gz'), recursive=True) + glob(os.path.join(inputX, '*.nii'),
                                                                              recursive=True))

        if isinstance(inputY, list):
            self.inputFilesY = inputY
        else:
            self.inputFilesY = sorted(
                glob(os.path.join(inputY, '*.nii.gz'), recursive=True) + glob(os.path.join(inputY, '*.nii'),
                                                                              recursive=True))

        num_Xfiles = len(self.inputFilesX)
        num_Yfiles = len(self.inputFilesY)
        module_logger.info('{} datasets were found for X'.format(num_Xfiles))
        module_logger.info('{} datasets were found for Y'.format(num_Yfiles))

        if num_Xfiles != num_Yfiles:
            module_logger.error('Fatal Error: Mismatch in number of datasets.')
            sys.exit(1)

        if augOptions is None:
            module_logger.warning('No augmentation options were specified.')
            self.augOptions = self.get_default_augOptions()
        else:
            self.augOptions = augOptions

        if normOptions is None:
            module_logger.warning('No normalization options were specified.')
            self.normOptions = self.get_default_normOptions()
        else:
            self.normOptions = normOptions

        # handle normalization
        if self.normOptions.normXtype == 'auto'.lower():
            module_logger.info('X normalization is ''auto''. Computing normalizations now...')
            self.normXready = [False] * num_Xfiles
            self.normXoffset = [0] * num_Xfiles
            self.normXscale = [1] * num_Xfiles
            for i in range(num_Xfiles):
                Ximg = nib.load(self.inputFilesX[i])
                tmpX = Ximg.get_fdata()
                self.normXoffset[i] = np.mean(tmpX)
                self.normXscale[i] = np.std(tmpX)
                self.normXready[i] = True
        elif self.normOptions.normXtype == 'fixed'.lower():
            self.normXready = [True] * num_Xfiles
            self.normXoffset = [self.normOptions.normXoffset] * num_Xfiles
            self.normXscale = [self.normOptions.normXscale] * num_Xfiles
        elif self.normOptions.normXtype == 'function'.lower():
            self.normXready = [False] * num_Xfiles
        elif self.normOptions.normXtype == 'none'.lower():
            self.normXready = [True] * num_Xfiles
            self.normXoffset = [0] * num_Xfiles
            self.normXscale = [1] * num_Xfiles
        else:
            module_logger.error('Fatal Error: Normalization for X was specified as an unknown value.')
            sys.exit(1)

        if self.normOptions.normYtype == 'auto'.lower():
            module_logger.info('Y normalization is ''auto''. Computing normalizations now...')
            self.normYready = [False] * num_Yfiles
            self.normYoffset = [0] * num_Yfiles
            self.normYscale = [1] * num_Yfiles
            for i in range(num_Yfiles):
                Yimg = nib.load(self.inputFilesY[i])
                tmpY = Yimg.get_fdata()
                self.normYoffset[i] = np.mean(tmpY)
                self.normYscale[i] = np.std(tmpY)
                self.normYready[i] = True
        elif self.normOptions.normYtype == 'fixed'.lower():
            self.normYready = [True] * num_Yfiles
            self.normYoffset = [self.normOptions.normYoffset] * num_Yfiles
            self.normYscale = [self.normOptions.normYscale] * num_Yfiles
        elif self.normOptions.normYtype == 'function'.lower():
            self.norYready = [False] * num_Yfiles
        elif self.normOptions.normYtype == 'none'.lower():
            self.normYready = [True] * num_Yfiles
            self.normYoffset = [0] * num_Yfiles
            self.normYscale = [1] * num_Yfiles
        else:
            module_logger.error('Fatal Error: Normalization for Y was specified as an unknown value.')
            sys.exit(1)

        # optional function to transform each batch
        if batchTransformFunction is not None:
            self.batchTransformFunction = batchTransformFunction

        # set random seed
        np.random.seed(self.augOptions.augseed)

        # we are now initialized
        self.initialized = True

    def get_default_normOptions(self):
        """Get a Normalization Options namespace for a PairedNiftiGenerator, with the default options
        Returns
            a normOptions namespace with the following structure for paired data X and Y:
                normOptions.normXType = 'none', where the options are 'none', 'auto', 'fixed', and 'function'
                                        for none, no normalization is done
                                        for auto, a Z-score normalization is done on the Nifti volume to make mean=0, stdev=1
                                        for fixed, a specified offset and scaling factor is applied (data-offset)/scale
                                        for function, a python function is passed that takes the input data and returns a normalized version
                normOptions.normXoffset = 0, where the value (floating point) is the offset applied to 'fixed' normalization, where the normalization is (data-normXoffset)/normXscale
                normOptions.normXscale = 0, where the value (floating point) is the scale applied to 'fixed' normalization, where the normalization is (data-normXoffset)/normXscale
                normOptions.normXfunction = None, where a user-specified function is specified that takes a single dataset as its input and performs a normalization and
                                            returns new data in the same shape.
                normOptions.normXinterp = cv2.INTER_CUBIC, which specifies the OpenCV interpolation method that is used when the augmentation transformation is applied.
                                          valid types are cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.LANCZOS4, cv2.INTER_LINEAR_EXACT,
                                              cv2.INTER_NEAREST_EXACT, cv2.INTER_MAX
                                          see https://docs.opencv.org/3.4/da/d54/group__imgproc__transform.html for more information
                normOptions.normYType = 'none', where the options are 'none', 'auto', 'fixed', and 'function'
                                        for none, no normalization is done
                                        for auto, a Z-score normalization is done on the Nifti volume to make mean=0, stdev=1
                                        for fixed, a specified offset and scaling factor is applied (data-offset)/scale
                                        for function, a python function is passed that takes the input data and returns a normalized version
                normOptions.normYoffset = 0, where the value (floating point) is the offset applied to 'fixed' normalization, where the normalization is (data-normXoffset)/normXscale
                normOptions.normYscale = 0, where the value (floating point) is the scale applied to 'fixed' normalization, where the normalization is (data-normXoffset)/normXscale
                normOptions.normYfunction = None, where a user-specified function is specified that takes a single dataset as its input and performs a normalization and
                                            returns new data in the same shape.
                normOptions.normYinterp = cv2.INTER_CUBIC, which specifies the OpenCV interpolation method that is used when the augmentation transformation is applied.
                                          valid types are cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.LANCZOS4, cv2.INTER_LINEAR_EXACT,
                                              cv2.INTER_NEAREST_EXACT, cv2.INTER_MAX
                                          see https://docs.opencv.org/3.4/da/d54/group__imgproc__transform.html for more information
        """
        normOptions = types.SimpleNamespace()

        normOptions.normXtype = 'none'
        normOptions.normXoffset = 0
        normOptions.normXscale = 1
        normOptions.normXfunction = None
        normOptions.normXinterp = cv2.INTER_CUBIC

        normOptions.normYtype = 'none'
        normOptions.normYoffset = 0
        normOptions.normYscale = 1
        normOptions.normYinterp = cv2.INTER_CUBIC
        normOptions.normXfunction = None

        return normOptions

    def get_batch(self, chunk_size=(32, 32, 32), batch_size=16):
        """Get a batch of samples as chunks (samples can be the full image size if desired)
        Arguments
            chunk_size - a length-3 tuple, e.g., (32,32,32) that indicates the size of the sampled chunks
            batch_size - the number of chunks to sample in a batch
        Returns
            a batch of paired data
        """
        # create empty variables for this batch
        batch_X = np.zeros([batch_size, chunk_size[0], chunk_size[1], chunk_size[2]])
        batch_Y = np.zeros([batch_size, chunk_size[0], chunk_size[1], chunk_size[2]])

        for i in range(batch_size):
            # get a random subject
            j = np.random.randint(0, len(self.inputFilesX))
            currImgFileX = self.inputFilesX[j]
            currImgFileY = self.inputFilesY[j]

            # load nifti header
            module_logger.debug('reading files {}, {}'.format(currImgFileX, currImgFileY))
            Ximg = nib.load(currImgFileX)
            Yimg = nib.load(currImgFileY)

            XimgShape = Ximg.header.get_data_shape()
            YimgShape = Yimg.header.get_data_shape()

            if not XimgShape == YimgShape:
                module_logger.warning(
                    'input data ({} and {}) is not the same size. this may lead to unexpected results or errors!'.format(
                        currImgFileX, currImgFileY))

            # determine sampling location
            if chunk_size[0] == XimgShape[0]:
                x = 0
            else:
                x = np.random.randint(0, XimgShape[0] - chunk_size[0] - 1)
            if chunk_size[1] == XimgShape[1]:
                y = 0
            else:
                y = np.random.randint(0, XimgShape[1] - chunk_size[1] - 1)
            if chunk_size[2] == XimgShape[2]:
                z = 0
            else:
                z = np.random.randint(0, XimgShape[2] - chunk_size[2] - 1)

            xe = x + chunk_size[0]
            ye = y + chunk_size[1]
            ze = z + chunk_size[2]

            # handle input data normalization and sampling
            if self.normOptions.normXtype == 'function'.lower():
                # normalization is performed via a specified function
                # get normalized data (and read whole volume)
                tmpX = self.normOptions.normXfunction(Ximg.get_fdata())
                # sample data
                XimgSlices = tmpX[:, :, z:ze]
            else:
                # type is none, auto, or fixed, no computation should be needed
                # sample data
                XimgSlices = Ximg.dataobj[:, :, z:ze]
                # do normalization
                XimgSlices = (XimgSlices - self.normXoffset[j]) / self.normXscale[j]

            if self.normOptions.normYtype == 'function'.lower():
                # normalization is performed via a specified function
                # get normalized data (and read whole volume)
                tmpY = self.normOptions.normYfunction(Yimg.get_fdata())
                # sample data
                YimgSlices = tmpY[:, :, z:ze]
            else:
                # type is none, auto, or fixed, no computation should be needed
                # sample data
                YimgSlices = Yimg.dataobj[:, :, z:ze]
                # do normalization
                YimgSlices = (YimgSlices - self.normYoffset[j]) / self.normYscale[j]

                # ensure 3D matrix if z size is equal to 1
            if XimgSlices.ndim == 2:
                XimgSlices = XimgSlices[..., np.newaxis]
            if YimgSlices.ndim == 2:
                YimgSlices = YimgSlices[..., np.newaxis]

                # augmentation here
            M = self.get_augment_transform()
            XimgSlices = self.do_augment(XimgSlices, M)
            YimgSlices = self.do_augment(YimgSlices, M)

            # sample chunks of data
            XimgChunk = XimgSlices[x:xe, y:ye, :]
            YimgChunk = YimgSlices[x:xe, y:ye, :]

            # put into data array for batch for this batch of samples
            batch_X[i, :, :, :] = XimgChunk
            batch_Y[i, :, :, :] = YimgChunk

        # optional additional transform to the batch of data
        if self.batchTransformFunction:
            batch_X, batch_Y = self.batchTransformFunction(batch_X, batch_Y)

        return (batch_X, batch_Y)
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
# end PairedNiftiGenerator
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------