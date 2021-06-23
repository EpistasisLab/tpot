#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""This file is part of the TPOT library.

TPOT was primarily developed at the University of Pennsylvania by:
    - Randal S. Olson (rso@randalolson.com)
    - Weixuan Fu (weixuanf@upenn.edu)
    - Daniel Angell (dpa34@drexel.edu)
    - and many more generous open source contributors

TPOT is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as
published by the Free Software Foundation, either version 3 of
the License, or (at your option) any later version.

TPOT is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public
License along with TPOT. If not, see <http://www.gnu.org/licenses/>.
"""
from abc import abstractmethod

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_array


#Attempt to import torch+torchvision for the prebuilt networks, but also just pass if not found
#Error handling for import failure is done when the methods are called.
try:
    from .nn import _get_cuda_device_if_available

    import torch
    from torch import nn
    from torch.autograd import Variable
    from torch.optim import Adam
    from torch.utils.data import TensorDataset, DataLoader
    import torchvision
    from torchvision import models, transforms
except:
    pass

#Attempt to import torchvision for the deep, prebuilt networks
#If not found, pass, but error within the operators that use these


class FeatureExtractor(BaseEstimator, TransformerMixin):
    """Base class for feature extractors for use in TPOT."""

    @property
    def __name__(self):
        """Instance name is the same as the class name."""
        return self.__class__.__name__

    def fit(self, X, y=None):
        """Does nothing but inits model; returns the estimator unchanged
        Method exists to recapitulate API needed for pipelines
        Parameters
        ----------
        X : array-like
        """
        self._init_model(X,y)
        return self

    @abstractmethod
    def _init_model(self, X, y=None): # pragma: no cover
        pass

    @abstractmethod
    def validate_inputs(self, X, y=None): # pragma: no cover
        pass

    @abstractmethod
    def extract(self, X, y=None): # pragma: no cover
        pass

    @abstractmethod
    def expected_input_type(self): # pragma: no cover
        pass

    @abstractmethod
    def expected_output_type(self): # pragma: no cover
        pass

    def transform(self, X, y=None): # pragma: no cover
        return self.extract(X)

    def predict(self, X, y=None):
        return self.transform(X)

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)


class ImageExtractor(FeatureExtractor):
    """Base class for image feature extractors for use in TPOT."""

    @abstractmethod
    def extract(self, X, y=None): # pragma: no cover
        pass

    def validate_inputs(self, X, y=None):
        #Need to confirm that X is an array of images
        #Also forces casting of X to floats
        #y/targets irrelevant for feature extractors; is unused/unchanged

        X = check_array(X, allow_nd=True)

        if np.any(np.iscomplex(X)):
            raise ValueError("Complex data not supported")
        if np.issubdtype(X.dtype, np.object_):
            try:
                X = X.astype(float)
            except (TypeError, ValueError):
                raise ValueError("Input is not, and cannot be cast to, an array of floats")

        return X

    #Define expected inputs and outputs for TPOT Operator Factory setting
    @classmethod
    def expected_input_type(cls):
        return "image"

    @classmethod
    def expected_output_type(cls):
        return np.ndarray

    @abstractmethod
    def _init_model(self, X=None, y=None): # pragma: no cover
        pass


class DeepImageFeatureExtractor(ImageExtractor):
    """Image Feature Extractor using pretrained/premade Deep Learning models"""

    def __init__(
        self,
        network_name="AlexNet",
        verbose=False
    ):
        self.network_name = network_name
        self.verbose = verbose

        self.input_size = None
        self.network = None
        self.device = None
        self.out_feature_num = None


    def _init_model(self, X=None, y=None):
        self.device = _get_cuda_device_if_available()
        self.network = _torchvisionModelAsFeatureExtractor(self.network_name).to(self.device)
        self.out_feature_num = self.network.get_feature_num()

        if(self.verbose):
            print("DeepImageFeatureExtractor init: network: {}; number features output: {}; device: {}".format(self.network_name, self.out_feature_num, self.device))


    def transform(self, X, y=None):
        self.input_size = X.shape
        X = self.validate_inputs(X)

        X_size = X.shape

        # Place X into the expected form if only 1 channel input but as a 3D array
        # (as expected size for all prebuilt models is 4D with [N, 3, H, W], need to stack to RGB 
        # if X is only a sequence of 2D images)
        init_input_size = X.shape
        if(X.ndim == 3):
            X = np.stack((X, X, X), axis=-1)
            X = X.reshape(init_input_size[0], -1, init_input_size[1], init_input_size[2])

        X = torch.tensor(X, dtype=torch.float32)

        #Create normalizing transform for all prebuilt models in torchvision (except inception, which is unsupported)
        #Also running the check here to see if torchvision is installed; if not, raise an error
        try:
            resize_norm_transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
             )])
        except ModuleNotFoundError:
            raise

        #Preallocate feature array based on input batch size and number of features expected
        featureArr = np.empty((init_input_size[0], self.out_feature_num))

        #Feed images into the network (in the appropriate size for the network)

        #To try to avoid hitting memory limits, only feed in images in one at a time 
        #(and unsqueeze to make it 4D)
        #Applying the transform to each as it is passed in
        #TODO: Is there a way to speed this up? Batch in images instead of doing it one by one?
        #Need to consider memory concerns/issues - maybe just accept that the user needs to be aware?

        for i, im in enumerate(X):
            #Create 4D transformed image
            transformedIm = torch.unsqueeze(resize_norm_transform(im),0).to(self.device)

            #Get feature outputs from the network and set in featureArr
            feature_outputs = self.network(transformedIm)
            featureArr[i, :] = feature_outputs.detach().numpy()

            if(self.verbose):
                if(i % 100 == 0):
                    print("DeepImageFeatureExtractor transform: Processing image {} of {}".format(i+1, init_input_size[0]))

        return featureArr


class _torchvisionModelAsFeatureExtractor(nn.Module):
    # pylint: disable=arguments-differ
    def __init__(self, network_name):

        super(_torchvisionModelAsFeatureExtractor, self).__init__()

        network_name = network_name.lower()
        pretrained_val = True

        #Note that input size is: (N, 3, 224, 224) to match every prebuilt network.

        #Every network that is supported will be grabbed from the Pytorch model zoo (trained on imagenet). 
        #Need to change final layer(s) to only output the relevant features
        #This will be different for every single network because they aren't constructed the same way.
        #As such, support for other networks will need to be manually added. 

        #Currently accepts any of the following as input for network name:  
        #[resnet, alexnet, vgg, densenet, googlenet, shufflenet, mobilenet, resnext, wide_resnet, mnasnet]
        #(Not including inception because it has a different input size of 299x299 instead)
        #(Also not including squeezenet because the output layers produce image feature maps rather than vectorized features)

        #Essentially nulling out all layers after (and including) the last ReLU layer (and any following the classifier layer)

        if(network_name == 'resnet'):
            model = models.resnet18(pretrained=pretrained_val)
            model.fc = nn.Identity()
            out_feature_num = 512

        elif(network_name == 'alexnet'):
            model = models.alexnet(pretrained=pretrained_val)
            model.classifier[5] = nn.Identity()
            model.classifier[6] = nn.Identity()
            out_feature_num = 4096

        elif(network_name == 'vgg'):
            model = models.vgg16(pretrained=pretrained_val)
            model.classifier[4] = nn.Identity()
            model.classifier[5] = nn.Identity()
            model.classifier[6] = nn.Identity()
            out_feature_num = 4096

        # elif(network_name == 'squeezenet'):
        #     model = models.squeezenet1_0(pretrained=pretrained_val)
        #     print(model)
        #     model.classifier[1] = nn.Conv2d(512, 512, kernel_size=(1,1), stride=(1,1))
        #     model.classifier[2] = nn.Identity()
        #     model.classifier[3] = nn.Identity()
        #     out_feature_num = 512

        elif(network_name == 'densenet'):
            model = models.densenet161(pretrained=pretrained_val)
            model.classifier = nn.Identity()
            out_feature_num = 2208

        # elif(network_name == 'inception'):
        #     model = models.inception_v3(pretrained=pretrained_val)

        elif(network_name == 'googlenet'):
            model = models.googlenet(pretrained=pretrained_val)
            model.fc = nn.Identity()
            out_feature_num = 1024

        elif(network_name == 'shufflenet'):
            model = models.shufflenet_v2_x1_0(pretrained=pretrained_val)
            model.fc = nn.Identity()
            out_feature_num = 1024

        elif(network_name == 'mobilenet'):
            model = models.mobilenet_v2(pretrained=pretrained_val)
            model.classifier[1] = nn.Identity() #maybe should be model.classifier instead
            out_feature_num = 1280

        elif(network_name == 'resnext'):
            model = models.resnext50_32x4d(pretrained=pretrained_val)
            model.fc = nn.Identity()
            out_feature_num = 2048

        elif(network_name == 'wide_resnet'):
            model = models.wide_resnet50_2(pretrained=pretrained_val)
            model.fc = nn.Identity()
            out_feature_num = 2048

        elif(network_name == 'mnasnet'):
            model = models.mnasnet1_0(pretrained=pretrained_val)
            model.classifier[1] = nn.Identity() #maybe should be model.classifier instead
            out_feature_num = 1280

        else:
            raise NotImplementedError("{} is not supported in TPOT for feature extraction yet. \
                Supported: [resnet, alexnet, vgg, densenet, googlenet, shufflenet, mobilenet, resnext, wide_resnet, mnasnet]".format(network_name))


        self.nn_model = model
        self.out_feature_num = out_feature_num


    def forward(self, x):
        x = self.nn_model(x)
        return x

    def get_feature_num(self):
        return self.out_feature_num


