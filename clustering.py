from numpy import expand_dims
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.models import Model, Sequential
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, GlobalAveragePooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.optimizers import RMSprop, SGD
from keras import backend as K
import os
import keras
import tensorflow as tf
from PIL import Image
import numpy as np
import mtcnn
from mtcnn.mtcnn import MTCNN
from keras.models import load_model
import seaborn as sns
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
import pandas as pd

class CLUSTERING:
    """
    Clase...
    """
    def __init__(self):
        """
        Inicializador...
        """
        print("Instancia de la clase CLUSTERING creada")
        self._model_name = ''

    def _extract_face(self, filename):
        """[summary]

        Args:
            filename ([type]): [description]

        Returns:
            [type]: [description]
        """
        if self._model_name == 'facenet_keras.h5':
            required_size=(160, 160)
        else:
            required_size=(200, 200)
        # load image from file
        image = Image.open(filename)
        # convert to RGB, if needed
        image = image.convert('RGB')
        # convert to array
        pixels = np.asarray(image)
        # create the detector, using default weights
        detector = MTCNN()
        # detect faces in the image
        results = detector.detect_faces(pixels)
        # extract the bounding box from the first face
        x1, y1, width, height = results[0]['box']
        # deal with negative pixel index
        x1, y1 = abs(x1), abs(y1)
        x2, y2 = x1 + width, y1 + height
        # extract the face
        face = pixels[y1:y2, x1:x2]
        # resize pixels to the model size
        image = Image.fromarray(face)
        image = image.resize(required_size)
        face_array = np.asarray(image)
        return face_array

    def _get_embedding(self, model, face_pixels):
        """[summary]

        Args:
            model ([type]): [description]
            face_pixels ([type]): [description]

        Returns:
            [type]: [description]
        """
        # scale pixel values
        face_pixels = face_pixels.astype('float32')
        # standardize pixel values across channels (global)
        mean, std = face_pixels.mean(), face_pixels.std()
        face_pixels = (face_pixels - mean) / std
        # transform face into one sample
        samples = expand_dims(face_pixels, axis=0)
        # make prediction to get embedding
        yhat = model.predict(samples)
        return yhat[0]

    def _load_face(self, dir):
        """[summary]

        Args:
            dir ([type]): [description]

        Returns:
            [type]: [description]
        """
        faces = list()
        # enumerate files
        for filename in os.listdir(dir):
            path = dir + filename
            face = self._extract_face(path)
            faces.append(face)
        return faces

    def _load_dataset(self, dir):
        """[summary]

        Args:
            dir ([type]): [description]

        Returns:
            [type]: [description]
        """
        # list for faces and labels
        X, y = list(), list()
        for subdir in os.listdir(dir):
            path = dir + subdir + '/'
            faces = self._load_face(path)
            labels = [subdir for i in range(len(faces))]
            print("loaded %d sample for class: %s" % (len(faces),subdir) ) # print progress
            X.extend(faces)
            y.extend(labels)
        return np.asarray(X), np.asarray(y)

    def train(self, train_data_dir: str = 'archive/data/train/', validation_data_dir: str = 'archive/data/val/', base_model: str = 'facenet_keras.h5', dir_model: str = ''):
        """[summary]

        Args:
            train_data_dir (str, optional): [description]. Defaults to 'archive/data/train/'.
            validation_data_dir (str, optional): [description]. Defaults to 'archive/data/val/'.
            base_model (str, optional): [description]. Defaults to 'facenet_keras.h5'.
            dir_model (str, optional): [description]. Defaults to ''.
        """
        self._model_name = base_model
        # load train dataset
        trainX, trainy = self._load_dataset(train_data_dir)
        print(trainX.shape, trainy.shape)
        # load test dataset
        testX, testy = self._load_dataset(validation_data_dir)
        print(testX.shape, testy.shape)
        
        model = load_model(dir_model + base_model)
        print('Loaded Model')
        print(model.inputs)
        print(model.outputs)

        # convert each face in the train set to an embedding
        newTrainX = list()
        for face_pixels in trainX:
            embedding = self._get_embedding(model, face_pixels)
            newTrainX.append(embedding)
        newTrainX = np.asarray(newTrainX)
        print(newTrainX.shape)
        # convert each face in the test set to an embedding
        newTestX = list()
        for face_pixels in testX:
            embedding = self._get_embedding(model, face_pixels)
            newTestX.append(embedding)
        newTestX = np.asarray(newTestX)
        print(newTestX.shape)

        df = pd.DataFrame(newTrainX)
        df["target"] = trainy
        df.head()

        # Create a PCA instance:
        pca = PCA(n_components=2) 
        # Fit pca to 'X'
        pca_features = pca.fit_transform(newTrainX)
        print (pca_features.shape)

        df_plot = pd.DataFrame(pca_features)
        df_plot["target"] = trainy

        plt.figure(figsize=(16, 6))
        sns.scatterplot(x=df_plot[0] , y= df_plot[1], data = df_plot,  hue = "target")
