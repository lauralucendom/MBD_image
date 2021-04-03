from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
import os
import keras
from keras.models import Model, Sequential
from keras.layers import Dropout, Flatten, Dense
from keras.optimizers import RMSprop
from keras import backend as K
import seaborn as sns
import matplotlib.pyplot as plt
from keras.models import load_model
import numpy as np
from keras.preprocessing import image
import cv2
import json
from results import Results


class CNN:
    """
    Clase...
    """
    def __init__(self):
        """
        Constructor de la clase
        """
        print("Instancia de la clase DATA_AUGMENTATION creada")
        self._model_name = ""
        self._model = None

    def _vgg16CNNtl(self, input_shape, outclass, sigma='sigmoid', dropout: float = 0.5, vgg16weight: str = 'archive/models/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'):
        """[summary]

        Args:
            input_shape ([type]): [description]
            outclass ([type]): [description]
            sigma (str, optional): [description]. Defaults to 'sigmoid'.
            dropout (float, optional): [description]. Defaults to 0.5.
            vgg16weight (str, optional): [description]. Defaults to 'archive/models/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'.

        Returns:
            [type]: [description]
        """
        base_model = None
        base_model = keras.applications.VGG16(weights=None, include_top=False, input_shape=input_shape)
        base_model.load_weights(vgg16weight)
            
        top_model = Sequential()
        top_model.add(Flatten(input_shape=base_model.output_shape[1:]))
        for i in range(2):
            top_model.add(Dense(4096, activation='relu'))
            top_model.add(Dropout(dropout))
        top_model.add(Dense(outclass, activation=sigma))

        model = None
        model = Model(inputs=base_model.input, outputs=top_model(base_model.output))
        
        return model

    def train(self, train_data_dir: str = 'archive/data/train', validation_data_dir: str ='archive/data/val', training_batch_size: int = 30, validation_batch_size: int = 30, model: str = 'vgg16CNNtl', dropout: float = 0.5, epochs: int = 50, learning_rate: float = 1e-5, decay: float = 1e-7, test_data_aug: bool = False, save_dir: str = 'archive/data_aug_images'):
        """[summary]

        Args:
            train_data_dir (str, optional): [description]. Defaults to 'archive/data/train'.
            validation_data_dir (str, optional): [description]. Defaults to 'archive/data/val'.
            training_batch_size (int, optional): [description]. Defaults to 30.
            validation_batch_size (int, optional): [description]. Defaults to 30.
            model (str, optional): [description]. Defaults to 'vgg16CNNtl'.
            dropout (float, optional): [description]. Defaults to 0.5.
            epochs (int, optional): [description]. Defaults to 50.
            learning_rate (float, optional): [description]. Defaults to 1e-5.
            decay (float, optional): [description]. Defaults to 1e-7.
            test_data_aug (bool, optional): [description]. Defaults to False.
            save_dir (str, optional): [description]. Defaults to 'archive/data_aug_images'.

        Returns:
            [type]: [description]
        """
        # The train_datagen corresponds to an augmentation tool which will enable us to generate
        # images for our training dataset according to the configuration set.
        train_datagen = ImageDataGenerator(rescale=1./255, # rescale enables us to normalize the images
                        rotation_range=10,  # rotation_range randomly rotate images in the range between 0 and 10 degrees
                        zoom_range = 0.1, # zoom_range zooms the images in the range from 0 to 0.1
                        width_shift_range=0.1,  # width_shift_range randomly shift images horizontally (fraction of total width)
                        height_shift_range=0.1,  # height_shift_range randomly shift images vertically (fraction of total height)
                        vertical_flip=False, # vertical_flip allows us to unenable the flip of the image in the vertical axis
                        horizontal_flip=True) # horizontal_flip allows us to enable the flip of the image in the horizontal axis
        # The test_datagen corresponds to another augmentation tool which will enable us to generate 
        # images for our validation dataset according to the configuration set.
        # However, for this case only normalization of the images will be used
        test_datagen = ImageDataGenerator(rescale=1./255)

        # Directorio de trabajo
        cwd = os.getcwd()
        # Creamos el directorio de las imágenes generadas, en caso de que no exista
        save_dir=os.path.join(cwd,save_dir)
        train_data_dir=os.path.join(cwd,train_data_dir)
        validation_data_dir=os.path.join(cwd,validation_data_dir)
        if os.path.isdir(save_dir) == False:
            os.makedirs(save_dir)

        # Si solo se quiere realizar data augmentation para ver como funciona
        if test_data_aug == True:
            train_generator = train_datagen.flow_from_directory(
                # This is the target directory
                train_data_dir,
                # All images will be resized to 200x200
                target_size=(200, 200),
                # The size of the batches will be 30: the number of samples that will be propagated through the network
                batch_size=training_batch_size,
                # The class mode will be binary: 1D numpy array of binary labels
                shuffle=True,class_mode='categorical',save_to_dir=save_dir,save_format="jpg")

            batch=next(train_generator) # returns the next batch of images and labels
            return batch
        # Si se quiere entrenar el modelo
        else:

            train_generator = train_datagen.flow_from_directory(
                # This is the target directory
                train_data_dir,
                # All images will be resized to 200x200
                target_size=(200, 200),
                # The size of the batches will be 30: the number of samples that will be propagated through the network
                batch_size=training_batch_size,
                # The class mode will be binary: 1D numpy array of binary labels
                shuffle=True,class_mode='categorical')

            validation_generator = test_datagen.flow_from_directory(
                # This is the target directory
                validation_data_dir,
                # All images will be resized to 150x150
                target_size=(200,200),
                # The size of the batches will be 30: the number of samples that will be propagated through the network
                batch_size=validation_batch_size,
                # The class mode will be binary: 1D numpy array of binary labels
                class_mode='categorical')

            batch=next(train_generator) # returns the next batch of images and labels

            img_width, img_height = 200,200
            if K.image_data_format() == 'channels_first':
                input_shape = (3, img_width, img_height)
            else:
                input_shape = (img_width, img_height, 3)
            numclasses=batch[1].shape[1]

            if model == 'vgg16CNNtl':
                self._model = self._vgg16CNNtl(input_shape, numclasses, 'softmax', dropout)
                optimizer = RMSprop(lr=learning_rate, decay=decay)
                self._model.compile(loss='categorical_crossentropy',  
                            optimizer=optimizer,
                            metrics=['accuracy'])

                #Model fit 
                history = self._model.fit(train_generator, 
                    steps_per_epoch = train_generator.samples // training_batch_size, 
                    epochs=epochs,
                    validation_data=validation_generator,
                    validation_steps=validation_generator.samples // validation_batch_size)
        
        sns.set(font_scale=2)
        # Get training and test loss histories
        training_loss = history.history['loss']
        training_acc = history.history['accuracy']

        # Create count of the number of epochs
        epoch_count = range(1, len(training_loss) + 1)

        fig=plt.figure(figsize=(20, 10))
        # Visualize loss history
        fig.add_subplot(121)
        sns.lineplot(epoch_count, training_loss)
        sns.lineplot(epoch_count, training_acc)
        plt.legend(['Training Loss', 'Training Accuracy'])
        plt.xlabel('Epoch')
        plt.ylabel('Training Loss/Acc')
        plt.title('Training Loss/Accuracy vs Epoch',weight='bold')

        # Get training and test loss histories
        val_acc = history.history['val_accuracy']
        training_acc = history.history['accuracy']

        # Create count of the number of epochs
        epoch_count = range(1, len(val_acc) + 1)

        # Visualize loss history
        fig.add_subplot(122)
        sns.lineplot(epoch_count, val_acc)
        sns.lineplot(epoch_count, training_acc)
        plt.legend(['Validation Accuracy', 'Training Accuracy'])
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Accuracy vs Epoch',weight='bold')
        plt.savefig(model +'_dropout'+ str(dropout).replace(".","")+'_history.png')

    def save(self, filename: str):
        """Saves the model to an .h5 file and the model name to a .json file.

        Args:
           filename: Relative path to the file without the extension.

        """
        # Save Keras model
        self._model.save(filename + '.h5')
    
    def test(self, model, train_data_dir: str = 'archive/data/train', validation_data_dir: str = 'archive/data/val'):
        """[summary]

        Args:
            model ([type]): [description]
            train_data_dir (str, optional): [description]. Defaults to 'archive/data/train'.
            validation_data_dir (str, optional): [description]. Defaults to 'archive/data/val'.
        """
        labels = os.listdir(train_data_dir)
        test_images=[]
        for root, dirs, files in os.walk(validation_data_dir):
            for name in files:
                test_images.append(root+'/'+name)
                # print(test_images)
        test_imgs=np.random.choice(test_images,8)
        for test_img in test_imgs:
            fig, ax = plt.subplots()
            # print(test_img)
            img = image.load_img(test_img, target_size=(200, 200))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x /= 255.
            classes = model.predict(x)
            result = np.squeeze(classes)
            result_indices = np.argmax(result)
            
            img = cv2.imread(test_img, cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            ax.axis('off')
            plt.title("{}, {:.2f}%".format(labels[result_indices], result[result_indices]*100),size=16,weight='bold')
            ax.imshow(img)


    def predict(self, test_dir: str, dataset_name: str = "", save: bool = True):
        """Evaluates a new set of images using the trained CNN.

        Args:
            test_dir: Relative path to the validation directory (e.g., 'dataset/test').
            dataset_name: Dataset descriptive name.
            save: Save results to an Excel file.

        """
        # Configure loading and pre-processing functions
        print('Reading test data...')
        test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(preprocessing_function=self._preprocessing_function)

        test_generator = test_datagen.flow_from_directory(
            test_dir,
            target_size=self._target_size,
            batch_size=1,  # A batch size of 1 ensures that all test images are processed
            class_mode='categorical',
            shuffle=False
        )

        # Predict categories
        predictions = self._model.predict(test_generator)
        predicted_labels = np.argmax(predictions, axis=1).ravel().tolist()

        # Format results and compute classification statistics
        results = Results(test_generator.class_indices, dataset_name=dataset_name)
        accuracy, confusion_matrix, classification = results.compute(test_generator.filenames, test_generator.classes,
                                                                     predicted_labels)
        # Display and save results
        results.print(accuracy, confusion_matrix)

        if save:
            results.save(confusion_matrix, classification, predictions)