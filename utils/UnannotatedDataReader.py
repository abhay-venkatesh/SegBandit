import numpy as np
import cv2
import os
import random
from utils.ImageResizer import ImageResizer 
from utils.RecordFileGenerator import RecordFileGenerator
from random import randint

class UnannotatedDataReader:
    """ 
        Helper class to SegNet that handles data reading, conversion 
        and all things related to data 
    """

    def __init__(self, directory, current_step, batch_size):

        # Save variables
        self.batch_size = batch_size
        self.current_step = current_step
        self.directory = directory

        # Prepare dataset record files
        rfg = RecordFileGenerator(directory)
        self.num_train, self.num_val = rfg.create_trainval_only()

        # Say we have 100 training images
        # Say we are at training step 10
        # Then, our index in our training images should be 
        # (10 * 5) % 100 = 50
        # This corresponds to, 
        self.train_index = (current_step * self.batch_size) % self.num_train

        # Read dataset items
        self.training_data = open(directory + 'train.txt').readlines()
        self.validation_data = open(directory + 'val.txt').readlines()

    def shuffle_training_data(self):
        lines = open(self.directory + 'train.txt').readlines()
        random.shuffle(lines)
        open(self.directory + 'train.txt', 'w').writelines(lines)
        self.training_data = open(self.directory + 'train.txt').readlines()

    def next_training_batch(self):
        images = []
        ground_truths = []

        for i in range(self.batch_size):

            if self.train_index == 0:
                self.shuffle_training_data()

            # Load image
            image_directory = self.directory + 'images/'
            image_file = self.training_data[self.train_index].rstrip()
            image = cv2.imread(image_directory + image_file)
            image = np.float32(image)
            # With 0.5 probability, flip the image/ground truth pair
            # for data augmentation
            random_number = randint(0,1)
            if random_number == 1:
                image = np.fliplr(image)
            images.append(image)

            # Update training index
            self.train_index += 1
            self.train_index %= self.num_train
            
        return images

    def next_val_batch(self):
        images = []
        ground_truths = []

        for i in range(self.batch_size):
            # Load image
            image_directory = self.directory + 'images/'
            image_file = random.choice(self.validation_data).rstrip()
            image = cv2.imread(image_directory + image_file)
            image = np.float32(image)
            images.append(image)

        return images


def main():
    pass

if __name__ == "__main__":
    main()
