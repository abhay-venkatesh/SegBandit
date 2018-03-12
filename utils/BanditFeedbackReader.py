import numpy as np
import cv2
import pickle
import csv
import os
import random

class BanditFeedbackReader:
    """ 
        Serializes Logged Bandit Feedback for the form
        D = ((x1, y1, d1, p1), ..., (xn, yn, dn, pn))
    """
    def __init__(self, feedback_dir, current_step, batch_size=5):

        # Save variables
        self.batch_size = batch_size
        self.current_step = current_step
        self.feedback_dir = feedback_dir

        # Get meta information
        with open(self.feedback_dir + "meta", "r") as infile:
            reader = csv.reader(infile, delimiter=",")
            for row in reader:
                if row[0] == "size":
                    self.dataset_size = int(row[1])

        """
        Say we have 100 training images
        Say we are at training step 10
        Then, our index in our training images should be 
        (10 * 5) % 100 = 50
        This corresponds to, 
        """
        self.train_index = (current_step * self.batch_size) % self.dataset_size

        """
        Load the appropriate logged bandit feedback file
        """
        self.log_file_number = int(self.train_index/1000) + 1
        log_file_path = self.feedback_dir + 'log-' + str(self.log_file_number)
        with open(log_file_path, 'rb') as fp:
            self.logged_data = pickle.load(fp)

    def shuffle_training_data(self):

        # Shuffle the log file and write to disk
        lines = self.logged_data 
        random.shuffle(lines)
        log_file_path = self.feedback_dir + 'log-' + str(self.log_file_number)
        with open(log_file_path, 'wb') as fp:
            pickle.dump(lines, fp)

        # Proceed to next log file
        self.log_file_number += 1
        log_file_path = self.feedback_dir + 'log-' + str(self.log_file_number)
        if os.path.exists(log_file_path):
            with open(log_file_path, 'rb') as fp:
                self.logged_data = pickle.load(fp)
        else:
            # Reset if next log file does not exist
            self.train_index = 0
            self.log_file_number = 1
            log_file_path = (self.feedback_dir + 'log-' + 
                             str(self.log_file_number))
            with open(log_file_path, 'rb') as fp:
                self.logged_data = pickle.load(fp)


    def next_item_batch(self):
        """
            Returns loss, propensity
        """
        propensities = []
        deltas = []

        for i in range(self.batch_size):

            if self.train_index == 0:
                self.shuffle_training_data()

            # First, we get the feedback item from our training data
            feedback_tuple = self.logged_data[self.train_index]

            # Update training index
            self.train_index += 1
            self.train_index %= self.dataset_size

            deltas.append(feedback_tuple[1])
            propensities.append(feedback_tuple[2])
            
        return deltas, propensities

if __name__ == "__main__":
    bfr = BanditFeedbackReader("./logged_bandit_feedback/", 0)
    loss, propensity = bfr.next_item()
    print(loss)
    print(propensity)