import numpy as np
import cv2
import os
import random
from random import randint

class BanditFeedbackReader:
    """ 
        Serializes Logged Bandit Feedback for the form
        D = ((x1, y1, d1, p1), ..., (xn, yn, dn, pn))
    """
    def __init__(self, feedback_dir, dataset_dir, current_step, batch_size):

        # Save variables
        self.batch_size = batch_size
        self.current_step = current_step
        self.feedback_dir = feedback_dir
        self.dataset_dir = dataset_dir

        # Get meta information
        with open(self.feedback_dir + "meta", "r") as infile:
            reader = csv.reader(infile, delimiter=",")
            for row in reader:
                if row[0] == "size":
                    self.dataset_size = int(row[1])

        """
            Creates a record file of form:
            0
            1
            2
            .
            .
            .
            dataset_size - 1
        """
        with open(self.feedback_dir + "record", "w") as outfile:
            for i in range(self.dataset_size):
                outfile.write(i)

        """
            We have three types of logging information:
                1. Image
                2. Segmentation
                3. Feedback/Loss (Delta)
                4. Propensities (Probability distribution at theta = 0)
         
            Logging data is available in the following format:
                x: Numpy array object with tuples of the form (i, image_file_name),
                    where i is the index, and image_file_name is the name of file
                    in the Unreal-20View-11class dataset 
                y(k): Segmentation images, stored as images
                d: Numpy array object with tuples of the form (i, loss)
                p: Numpy array object with tuples of the form (i, propensity)
        """

    def next_item(self):
        """
            Returns next 4-tuple with (x,y,d,p) data

            Shuffle record file on making one full-pass of dataset.

        """
        pass


