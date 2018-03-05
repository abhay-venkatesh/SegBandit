import numpy as np
import cv2

class BanditFeedbackReader:
    """ 
        Serializes Logged Bandit Feedback for the form
        D = ((x1, y1, d1, p1), ..., (xn, yn, dn, pn))
    """
    def __init__(self, feedback_dir, dataset_dir, current_step):

        # Save variables
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
        Say we have 100 training images
        Say we are at training step 10
        Then, our index in our training images should be 
        (10 * 5) % 100 = 50
        This corresponds to, 
        """
        self.train_index = current_step % self.dataset_size

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

        self.logged_feedback = open(self.directory + 'train.txt').readlines()


    def shuffle_training_data(self):
        lines = open(self.feedback_dir + 'record').readlines()
        random.shuffle(lines)
        open(self.feedback_dir + 'record', 'w').writelines(lines)
        self.logged_feedback = open(self.feedback_dir + 'record').readlines()

    def next_item(self):
        """
            Returns next 4-tuple with (x,y,d,p) data

            Shuffle record file on making one full-pass of dataset.

        """

        if self.train_index == 0:
            self.shuffle_training_data()

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

        # First, we get the feedback item from our training data
        feedback_item_i = self.logged_feedback[self.train_index]

        # Now, we have to fetch the image from the dataset
        images = np.load(open(self.feedback_dir + "x", 'rb'))
        image_file_name = images[feedback_item_i][1]
        image_directory = self.dataset_dir + 'images/'
        image = cv2.imread(image_directory + image_file)
        image = np.float32(image)

        # Load ground truth
        ground_truth_directory = self.directory + 'ground_truths/'
        ground_truth_file = image_file.replace('pic', 'seg')
        ground_truth = cv2.imread((ground_truth_directory + 
                                   ground_truth_file), cv2.IMREAD_GRAYSCALE)
        ground_truth = ground_truth/8

        # Update training index
        self.train_index += 1
        self.train_index %= self.dataset_size
            
        return images, ground_truths


