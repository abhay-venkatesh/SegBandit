from lib.SegNetLogger import SegNetLogger
from lib.BanditSegNet import BanditSegNet

def main():
    dataset_directory = './datasets/Unreal-20View-11class/view0/'
    feedback_directory = './logged_bandit_feedback/'

    # net = SegNetLogger()
    # net.train(dataset_directory, num_iterations=100000, 
    #           learning_rate=1e-2, batch_size=5)
    # net.build_contextual_feedback_log(dataset_directory)
    net = BanditSegNet()
    net.train(dataset_directory, feedback_directory, lagrange=0.9, 
    		  num_iterations=100000, learning_rate=1e-2, batch_size=5)

if __name__ == "__main__":
    main()
