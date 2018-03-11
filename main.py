from lib.SegNetLogger import SegNetLogger

def main():
    dataset_directory = './datasets/Unreal-20View-11class/view0/'

    net = SegNetLogger()
    # net.train(dataset_directory, num_iterations=100000, 
    #           learning_rate=1e-2, batch_size=5)
    net.build_contextual_feedback_log(dataset_directory)

if __name__ == "__main__":
    main()
