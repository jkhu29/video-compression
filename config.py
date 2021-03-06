import argparse


def get_options(parser=argparse.ArgumentParser()):
    parser.add_argument('--output_dir', type=str, default="output")
    parser.add_argument('--workers', type=int, default=4, help='number of data loading workers, you had better put it '
                                                               '4 times of your gpu')
    parser.add_argument('--batch_size', type=int, default=64, help='input batch size, default=64')
    parser.add_argument('--niter', type=int, default=10, help='number of epochs to train for, default=10')
    parser.add_argument('--lr', type=float, default=5e-5, help='select the learning rate, default=5e-5')
    parser.add_argument('--seed', type=int, default=118, help="random seed")
    opt = parser.parse_args()

    return opt
