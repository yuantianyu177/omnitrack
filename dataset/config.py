import configargparse


def config_parser(dafault_config=""):
    parser = configargparse.ArgumentParser()

    parser.add_argument('--data_dir', type=str,
                        default='/data/datasets/dataset/parkour')
    parser.add_argument('--num_imgs', type=int, default=10000)

    args = parser.parse_args()
    return args
