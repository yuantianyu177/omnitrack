import configargparse

def config_parser(dafault_config=""):
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True,
                        help='config file path', default=dafault_config)

    args = parser.parse_args()
    return args
