import os
import yaml
import pprint

from exatrkx import FeatureStore

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="build a feature storage")
    add_arg = parser.add_argument
    add_arg("--config", help="configuration file", default=None)
    add_arg("--input-dir", help="input directory", default=None)
    add_arg("--output-dir", help='output directory', default=None)
    add_arg("--detector-path", help='detector file path', default=None)
    add_arg("--n-files", help='number of files to process', default=None, type=int)
    add_arg('--n-workers', help='number of workers/threads', default=None, type=int)
    add_arg("--pt-min", help='minimum pT', default=None, type=float)

    args = parser.parse_args()

    if args.config is None or not os.path.exists(args.config):
        print("missing configuration, using default")
        import pkg_resources
        config_file = pkg_resources.resource_filename("exatrkx", 'configs/prepare_feature_store.yaml')
    else:
        config_file = args.config

    pp = pprint.PrettyPrinter(indent=4, sort_dicts=False)

    with open(config_file) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    for key,value in args.__dict__.items():
        if key in config and value is not None:
            config[key] = value

    pp.pprint(config)
    ctn = input("Continue? [y/n]: ")
    print(ctn)
    if ctn.lower() == "y":
        preprocess_dm = FeatureStore(config)
        preprocess_dm.prepare_data()
    else:
        parser.print_help()
        print("nothing is done. bye.")