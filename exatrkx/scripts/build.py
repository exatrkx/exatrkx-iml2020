import os
import yaml
from exatrkx import FeatureStore
import pprint


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="build a feature storage")
    add_arg = parser.add_argument
    add_arg("config", help="configuration file", default=None)

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

    pp.pprint(config)
    # preprocess_dm = FeatureStore(config)
    # preprocess_dm.prepare_data()