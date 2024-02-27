"""by lyuwenyu
"""

import os 
import sys 
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import argparse

import src.misc.dist as dist 
from src.core import YAMLConfig 
from src.solver import TASKS

from clearml import Dataset


def main(args, ) -> None:
    '''main
    '''
    dist.init_distributed()

    assert not all([args.tuning, args.resume]), \
        'Only support from_scrach or resume or tuning at one time'

    cfg = YAMLConfig(
        args.config,
        args.epoches,
        resume=args.resume, 
        use_amp=args.amp,
        tuning=args.tuning
    )

    solver = TASKS[cfg.yaml_cfg['task']](cfg)
    
    if args.test_only:
        solver.val()
    else:
        solver.fit()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str, )
    parser.add_argument('--resume', '-r', type=str, )
    parser.add_argument('--tuning', '-t', type=str, )
    parser.add_argument('--test-only', action='store_true', default=False,)
    parser.add_argument('--use-cloud-dataset', action='store_true', default=False, )
    parser.add_argument('--amp', action='store_true', default=False,)
    parser.add_argument('--epoches', '-e', type=int, default=1)


    parser.add_argument('--dataset-name', '-d', type=str, default="soccer_6k_single")
    parser.add_argument('--dataset-project', '-p', type=str, default="Pytorch Test")


    args = parser.parse_args()

    if args.use_cloud_dataset:
        dataset_path = Dataset.get(
            dataset_name=args.dataset_name,
            dataset_project=args.dataset_project,
            alias="Soccer 6k dataset"
        ).get_local_copy()
        os.symlink(dataset_path, "dataset")

    main(args)
