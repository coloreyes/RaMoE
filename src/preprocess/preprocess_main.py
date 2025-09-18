import os
os.chdir(os.path.abspath(os.path.dirname(__file__)))
from argparse import ArgumentParser
from utils import get_logger
import pandas as pd
from datasets_class import ICIPDataset, SMPDDataset, INSDataset, UGCDataset
from file_path import FilePath
import time
from typing import Type
class Preprocess:

    def __init__(self, args):
        self.dataset = args.dataset    
        self.datasets_path = os.path.join(args.datasets_path, args.dataset)
        self.model_path = args.pretrained_model_path
        self.retrieval_num = args.retrieval_num
        self.logger = get_logger(FilePath.get_preprocess_log_path())
        self.device = args.device
        self.sample_ratio = args.sample_ratio

    def preprocess(self):
        dataset_class = self.get_dataset_class()
        if dataset_class is None:
            self.logger.error("Error dataset name, dataset must be in [ICIP, SMPD, INS].")
            return

        dataset = dataset_class(
            dataset=self.dataset,
            datasets_path=self.datasets_path,
            logger=self.logger,
            sample_ratio=self.sample_ratio
        )

        self.logger.info("=================== Start loading original data ===================")
        self._time_log(dataset.load_origin_data, "Load origin data")

        self.logger.info("=================== Start transforming data ===================")
        self._time_log(
            lambda: dataset.transform_dataset(
                self.model_path, FilePath.get_image_path(self.dataset, self.datasets_path), self.device
            ),
            "Transform origin data"
        )

        self.logger.info("=================== Start retrieval ===================")
        self._time_log(lambda: dataset.retrival(self.retrieval_num), "Retrieval")

    def _time_log(self, func, description):
        start = time.time()
        func()
        end = time.time()
        elapsed = end - start
        self.logger.info(f"{description} cost time: {int(elapsed // 3600)}h {int((elapsed % 3600) // 60)}min {elapsed % 60:.2f}s")

    def get_dataset_class(self) -> Type[UGCDataset]:
        try:
            return {
                "ICIP": ICIPDataset,
                "SMPD": SMPDDataset,
                "INS": INSDataset
            }.get(self.dataset)
        except Exception as e:
            self.logger.error(f"Error dataset name: {self.dataset}, dataset must be in [ICIP, SMPD, INS].")
            raise
    
if __name__ == '__main__':
    parser = ArgumentParser(description='parameters')
    parser.add_argument("--dataset", default="INS", type=str, choices=["ICIP","SMPD","INS"], required=False)
    parser.add_argument("--datasets_path", default="../../datasets", type=str,required=False)
    parser.add_argument("--pretrained_model_path", default="../../model", type=str,required=False)
    parser.add_argument("--retrieval_num", default="500", type=int,required=False)
    parser.add_argument("--device", default="cuda:0", type=str,required=False)
    parser.add_argument("--sample_ratio", default=1, type=float)
    parser.add_argument("--postfix", type=str, default="")
    args =  parser.parse_args()    

    preprocess = Preprocess(args)
    preprocess.preprocess()

