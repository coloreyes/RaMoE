import os
os.chdir(os.path.abspath(os.path.dirname(__file__)))
from argparse import ArgumentParser
from experiment import Experiment, ExperimentConfig
if __name__ == '__main__':

    parser = ArgumentParser(description='Training and Evaluation Parameters')

    parser.add_argument("--dataset", default="INS", type=str,
                        choices=["ICIP", "SMPD", "INS"],
                        help="Dataset to use.")
    parser.add_argument("--datasets_path", default="../datasets", type=str,
                        help="Path to datasets directory.")
    parser.add_argument("--mode", default="train", type=str,
                        choices=["train", "test"],
                        help="Run mode: train or test.")
    parser.add_argument("--save", default="Result", type=str,
                        help="Directory to save results.")
    parser.add_argument("--seed", default=2025, type=int,
                        help="Random seed.")

    parser.add_argument("--moe_model", default="ACMoE", type=str,
                        choices=[
                            "MultiMoe",
                            "ACMoE",
                            "ACMoEWOR"
                        ],
                        help="Model type to use.")
    parser.add_argument("--num_experts", default=3, type=int,
                        help="Number of experts in the MoE model.")
    parser.add_argument("--retrieval_num", default=500, type=int,
                        help="Number of retrieved items.")

    parser.add_argument("--epochs", default=1000, type=int,
                        help="Max number of training epochs.")
    parser.add_argument("--batch_size", default=64, type=int,
                        help="Training batch size.")
    parser.add_argument("--early_stop_turns", default=5, type=int,
                        help="Early stop patience.")
    parser.add_argument("--loss", default="MSE", type=str,
                        choices=["BCE", "MSE"],
                        help="Loss function.")
    parser.add_argument("--optim", default="Adam", type=str,
                        choices=["SGD", "Adam"],
                        help="Optimizer to use.")
    parser.add_argument("--lr", default=1e-4, type=float,
                        help="Learning rate.")

    parser.add_argument("--device", default="cuda:1",
                        help="Device to use for computation.")

    parser.add_argument("--modalities", nargs='+', default=["image", "user", "text"],
                        choices=["text", "image", "user"],
                        help="Modalities to use. e.g.: --modalities text image")

    parser.add_argument("--test_model", default='', type=str,
                        help="Path to the model checkpoint for testing.")
    parser.add_argument("--NOTE", default='TwoModal[user+vision]', type=str,
                        help="Additional notes.")
    parser.add_argument("--use_decoupler", action='store_true',
                        help="Whether to use the decoupler module.")
    parser.add_argument("--use_CL", action='store_true',
                        help="Whether to use the CL module.(only between image and text)")
    parser.add_argument("--normalize", action='store_true',
                        help="Normalize label.")
    parser.add_argument("--metric", type=lambda s: [x.strip() for x in s.split(',')],
                        default=['MSE', 'SRC', 'MAE'],
                        help="Comma-separated list of metrics, e.g.: 'MSE,SRC,MAE'")
    parser.add_argument("--sample_ratio",type=float,
        default=1.0,
        help="Sampling ratio of the dataset. Value in (0,1]. Default is 1.0 (use the entire dataset)."
    )
    parser.add_argument("--dataset_postfix", type=str, default="")
    args = parser.parse_args()

    config = ExperimentConfig(args)
    experiment = Experiment(config)
    experiment.run()