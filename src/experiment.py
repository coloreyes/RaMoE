import os
os.chdir(os.path.abspath(os.path.dirname(__file__)))
import torch
from argparse import Namespace
from utils import setup_logging, create_save_folders, delete_model, set_seed
from model.RaMoE import RaMoE
from dataset import UGCDataset, custom_collate_fn
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch.nn.parameter import UninitializedParameter
import random
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import spearmanr
from tqdm import tqdm
from logging import Logger
MODEL_REGISTRY = {
    "RaMoE": RaMoE
}

class ExperimentConfig:
    def __init__(self, args: Namespace):
        self.dataset = args.dataset
        self.datasets_path = args.datasets_path
        self.seed = args.seed
        self.retrieval_num = args.retrieval_num
        self.mode = args.mode
        self.batch_size = args.batch_size
        self.loss = args.loss
        self.device = args.device
        self.optim = args.optim
        self.lr = args.lr
        self.save = args.save
        self.early_stop_turns = args.early_stop_turns
        self.metric = args.metric
        self.epochs = args.epochs
        self.test_model = args.test_model
        self.num_experts = args.num_experts
        self.moe_model = args.moe_model
        self.modalities = set([s.strip().lower() for s in args.modalities])
        self.normalize = args.normalize
        self.sample_ratio = args.sample_ratio
        self.dataset_postfix = args.dataset_postfix
    def to_log(self, logger: Logger):
        for k, v in vars(self).items():
            logger.info(f"{k}: {v}")


class Experiment:
    def __init__(self, config: ExperimentConfig):
        self.config = config
        set_seed(self.config.seed)

    def run(self):
        if self.config.mode == "train":
            self.train()
        elif self.config.mode == "test":
            self.test()
        else:
            raise ValueError(f"Invalid mode: {self.config.mode}")
    
    def train(self):
        father_folder, folder = create_save_folders(
            mode="train", 
            dataset=self.config.dataset, 
            num_experts=self.config.num_experts, 
            moe_model=self.config.moe_model, 
            save_root=self.config.save
        )
        logger = setup_logging(father_folder, folder)
        logger.info("========= Experiment Configuration =========")
        self.config.to_log(logger=logger)
        logger.info("=============================================")

        train_loader, valid_loader = self._load_data()
        model, optim = self._initialize_model_and_optimizer()
        min_valid_loss = float("inf")
        min_turn = 0

        for epoch in range(self.config.epochs):
            logger.info(f"-------- Epoch {epoch + 1} Start --------")
            (   
                avg_train_loss, 
                avg_val_loss, 
                avg_val_mse, 
                avg_val_mae, 
                avg_val_src
            ) = self.run_one_epoch(
                model, 
                optim, 
                train_loader, 
                valid_loader, 
                epoch
            )
            
            logger.info(f"Epoch {epoch+1} Summary:")
            logger.info(f"  Train Loss: {avg_train_loss:.4f}")
            logger.info(f"  Val   Loss: {avg_val_loss:.4f}")
            logger.info(f"  Val   MSE: {avg_val_mse:.4f},  MAE: {avg_val_mae:.4f}, SRC: {avg_val_src:.4f}")

            if avg_val_loss < min_valid_loss:
                min_valid_loss = avg_val_loss
                min_turn = epoch + 1
            logger.critical(f"Best Loss at Epoch {min_turn}: {min_valid_loss}")

            torch.save({"model_state_dict": model.state_dict()},
                       os.path.join(father_folder, folder, f"checkpoint_{epoch + 1}_epoch.pkl"))

            if (epoch + 1) - min_turn > self.config.early_stop_turns:
                break

        delete_model(father_folder, folder, min_turn)
        self.config.test_model = os.path.join(father_folder, folder, f"checkpoint_{min_turn}_epoch.pkl")
        self.test()
    
    def test(self):
        father_folder, folder = create_save_folders(
            mode="test",
            dataset=self.config.dataset,
            num_experts=self.config.num_experts,
            moe_model=self.config.moe_model,
            save_root=self.config.save
        )
        logger = setup_logging(father_folder, folder)
        logger.info("========= Experiment Configuration =========")
        self.config.to_log(logger=logger)
        logger.info("=============================================")

        model, _ = self._initialize_model_and_optimizer()
        torch.serialization.add_safe_globals([UninitializedParameter])
        checkpoint = torch.load(self.config.test_model, map_location=self.config.device, weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"])

        loader = self._load_data(is_train=False)

        model.eval()
        all_labels = []
        all_preds_dict = {} 

        with torch.no_grad():
            for batch in tqdm(loader, desc="Testing"):
                batch = [b.to(self.config.device) if isinstance(b, torch.Tensor) else b for b in batch]
                output, label = self._forward_modalities(model, batch)
                label = label.cpu().numpy()
                all_labels.append(label)

                # 初始化预测容器
                if not all_preds_dict:
                    all_preds_dict = {"mixture": []}
                    for m in output.get("modal_predictions", {}).keys():
                        all_preds_dict[m] = []

                # 记录预测值
                all_preds_dict["mixture"].append(output["mixture_prediction"].cpu().numpy())
                for m, pred_tensor in output.get("modal_predictions", {}).items():
                    all_preds_dict[m].append(pred_tensor.cpu().numpy())

        # 拼接所有标签
        all_labels = np.concatenate(all_labels, axis=0)

        logger.info(f"Test Model: {self.config.test_model}")
        for m, pred_list in all_preds_dict.items():
            preds = np.concatenate(pred_list, axis=0)
            mae = mean_absolute_error(all_labels, preds)
            mse = mean_squared_error(all_labels, preds)
            src, _ = spearmanr(all_labels, preds)
            if np.isnan(src):
                src = 0.0
            logger.info(
                f"[{m}] MSE: {mse:.4f}, MAE: {mae:.4f}, SRC: {src:.4f}"
            )
        logger.info("Test ended.")

    def _forward_modalities(self, model, batch):
        (
            mean_pooling_vec, merge_text_vec, user_vec,
            retrieved_visual_feature_embedding_cls,
            retrieved_textual_feature_embedding,
            retrieved_user_feature_embedding,
            retrieved_item_similarity,
            retrieved_label_list,
            label
        ) = batch

        input_dict = {}
        if "image" in self.config.modalities:
            input_dict["image"] = (
                mean_pooling_vec,
                retrieved_visual_feature_embedding_cls
            )
        if "text" in self.config.modalities:
            input_dict["text"] = (
                merge_text_vec,
                retrieved_textual_feature_embedding
            )
        if "user" in self.config.modalities:
            input_dict["user"] = (
                user_vec,
                retrieved_user_feature_embedding
            )
        output = model(input_dict, retrieved_item_similarity, retrieved_label_list)
        return output, label.float()

    def _load_data(self, is_train=True):
        if is_train:
            train = UGCDataset(
                self.config.retrieval_num, 
                os.path.join(
                    self.config.datasets_path, 
                    self.config.dataset, 
                    f"train{self.config.dataset_postfix}.pkl"
                ), 
                normalize=self.config.normalize)
            valid = UGCDataset(
                self.config.retrieval_num, 
                os.path.join(
                    self.config.datasets_path, 
                    self.config.dataset, 
                    f"valid{self.config.dataset_postfix}.pkl"
                ), 
                normalize=self.config.normalize
            )

            num_samples = int(len(train) * self.config.sample_ratio)
            sample_indices = random.sample(range(len(train)), num_samples)
            sampler = SubsetRandomSampler(sample_indices)

            train_loader = DataLoader(
                train,
                batch_size=self.config.batch_size,
                sampler=sampler,
                collate_fn=custom_collate_fn,
                shuffle=False
            )
            valid_loader = DataLoader(
                valid, 
                batch_size=self.config.batch_size, 
                collate_fn=custom_collate_fn
            )
            return train_loader, valid_loader
        else:
            test = UGCDataset(
                self.config.retrieval_num, 
                os.path.join(
                    self.config.datasets_path, 
                    self.config.dataset, 
                    f"test{self.config.dataset_postfix}.pkl"
                ), 
                normalize=self.config.normalize
            )
            return DataLoader(
                test, 
                batch_size=self.config.batch_size, 
                collate_fn=custom_collate_fn
            )

    def _initialize_model_and_optimizer(self):

        if self.config.moe_model not in MODEL_REGISTRY:
            raise ValueError(f"Invalid moe_model: {self.config.moe_model}")

        ModelClass = MODEL_REGISTRY[self.config.moe_model]
        model = ModelClass(
            modalities=list(self.config.modalities),
            retrieval_num=self.config.retrieval_num,
            rerank_experts=self.config.num_experts
        )
        # 移到设备
        model = model.to(self.config.device)
        optim = torch.optim.Adam(model.parameters(), lr=self.config.lr)
        return model, optim

    def run_one_epoch(self, model, optim, train_loader, valid_loader, epoch):
        model.train()
        total_train_loss = 0.0
        total_train_samples = 0

        for batch in tqdm(train_loader, desc=f"Training Epoch {epoch+1}"):

            batch = [b.to(self.config.device) if isinstance(b, torch.Tensor) else b for b in batch]
            output, label = self._forward_modalities(model, batch)

            joint_loss, mixture_loss = model.calculate_loss(**output, label=label, epoch=epoch)

            optim.zero_grad()
            joint_loss.backward()
            optim.step()

            batch_size = label.size(0)
            total_train_loss += joint_loss.item() * batch_size
            total_train_samples += batch_size

        avg_train_loss = total_train_loss / total_train_samples if total_train_samples > 0 else float("inf")

        # ------------------ 验证 ------------------
        model.eval()
        total_val_loss = 0.0
        total_val_samples = 0
        val_preds = []
        val_labels = []

        with torch.no_grad():
            for batch in tqdm(valid_loader, desc=f"Validating Epoch {epoch+1}"):
                batch = [b.to(self.config.device) if isinstance(b, torch.Tensor) else b for b in batch]
                output, label = self._forward_modalities(model, batch)

                # loss
                # val_loss, _ = model.calculate_loss(**output, label=label, epoch=epoch)
                joint_loss, mixture_loss = model.calculate_loss(**output, label=label, epoch=epoch)
                batch_size = label.size(0)
                total_val_loss += joint_loss.item() * batch_size
                total_val_samples += batch_size

                pred = output["mixture_prediction"].cpu().numpy()
                label = label.cpu().numpy()
                val_preds.append(pred)
                val_labels.append(label)

        avg_val_loss = total_val_loss / total_val_samples if total_val_samples > 0 else float("inf")
        val_preds = np.concatenate(val_preds, axis=0)
        val_labels = np.concatenate(val_labels, axis=0)
        val_mae = mean_absolute_error(val_labels, val_preds)
        val_mse = mean_squared_error(val_labels, val_preds)
        val_src, _ = spearmanr(val_labels, val_preds)
        val_src = val_src if not np.isnan(val_src) else 0.0

        return avg_train_loss, avg_val_loss, val_mse, val_mae, val_src