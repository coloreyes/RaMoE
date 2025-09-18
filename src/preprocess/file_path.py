import os 
from utils import get_timestamp
class FilePath:
    
    @staticmethod
    def preprocessed_origin_data_path(datasets_path: str) -> str:
        return f"{datasets_path}/preprocessed_origin_data_single.pkl"

    @staticmethod
    def get_dataset_path(dataset:str, datasets_path:str) -> tuple:
            if dataset == "ICIP":
                users_path = os.path.join(datasets_path, 'origin_data', 'users_TRAIN.csv')
                headers_path = os.path.join(datasets_path, 'origin_data', 'headers_TRAIN.csv')
                img_info_path = os.path.join(datasets_path, 'origin_data', 'img_info_TRAIN.csv')
                popularity_path = os.path.join(datasets_path, 'origin_data', 'popularity_TRAIN.csv')
                return users_path,headers_path,img_info_path,popularity_path
            elif dataset == "SMPD":
                train_additional_information_path = os.path.join(datasets_path, 'origin_data', 'train_additional_information.json')
                train_category_path = os.path.join(datasets_path, 'origin_data', 'train_category.json')
                train_temporalspatial_information_path = os.path.join(datasets_path, 'origin_data', 'train_temporalspatial_information.json')
                train_user_data_path = os.path.join(datasets_path, 'origin_data', 'train_user_data.json')
                label_data_path = os.path.join(datasets_path, 'origin_data', 'train_label.txt')
                train_text_path = os.path.join(datasets_path, 'origin_data', 'train_text.json')
                return train_additional_information_path, train_category_path, train_temporalspatial_information_path, train_user_data_path, label_data_path, train_text_path
            elif dataset == "INS":
                meta_data_path = os.path.join(datasets_path, 'origin_data', 'post_info.txt')
                json_path = os.path.join(datasets_path, 'origin_data', 'json')
                return meta_data_path,json_path
            else:
                raise ValueError("Error dataset name,dataset must in [ICIP,SMPD,INS].")

    @staticmethod
    def get_dataset_save_name(dataset: str, datasets_path: str) -> tuple:
        return os.path.join(datasets_path, f'dataset_{dataset}.pkl')

    @staticmethod
    def get_split_path(datasets_path: str):
        return os.path.join(datasets_path, 'train.pkl'), os.path.join(datasets_path, 'valid.pkl'), os.path.join(datasets_path,'test.pkl')

    @staticmethod
    def get_dissembled_path(datasets_path: str) -> tuple:
        dissembled_path = os.path.join(datasets_path, "dissembled")
        if not os.path.exists(dissembled_path):
            os.makedirs(dissembled_path)
        return os.path.join(dissembled_path,'train.pkl'), os.path.join(dissembled_path,'valid.pkl'), os.path.join(dissembled_path,'test.pkl')
    
    @staticmethod    
    def get_preprocess_log_path(root_path: str = None) -> str:
        """
        Return the file path for the preprocessing log.

        :param root_path: The root directory path. If None, the current working directory will be used.
        :return: The absolute path to the log file.
        """
        if root_path is None:
            root_path = os.getcwd()
        log_dir = os.path.join(root_path, "preprocess_log")
        os.makedirs(log_dir, exist_ok=True)
        filename = f"log_{get_timestamp()}.log"
        return os.path.join(log_dir, filename)
    
    @staticmethod
    def get_image_path(dataset: str, datasets_path: str) -> str:
        if dataset == "ICIP":
            return os.path.join(datasets_path, "origin_data", "pic")
        if dataset == "SMPD":
            return os.path.join(datasets_path, "origin_data", "pic")
        if dataset == "INS":
            return os.path.join(datasets_path, "origin_data", "img")
        else:
            raise ValueError("Error dataset name, dataset must in [ICIP,SMPD,INS].")
        
    @staticmethod        
    def get_transform_data_path(datasets_path: str) -> str:
        return os.path.join(datasets_path, "transform_data_single.pkl")