import pandas as pd 
import numpy as np
import torch.nn as nn
import torch
import math
import re
from tqdm import tqdm
from collections import defaultdict
from file_path import FilePath
from transform_data import TransformData
from sklearn.model_selection import train_test_split
import json
import os
import logging
class UGCDataset:
    def __init__(self, dataset: str, datasets_path: str, logger: logging, sample_ratio: float):
        self.datasets_path = datasets_path
        self.dataset = dataset
        self.logger = logger
        self.sample_ratio = sample_ratio

    def load_origin_data(self):
        preprocessed_origin_data_path = FilePath.preprocessed_origin_data_path(self.datasets_path)
        if os.path.exists(preprocessed_origin_data_path):
            self.logger.info("Load preprocessed origin data...")
            self.load_preprocessed_origin_data(preprocessed_origin_data_path)
        else:
            self.logger.info("Preprocessing origin data...")
            self.preprocess_origin_data()
            self.save_preprocessed_origin_data(preprocessed_origin_data_path)

    def load_preprocessed_origin_data(self):
        pass
    def preprocess_origin_data(self):
        pass
    def save_preprocessed_origin_data(self):
        pass
    def get_dataframe(self) -> pd.DataFrame:
        pass
    def calculate_similarity(self) -> tuple:
        pass
    def encode_user(self):
        pass
    def save_dataset(self): 
        save_path = FilePath.get_dataset_save_name(self.dataset, self.datasets_path)
        df = self.get_dataframe()
        df.to_pickle(save_path) 
    
    def transform_dataset(self, model_folder: str, image_folder: str, device: str):
        save_path = FilePath.get_transform_data_path(self.datasets_path)
        if not os.path.exists(save_path):
            transformer = TransformData(model_folder, image_folder,device=device)
            self.image_to_text = transformer.image_to_text(self.image_id)
            self.mean_pooling_vec, self.cls_vec = transformer.image_to_embedding(self.image_id)
            self.merged_text, self.merged_text_vec = transformer.text_to_embedding(self.text, self.image_to_text)
            self.nouns, self.verbs, self.adjectives = transformer.tokenize_text(self.merged_text)
            self.user_vec = transformer.user_to_embedding(self.user_format)
            transformData = {   
                'image_to_text': self.image_to_text,
                'mean_pooling_vec' : self.mean_pooling_vec,
                'cls_vec' : self.cls_vec,
                'merged_text' : self.merged_text,
                'merged_text_vec' : self.merged_text_vec,
                'nouns' : self.nouns,
                'verbs' : self.verbs,
                'adjectives' : self.adjectives,
                'user_vec': self.user_vec
            }
            transformData = pd.DataFrame(transformData)
            transformData.to_pickle(save_path)
        else:
            transformer = TransformData(model_folder, image_folder,device=device)
            transformData = pd.read_pickle(save_path)
            self.image_to_text = transformData['image_to_text']
            self.mean_pooling_vec = transformData['mean_pooling_vec']
            self.cls_vec = transformData['cls_vec']
            self.merged_text = transformData['merged_text']
            self.merged_text_vec = transformData['merged_text_vec']
            self.nouns = transformData['nouns']
            self.verbs = transformData['verbs']
            self.adjectives = transformData['adjectives']
            self.user_vec = transformData['user_vec']

    def retrival(self,retrieval_num:int):
        train_data_path, valid_data_path, test_data_path = FilePath.get_split_path(self.datasets_path)
        if os.path.exists(train_data_path) and os.path.exists(valid_data_path) and os.path.exists(test_data_path):
            train_data = pd.read_pickle(train_data_path)
            valid_data = pd.read_pickle(valid_data_path)
            test_data = pd.read_pickle(test_data_path)
        else:
            self.nouns = [list(set(x)) for x in self.nouns]
            self.verbs = [list(set(x)) for x in self.verbs]
            self.adjectives  = [list(set(x)) for x in self.adjectives]
            if hasattr(self, "tags") and self.tags:
                self.tags = [list(set(x)) for x in self.tags]  
            train_data, valid_data, test_data, retrieval_pool = self.split()

            train_data = self.retrieval_data(retrieval_num, train_data, retrieval_pool, "train")
            valid_data = self.retrieval_data(retrieval_num, valid_data, retrieval_pool, "valid")
            test_data = self.retrieval_data(retrieval_num, test_data, retrieval_pool, "test")
            
            train_data, valid_data, test_data = self.stack_retrieved_feature(train_data, valid_data, test_data)
            train_data.to_pickle(train_data_path)
            valid_data.to_pickle(valid_data_path)
            test_data.to_pickle(test_data_path)

    def split(self, train_size: float=0.8, test_size: float=0.5, random_state: int=42) -> tuple:
        df = self.get_dataframe()
        train_data, valid_data = train_test_split(df, test_size=(1 - train_size), random_state=random_state)
        valid_data, test_data = train_test_split(valid_data, test_size=test_size, random_state=random_state)

        train_data.reset_index(drop=True, inplace=True)
        valid_data.reset_index(drop=True, inplace=True)
        test_data.reset_index(drop=True, inplace=True)  

        retrieval_pool = pd.concat([train_data, valid_data], axis=0)
        retrieval_pool.reset_index(drop=True, inplace=True)
        return train_data, valid_data, test_data, retrieval_pool
    
    def retrieval_data(self, retrieval_num: int, data: pd.DataFrame, retrieval_pool: pd.DataFrame, parse: str) -> pd.DataFrame:
        all_features = ['user_id', 'date_posted', 'date_taken', 'date_crawl', 'tags', 'contacts',
                        'photo_count', 'mean_views', 'nouns', 'verbs']

        list_columns = [all_features.index(col) for col in ['tags', 'nouns', 'verbs']]
        retrieval_pool_array = retrieval_pool[all_features].values
        data_array = data[all_features].values
        N = len(retrieval_pool)
        retrieved_item_id_list = []
        retrieved_item_similarity_list = []
        retrieved_label_list = []

        for i in tqdm(range(len(data)), desc=f"Retrieving {parse}..."):
            query_features = data_array[i]
            similarities = self.calculate_similarity(query_features, retrieval_pool_array, N, list_columns)

            similarities[i] = 0
            retrieval_indices = np.argsort(similarities)[::-1][:retrieval_num]
            retrieved_items = retrieval_pool.iloc[retrieval_indices]

            retrieved_item_id_list.append(retrieved_items['image_id'].tolist())
            retrieved_item_similarity_list.append(similarities[retrieval_indices].tolist())
            retrieved_label_list.append(retrieved_items['label'].tolist())

        data['retrieved_item_id'] = retrieved_item_id_list
        data['retrieved_item_similarity'] = retrieved_item_similarity_list
        data['retrieved_label'] = retrieved_label_list
            
        return data

    def stack_retrieved_feature(self,train_data, valid_data, test_data):

        all_data = self.get_dataframe()
        all_data = all_data.drop_duplicates(subset='image_id',keep='first')
        all_data.set_index('image_id', inplace=True)
        train_data = self.get_retrieved_features(train_data, all_data)
        valid_data = self.get_retrieved_features(valid_data, all_data)
        test_data = self.get_retrieved_features(test_data, all_data)

        return train_data, valid_data, test_data

    def get_retrieved_features(self, data: pd.DataFrame, all_data: pd.DataFrame) -> pd.DataFrame:
        retrieved_visual_feature_embedding_cls_list = []
        retrieved_visual_feature_embedding_mean_list = []
        retrieved_textual_feature_embedding_list = []
        retrieved_user_feature_embedding_list = []
        retrieve_label_list = []

        for i in tqdm(range(len(data))):
            id_list = data['retrieved_item_id'][i]
            current_retrieved_visual_feature_embedding_cls_list = []
            current_retrieved_visual_feature_embedding_mean_list = []
            current_retrieved_textual_feature_embedding_list = []
            current_retrieved_user_feature_embedding_list = []
            current_retrieved_label_list = []

            for item_id in id_list:
                retrieved_item = all_data.loc[item_id]
                current_retrieved_visual_feature_embedding_cls_list.append(retrieved_item['cls_vec'])
                current_retrieved_visual_feature_embedding_mean_list.append(retrieved_item['mean_pooling_vec'])
                current_retrieved_textual_feature_embedding_list.append(retrieved_item['merged_text_vec'])
                current_retrieved_user_feature_embedding_list.append(retrieved_item['user_vec'])
                current_retrieved_label_list.append(retrieved_item['label'])

            sorted_list = sorted(
                zip(
                    current_retrieved_label_list,
                    current_retrieved_visual_feature_embedding_cls_list,
                    current_retrieved_visual_feature_embedding_mean_list,
                    current_retrieved_user_feature_embedding_list,
                    current_retrieved_textual_feature_embedding_list
                ),
                key=lambda x: x[0]
            )
            (
                current_retrieved_label_list,
                current_retrieved_visual_feature_embedding_cls_list,
                current_retrieved_visual_feature_embedding_mean_list,
                current_retrieved_user_feature_embedding_list,
                current_retrieved_textual_feature_embedding_list
            ) = zip(*sorted_list)

            retrieved_visual_feature_embedding_cls_list.append(current_retrieved_visual_feature_embedding_cls_list)
            retrieved_visual_feature_embedding_mean_list.append(current_retrieved_visual_feature_embedding_mean_list)
            retrieved_textual_feature_embedding_list.append(current_retrieved_textual_feature_embedding_list)
            retrieved_user_feature_embedding_list.append(current_retrieved_user_feature_embedding_list)
            retrieve_label_list.append(current_retrieved_label_list)
        data['retrieved_visual_feature_embedding_cls'] = retrieved_visual_feature_embedding_cls_list
        data['retrieved_visual_feature_embedding_mean'] = retrieved_visual_feature_embedding_mean_list
        data['retrieved_textual_feature_embedding'] = retrieved_textual_feature_embedding_list
        data['retrieved_user_feature_embedding'] = retrieved_user_feature_embedding_list
        data['retrieved_label_list'] = retrieve_label_list     

        return data    
    
class ICIPDataset(UGCDataset):
    def __init__(self, dataset: str, datasets_path: str, logger: logging, sample_ratio: float=1):
        super().__init__(dataset, datasets_path, logger, sample_ratio)

    def preprocess_origin_data(self):
        users_path,headers_path,img_info_path,popularity_path = FilePath.get_dataset_path(dataset=self.dataset,datasets_path=self.datasets_path)

        users = pd.read_csv(users_path)
        headers = pd.read_csv(headers_path)
        img_info = pd.read_csv(img_info_path)
        popularity = pd.read_csv(popularity_path)      

        merged_data = headers.merge(img_info, on='FlickrId').merge(popularity, on='FlickrId')
        merged_data['label'] = merged_data['Day30'].apply(lambda x: math.log2(x / 30 + 1))
        # ["tag1","tag2"] -> [0,1],["tag2","tag3","tag4"] -> [1,2,3]
        tags = self.encode_tags(merged_data['Tags'])
        merged_data['text'] = merged_data['Title'].astype(str) + ' ' + merged_data['Description'].astype(str)
        pattern = r'<a[^>]*>(.*?)</a>'
        merged_data['clean_text'] = merged_data['text'].apply(lambda x: re.sub(pattern, '', x))
        avg_group_members = merged_data['AvgGroupsMemb'].astype(int)
        avg_group_photos = merged_data['AvgGroupPhotos'].astype(int)        
        user_info = users[['UserId', 'Ispro', 'HasStats', 'Contacts', 'PhotoCount', 'MeanViews']]
        user_info = user_info.drop_duplicates(subset="UserId")
        user_info["User_format"] = user_info.apply(
            lambda row: (
                f"UserID: {row['UserId']} | "
                f"IsPro: {row['Ispro']} | "
                f"HasStats: {row['HasStats']} | "
                f"Contacts: {row['Contacts']} | "
                f"PhotoCount: {row['PhotoCount']} | "
                f"MeanViews: {row['MeanViews']}"
            ),
            axis=1
        )
        merged_data = merged_data.merge(user_info, on='UserId', how='left')

        self.image_id = merged_data['FlickrId']
        self.text = merged_data['clean_text']
        self.label = merged_data['label']
        self.user_id = merged_data['UserId']
        self.date_posted = merged_data['DatePosted']
        self.date_taken = merged_data['DateTaken']
        self.date_crawl = merged_data['DateCrawl']
        self.size = merged_data['Size']
        self.num_sets = merged_data['NumSets']
        self.num_groups = merged_data['NumGroups']
        self.avg_group_members = avg_group_members
        self.avg_group_photos = avg_group_photos
        self.tags = tags

        self.is_pro = merged_data['Ispro']
        self.has_status = merged_data['HasStats']
        self.contacts = merged_data['Contacts']
        self.photo_count = merged_data['PhotoCount']
        self.mean_views = merged_data['MeanViews']
        self.user_format = merged_data['User_format']

    def save_preprocessed_origin_data(self, path: str) -> None:

        df = pd.DataFrame({
            "image_id": self.image_id,
            "text": self.text,
            "label": self.label,
            "user_id": self.user_id,
            "date_posted": self.date_posted,
            "date_taken": self.date_taken,
            "date_crawl": self.date_crawl,
            "size": self.size,
            "num_sets": self.num_sets,
            "num_groups": self.num_groups,
            "avg_group_members": self.avg_group_members,
            "avg_group_photos": self.avg_group_photos,
            "tags": self.tags,
            "is_pro": self.is_pro,
            "has_status": self.has_status,
            "contacts": self.contacts,
            "photo_count": self.photo_count,
            "mean_views": self.mean_views,
            "user_format": self.user_format
        })

        df.to_pickle(path)
        self.logger.info(f"Preprocessed origin data be saved in: {path}")

    def load_preprocessed_origin_data(self, path: str) -> None:
        df = pd.read_pickle(path)

        self.image_id = df["image_id"].tolist()
        self.text = df["text"].tolist()
        self.label = df["label"].tolist()
        self.user_id = df["user_id"].tolist()
        self.date_posted = df["date_posted"].tolist()
        self.date_taken = df["date_taken"].tolist()
        self.date_crawl = df["date_crawl"].tolist()
        self.size = df["size"].tolist()
        self.num_sets = df["num_sets"].tolist()
        self.num_groups = df["num_groups"].tolist()
        self.avg_group_members = df["avg_group_members"].tolist()
        self.avg_group_photos = df["avg_group_photos"].tolist()
        self.tags = df["tags"].tolist()
        self.is_pro = df["is_pro"].tolist()
        self.has_status = df["has_status"].tolist()
        self.contacts = df["contacts"].tolist()
        self.photo_count = df["photo_count"].tolist()
        self.mean_views = df["mean_views"].tolist()
        self.user_format = df["user_format"].tolist()

        self.logger.info(f"Preprocessed origin data loaded from: {path}")

    def get_dataframe(self) -> pd.DataFrame:
        dataset = {
            'image_id': self.image_id,
            'text': self.text,
            'label': self.label,
            'user_id': self.user_id,
            'date_posted': self.date_posted,
            'date_taken': self.date_taken,
            'date_crawl': self.date_crawl,
            'size': self.size,
            'num_sets': self.num_sets,
            'num_groups': self.num_groups,
            'avg_group_members': self.avg_group_members,
            'avg_group_photos': self.avg_group_photos,
            'tags': self.tags,
            'is_pro' : self.is_pro,
            'has_status' : self.has_status,
            'contacts' : self.contacts,
            'photo_count' : self.photo_count,
            'mean_views' : self.mean_views,
            'user_format': self.user_format,
            'user_vec': self.user_vec,
            'image_to_text': self.image_to_text,
            'mean_pooling_vec' : self.mean_pooling_vec,
            'cls_vec' : self.cls_vec,
            'merged_text' : self.merged_text,
            'merged_text_vec' : self.merged_text_vec,
            'nouns' : self.nouns,
            'verbs' : self.verbs,
            'adjectives' : self.adjectives
            }    
        return pd.DataFrame(dataset)           
    
    def encode_tags(self, word_list: list) -> list:
        word_dict = defaultdict(lambda: len(word_dict) + 1)  
        encoded_list = []

        for sublist in word_list:
            if isinstance(sublist, str):
                words_list = eval(sublist) 
            else:
                words_list = sublist if isinstance(sublist, list) else []
            words = [word_dict[word] for word in words_list]
            encoded_list.append(words)

        return encoded_list
    
    def calculate_similarity(self, query_features, dataset_features, N, list_columns):
        result = np.zeros((len(dataset_features), len(query_features)), dtype=int)

        for i, feature in enumerate(query_features):
            if i in list_columns:
                result[:, i] = [bool(set(feature) & set(df_feature)) for df_feature in dataset_features[:, i]]
            else:
                result[:, i] = (dataset_features[:, i] == feature)

        n_values = result.sum(axis=0)

        def f_similarity(n):
            return abs( np.log((N - n + 0.5) / (n + 0.5)))
        similarity = np.dot(result, f_similarity(n_values))
        return similarity
    
class SMPDDataset(UGCDataset):
    def __init__(self, dataset: str, datasets_path: str, logger: logging, sample_ratio: float=0.1):
        super().__init__(dataset, datasets_path, logger, sample_ratio)

    def encode_tags_list(self,word_list):
        word_dict = {}
        encoded_list = []
        for sublist in word_list:
            if sublist == []:
                encoded_list.append([0])
                continue
            if isinstance(sublist, list):
                words_list = sublist
            else:
                words_list = eval(sublist)
            words = []
            for word in words_list:
                if word not in word_dict:
                    word_dict[word] = len(word_dict) + 1
                words.append(word_dict[word])
            encoded_list.append(words)
        return encoded_list

    def encode_tags(self,word_list):
        word_dict = {}
        encoded_list = []
        for sublist in word_list:
            if sublist == []:
                encoded_list.append([0])
                continue
            words = []
            if sublist not in word_dict.keys():
                word_dict[sublist] = len(word_dict) + 1
            words.append(word_dict[sublist])
            encoded_list.append(words)
        return encoded_list

    def preprocess_origin_data(self):

        (
            train_additional_information_path,
            train_category_path,
            train_temporalspatial_information_path,
            train_user_data_path,
            label_data_path,
            train_text_path
        ) = FilePath.get_dataset_path(dataset=self.dataset, datasets_path=self.datasets_path)

        train_additional_information = pd.read_json(train_additional_information_path)
        train_category = pd.read_json(train_category_path)
        train_temporalspatial_information = pd.read_json(train_temporalspatial_information_path)
        train_user_data = pd.read_json(train_user_data_path)
        label_data = pd.read_csv(label_data_path, header=None)
        train_text = pd.read_json(train_text_path)

        sample_ratio = getattr(self, "sample_ratio", 1.0)
        if sample_ratio < 1.0:
            sampled_indices = train_additional_information.sample(frac=sample_ratio, random_state=2025).index
        else:
            sampled_indices = train_additional_information.index

        train_additional_information = train_additional_information.loc[sampled_indices].reset_index(drop=True)
        train_category = pd.read_json(train_category_path).loc[sampled_indices].reset_index(drop=True)
        train_temporalspatial_information = pd.read_json(train_temporalspatial_information_path).loc[sampled_indices].reset_index(drop=True)
        train_user_data = pd.read_json(train_user_data_path).loc[sampled_indices].reset_index(drop=True)
        label_data = pd.read_csv(label_data_path, header=None).loc[sampled_indices].reset_index(drop=True)
        train_text = pd.read_json(train_text_path).loc[sampled_indices].reset_index(drop=True)

        self.tags = [tags.split() for tags in train_text['Alltags']]
        self.text = train_text['Title']
        self.pathalias = self.encode_tags(train_additional_information['Pathalias'])
        self.image_id = [f"{x}_{y}" for x, y in zip(train_additional_information['Uid'], train_additional_information['Pid'])]
        self.user_id = self.encode_tags(train_additional_information['Uid'])
        self.category = self.encode_tags(train_category['Category'])
        self.subcategory = self.encode_tags(train_category['Subcategory'])
        self.concepts = self.encode_tags(train_category['Concept'])
        self.postdate = train_temporalspatial_information['Postdate']

        self.photo_firstdate = train_user_data['photo_firstdate']
        self.photo_firstdatetaken = train_user_data['photo_firstdatetaken']
        self.photo_count = train_user_data['photo_count']
        self.time_zone_id = train_user_data['timezone_timezone_id']
        self.time_zone_offset = train_user_data['timezone_offset']
        self.label = label_data[0].tolist()
        self.user_format = [
            (
                f"Uid: {uid} | "
                f"FirstDate: {fd} | "
                f"FirstDateTaken: {fdt} | "
                f"PhotoCount: {pc} | "
                f"TimeZoneID: {tzid} | "
                f"TimeZoneOffset: {tzoff}"
            )
            for uid, fd, fdt, pc, tzid, tzoff in zip(
                train_additional_information['Uid'],
                train_user_data['photo_firstdate'],
                train_user_data['photo_firstdatetaken'],
                train_user_data['photo_count'],
                train_user_data['timezone_timezone_id'],
                train_user_data['timezone_offset'],
            )
        ]
    
    def save_preprocessed_origin_data(self, path: str) -> None:

        df = pd.DataFrame({
            "image_id": self.image_id,
            "user_id": self.user_id,
            "pathalias": self.pathalias,
            "text": self.text,
            "tags": self.tags,
            "category": self.category,
            "subcategory": self.subcategory,
            "concepts": self.concepts,
            "postdate": self.postdate,
            "photo_firstdate": self.photo_firstdate,
            "photo_firstdatetaken": self.photo_firstdatetaken,
            "photo_count": self.photo_count,
            "time_zone_id": self.time_zone_id,
            "time_zone_offset": self.time_zone_offset,
            "label": self.label,
            "user_format": self.user_format          
        })
        df.to_pickle(path)
        self.logger.info(f"Preprocessed origin data be saved in: {path}")

    def load_preprocessed_origin_data(self, path: str) -> None:
        df = pd.read_pickle(path)

        self.image_id = df["image_id"].tolist()
        self.user_id = df["user_id"].tolist()
        self.pathalias = df["pathalias"].tolist()
        self.text = df["text"].tolist()
        self.tags = df["tags"].tolist()
        self.category = df["category"].tolist()
        self.subcategory = df["subcategory"].tolist()
        self.concepts = df["concepts"].tolist()
        self.postdate = df["postdate"].tolist()
        self.photo_firstdate = df["photo_firstdate"].tolist()
        self.photo_firstdatetaken = df["photo_firstdatetaken"].tolist()
        self.photo_count = df["photo_count"].tolist()
        self.time_zone_id = df["time_zone_id"].tolist()
        self.time_zone_offset = df["time_zone_offset"].tolist()
        self.label = df["label"].tolist()
        self.user_format = df["user_format"].tolist()
        
        self.logger.info(f"Preprocessed origin data loaded from: {path}")

    def get_dataframe(self)->pd.DataFrame:
        dataset = {
            'image_id': self.image_id,
            'text': self.text,
            'label': self.label,
            'user_id': self.user_id,
            'category': self.category,
            'subcategory': self.subcategory,
            'concepts': self.concepts,
            'postdate': self.postdate,
            'photo_firstdate': self.photo_firstdate,
            'photo_firstdatetaken': self.photo_firstdatetaken,
            'photo_count': self.photo_count,
            'time_zone_id': self.time_zone_id,
            'time_zone_offset': self.time_zone_offset,
            'pathalias': self.pathalias,
            'tags': self.tags,
            'user_format': self.user_format,
            'user_vec': self.user_vec,
            'image_to_text': self.image_to_text,
            'mean_pooling_vec' : self.mean_pooling_vec,
            'cls_vec' : self.cls_vec,
            'merged_text' : self.merged_text,
            'merged_text_vec' : self.merged_text_vec,
            'nouns' : self.nouns,
            'verbs' : self.verbs,
            'adjectives' : self.adjectives
        }
        return pd.DataFrame(dataset)
    
    def retrieval_data(self, retrieval_num: int, data: pd.DataFrame, retrieval_pool: pd.DataFrame, parse: str) -> pd.DataFrame:
        all_features = ['user_id', 'pathalias', 'category', 'subcategory', 'concepts', 'postdate', 'photo_firstdate',
                    'photo_firstdatetaken', 'photo_count', 'time_zone_id', 'nouns', 'verbs']

        dataset_array = retrieval_pool[all_features].values
        data_array = data[all_features].values

        N = len(retrieval_pool)

        retrieved_item_id_list = []
        retrieved_item_similarity_list = []
        retrieved_label_list = []

        for i in tqdm(range(len(data)), desc=f"Retrieving {parse}..."):
            query_features = data_array[i]
            similarities = self.calculate_similarity((dataset_array == query_features).astype(int), N)

            similarities[i] = 0
            retrieval_indices = np.argsort(similarities)[::-1][:retrieval_num]
            retrieved_items = retrieval_pool.iloc[retrieval_indices]

            retrieved_item_id_list.append(retrieved_items['image_id'].tolist())
            retrieved_item_similarity_list.append(similarities[retrieval_indices].tolist())
            retrieved_label_list.append(retrieved_items['label'].tolist())

        data['retrieved_item_id'] = retrieved_item_id_list
        data['retrieved_item_similarity'] = retrieved_item_similarity_list
        data['retrieved_label'] = retrieved_label_list
            
        return data
    
    def calculate_similarity(self,all_retrieval_result, N):
        n_values = all_retrieval_result.sum(axis=0)

        def f_similarity(n):
            return np.log((N - n + 0.5) / (n + 0.5))

        similarity = np.dot(all_retrieval_result, f_similarity(n_values))
        return similarity
    
class INSDataset(UGCDataset):
    def __init__(self, dataset: str, datasets_path: str, logger: logging, sample_ratio: float=0.02):
        super().__init__(dataset, datasets_path, logger, sample_ratio)
        
    def preprocess_origin_data(self):
        meta_data_path, json_path = FilePath.get_dataset_path(self.dataset, self.datasets_path)

        df = pd.read_csv(meta_data_path, sep='\t', header=None, names=['id', 'username', 'value', 'user_data_json', 'image_list'], engine='python')

        id_list = df['id'].tolist()
        username_list = df['username'].tolist()
        value_list = df['value'].tolist()
        filename_list = df['user_data_json'].tolist()
        image_list = df['image_list'].tolist()

        meta_data = {
            'id': id_list,
            'username': username_list,
            'value': value_list,
            'user_data_json': filename_list,
            'image_list': image_list
        }

        image_id_list = []
        text_list = []
        comment_num_list = []
        label_list = []
        user_id_list = []
        taken_timestamp_list = []
        user_name_list = []

        for i in tqdm(range(len(meta_data['id']))):
            
            user_data_json_path = meta_data['user_data_json'][i]
            pic_name_list = eval(meta_data['image_list'][i])
            user_name = meta_data['username'][i]
            
            with open(f"{json_path}/{user_data_json_path}", 'r', encoding='UTF-8') as json_file:
                user_data = json.load(json_file)
                label = user_data["edge_media_preview_like"]["count"]
                edge_media_to_caption = user_data["edge_media_to_caption"]["edges"]
                caption = edge_media_to_caption[0]["node"]["text"] if len(edge_media_to_caption) != 0 else ""
                comment_num = user_data["edge_media_to_comment"]["count"]
                user_id = user_data["owner"]["id"]
                taken_at_timestamp = user_data["taken_at_timestamp"]

            if len(pic_name_list) > 0:
                image_id_list.append(pic_name_list[0])
                label_list.append(label)
                text_list.append(caption)
                comment_num_list.append(comment_num)
                user_id_list.append(user_id)
                taken_timestamp_list.append(taken_at_timestamp)
                user_name_list.append(user_name)
            else:
                continue

        full_df = pd.DataFrame({
            "image_id": image_id_list,
            "text": text_list,
            "label": label_list,
            "user_id": user_id_list,
            "taken_timestamp": taken_timestamp_list,
            "user_name": user_name_list,
            "comment_num": comment_num_list,
        })

        sampled_df = full_df.sample(frac=self.sample_ratio, random_state=2025).reset_index(drop=True)

        self.image_id = sampled_df["image_id"].tolist()
        self.text = sampled_df["text"].tolist()
        self.label = sampled_df["label"].tolist()
        self.user_id = sampled_df["user_id"].tolist()
        self.taken_timestamp = sampled_df["taken_timestamp"].tolist()
        self.user_name = sampled_df["user_name"].tolist()
        self.comment_num = sampled_df["comment_num"].tolist()
        self.user_format = [
            f"UserID: {uid} | UserName: {uname}"
            for uid, uname in zip(self.user_id, self.user_name)
        ]

    def save_preprocessed_origin_data(self, path: str) -> None:

        df = pd.DataFrame({
            "image_id": self.image_id,
            'text': self.text,
            'label': self.label,
            'user_id': self.user_id,
            'user_name': self.user_name,
            "taken_timestamp": self.taken_timestamp,
            'comment_num': self.comment_num,
            "user_format": self.user_format          
        })
        df.to_pickle(path)
        self.logger.info(f"Preprocessed origin data be saved in: {path}")

    def load_preprocessed_origin_data(self, path: str) -> None:
        df = pd.read_pickle(path)

        self.image_id = df["image_id"].tolist()
        self.text = df["text"].tolist()
        self.label = df["label"].tolist()
        self.user_id = df["user_id"].tolist()
        self.user_name = df["user_name"].tolist()
        self.taken_timestamp = df["taken_timestamp"].tolist()
        self.comment_num = df["comment_num"].tolist()
        self.user_format = df["user_format"].tolist()
        
        self.logger.info(f"Preprocessed origin data loaded from: {path}")

    def get_dataframe(self)->pd.DataFrame:    
        dataset = {
            'image_id': self.image_id,
            'text': self.text,
            'label': self.label,
            'user_id': self.user_id,
            'taken_timestamp': self.taken_timestamp,
            'user_name': self.user_name,
            'user_format': self.user_format,
            'user_vec': self.user_vec,
            'comment_num': self.comment_num,
            'image_to_text': self.image_to_text,
            'mean_pooling_vec' : self.mean_pooling_vec,
            'cls_vec' : self.cls_vec,
            'merged_text' : self.merged_text,
            'merged_text_vec' : self.merged_text_vec,
            'nouns' : self.nouns,
            'verbs' : self.verbs,
            'adjectives' : self.adjectives
        }
        return pd.DataFrame(dataset)
    
    def retrieval_data(self, retrieval_num: int, data: pd.DataFrame, retrieval_pool: pd.DataFrame, parse: str) -> pd.DataFrame:
        all_features = ['comment_num', 'user_id', 'taken_timestamp']

        dataset_array = retrieval_pool[all_features].values
        data_array = data[all_features].values

        N = len(retrieval_pool)

        retrieved_item_id_list = []
        retrieved_item_similarity_list = []
        retrieved_label_list = []

        for i in tqdm(range(len(data)), desc=f"Retrieving {parse}..."):
            query_features = data_array[i]
            similarities = self.calculate_similarity((dataset_array == query_features).astype(int),N)

            similarities[i] = 0
            retrieval_indices = np.argsort(similarities)[::-1][:retrieval_num]
            retrieved_items = retrieval_pool.iloc[retrieval_indices]

            retrieved_item_id_list.append(retrieved_items['image_id'].tolist())
            retrieved_item_similarity_list.append(similarities[retrieval_indices].tolist())
            retrieved_label_list.append(retrieved_items['label'].tolist())

        data['retrieved_item_id'] = retrieved_item_id_list
        data['retrieved_item_similarity'] = retrieved_item_similarity_list
        data['retrieved_label'] = retrieved_label_list
            
        return data
    def calculate_similarity(self, all_retrieval_result, N):
        n_values = all_retrieval_result.sum(axis=0)

        def f_similarity(n):
            return np.log((N - n + 0.5) / (n + 0.5))

        similarity = np.dot(all_retrieval_result, f_similarity(n_values))
        return similarity

    def retrieval_data(self, retrieval_num: int, data: pd.DataFrame, retrieval_pool: pd.DataFrame, parse: str) -> pd.DataFrame:
        all_features = ["comment_num", 'user_id', 'taken_timestamp']

        retrieval_pool_array = retrieval_pool[all_features].values
        data_array = data[all_features].values
        N = len(retrieval_pool)
        retrieved_item_id_list = []
        retrieved_item_similarity_list = []
        retrieved_label_list = []

        for i in tqdm(range(len(data)), desc=f"Retrieving {parse}..."):
            query_features = data_array[i]
            similarities = self.calculate_similarity((retrieval_pool_array == query_features).astype(int), N)

            similarities[i] = 0
            retrieval_indices = np.argsort(similarities)[::-1][:retrieval_num]
            retrieved_items = retrieval_pool.iloc[retrieval_indices]

            retrieved_item_id_list.append(retrieved_items['image_id'].tolist())
            retrieved_item_similarity_list.append(similarities[retrieval_indices].tolist())
            retrieved_label_list.append(retrieved_items['label'].tolist())

        data['retrieved_item_id'] = retrieved_item_id_list
        data['retrieved_item_similarity'] = retrieved_item_similarity_list
        data['retrieved_label'] = retrieved_label_list
            
        return data
