import torch
import pandas as pd
import numpy as np
def custom_collate_fn(batch):

    # Unzip the batch into respective features and labels
    (
        mean_pooling_vec, 
        merge_text_vec, 
        user_vec,
        retrieved_visual_feature_embedding_cls,
        retrieved_textual_feature_embedding,
        retrieved_user_feature_embedding,
        retrieved_item_similarity, 
        retrieved_label_list, 
        label) = zip(*batch)
 
    # Use torch.stack to convert the batch of data into tensors
    mean_pooling_vec = torch.stack([torch.tensor(item, dtype=torch.float32) for item in mean_pooling_vec])
    merge_text_vec = torch.stack([torch.tensor(item, dtype=torch.float32) for item in merge_text_vec])
    user_vec = torch.stack([torch.tensor(item, dtype=torch.float32) for item in user_vec])

    retrieved_visual_feature_embedding_cls = torch.stack([torch.tensor(item, dtype=torch.float32) for item in retrieved_visual_feature_embedding_cls])
    retrieved_textual_feature_embedding = torch.stack([torch.tensor(item, dtype=torch.float32) for item in retrieved_textual_feature_embedding])
    retrieved_user_feature_embedding = torch.stack([torch.tensor(item, dtype=torch.float32) for item in retrieved_user_feature_embedding])
    
    retrieved_label_list = torch.stack([torch.tensor(item, dtype=torch.float32) for item in retrieved_label_list])
    retrieved_item_similarity = torch.stack([torch.tensor(item, dtype=torch.float32) for item in retrieved_item_similarity])
    label = torch.stack([torch.tensor(item, dtype=torch.float32) for item in label])

    return (
        mean_pooling_vec.squeeze(1), 
        merge_text_vec.squeeze(1), 
        user_vec.squeeze(1),
        retrieved_visual_feature_embedding_cls,
        retrieved_textual_feature_embedding,
        retrieved_user_feature_embedding,
        retrieved_item_similarity,
        retrieved_label_list.squeeze(1), 
        label
    )

class UGCDataset(torch.utils.data.Dataset):

    def __init__(self, retrieval_num, path, collate_fn=custom_collate_fn, normalize = False):
        super().__init__()

        self.path = path
        self.retrieval_num = retrieval_num
        dataframe = pd.read_pickle(path)
        self.len = len(dataframe)
        if normalize:
            raw_label = dataframe['label'].values
            self.label = np.log2(raw_label + 1.0)
        else:
            self.label = dataframe['label']
        self.mean_pooling_vec = dataframe['mean_pooling_vec']
        self.merge_text_vec = dataframe['merged_text_vec']
        self.user_vec = dataframe['user_vec']
        
        self.retrieval_visual_feature_embedding_cls = dataframe['retrieved_visual_feature_embedding_cls']
        self.retrieval_textual_feature_embedding = dataframe['retrieved_textual_feature_embedding']
        self.retrieved_user_feature_embedding = dataframe['retrieved_user_feature_embedding']
        self.retrieved_item_similarity = dataframe['retrieved_item_similarity']
        self.retrieval_label_list = dataframe['retrieved_label_list']
        
    def __getitem__(self, item):
        label = self.label[item]
        mean_pooling_vec = self.mean_pooling_vec[item]
        merge_text_vec = self.merge_text_vec[item]
        user_vec = self.user_vec[item]

        # 截取前 retrieval_num 个检索样本
        retrieved_visual_feature_embedding_cls = self.retrieval_visual_feature_embedding_cls[item][:self.retrieval_num]
        retrieved_textual_feature_embedding = self.retrieval_textual_feature_embedding[item][:self.retrieval_num]
        retrieved_user_feature_embedding = self.retrieved_user_feature_embedding[item][:self.retrieval_num]
        retrieved_label_list = self.retrieval_label_list[item][:self.retrieval_num]
        retrieved_item_similarity = self.retrieved_item_similarity[item][:self.retrieval_num]

        return (
            mean_pooling_vec, 
            merge_text_vec, 
            user_vec,
            retrieved_visual_feature_embedding_cls,
            retrieved_textual_feature_embedding, 
            retrieved_user_feature_embedding,
            retrieved_item_similarity, 
            retrieved_label_list, 
            label
        )

    def __len__(self):
        return self.len
