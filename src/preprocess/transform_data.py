import os
import pandas as pd
from transformers import BlipProcessor, BlipForConditionalGeneration
from transformers import BartTokenizer, BartModel
from transformers import ViTImageProcessor, ViTModel
from tqdm import tqdm
from PIL import Image
import torch
from angle_emb import AnglE
import spacy

class TransformData:
    def __init__(self, model_folder: str, image_folder: str,device:str):
        self.model_folder = model_folder
        self.image_folder = image_folder
        self.device = device

    def image_to_text(self, pic_name_list: list) -> list:
        # load model
        model_path = os.path.join(self.model_folder, "BLIP")
        processor = BlipProcessor.from_pretrained(model_path)
        model = BlipForConditionalGeneration.from_pretrained(model_path).to(self.device)
        image_to_text_list = []
        
        batch_size = 128
        for i in tqdm(range(0, len(pic_name_list), batch_size), desc="Processing images to text."):
            batch_ids = pic_name_list[i:i + batch_size]
            batch_images = []

            # Load images for the current batch
            for current_id in batch_ids:
                current_id = str(current_id).replace(".jpg", "")
                path = os.path.join(self.image_folder, f"{current_id}.jpg")
                if os.path.exists(path):
                    raw_image = Image.open(path).convert('RGB')
                    batch_images.append(raw_image)
                else:
                    batch_images.append(None)  # No image found

            # Remove None entries and handle missing images
            valid_images = [img for img in batch_images if img is not None]
            if not valid_images:
                continue  # Skip if no valid images in the batch

            # Process the batch
            inputs = processor(images=valid_images, return_tensors="pt", padding=True).to(self.device)

            # Generate descriptions for the batch
            with torch.no_grad():
                out = model.generate(**inputs)

            # Decode and collect the results
            out_idx = 0
            for idx, current_id in enumerate(batch_ids):
                if batch_images[idx] is None:
                    text = ""  # If image doesn't exist, add empty text
                else:
                    text = processor.decode(out[out_idx], skip_special_tokens=True)
                    out_idx += 1
                image_to_text_list.append(text)

        return image_to_text_list

    def image_to_embedding(self, pic_name_list: list) -> tuple:
        # load model
        model_path = os.path.join(self.model_folder, "VIT")
        processor = ViTImageProcessor.from_pretrained(model_path)
        model = ViTModel.from_pretrained(model_path).to(self.device)
        
        mean_pooling_vec_list = []
        cls_vec_list = []        

        # transform
        batch_size = 64
        for i in tqdm(range(0, len(pic_name_list), batch_size), desc="Processing images to embeddings."):
            batch = pic_name_list[i:i + batch_size]
            batch_paths = [os.path.join(self.image_folder, f"{str(current_id).replace('.jpg', '')}.jpg") for current_id in batch]
            images = []

            for path in batch_paths:
                if os.path.exists(path):
                    images.append(Image.open(path).convert('RGB'))
                else:
                    images.append(None)

            # Filter out None images before processing
            valid_images = [img for img in images if img is not None]
            if not valid_images:
                continue  # Skip if no valid images in the batch

            inputs = processor(images=valid_images, return_tensors="pt", padding=True).to(self.device)
            with torch.no_grad():
                outputs = model(**inputs)

            cls_output = outputs.last_hidden_state[:, 0, :]
            mean_pooling_output = torch.mean(outputs.last_hidden_state, dim=1)

            # Process embeddings
            valid_idx = 0  # To track valid image indices
            for idx, image in enumerate(images):
                if image is None:
                    mean_pooling_vec_list.append(torch.zeros(768).to(self.device).tolist())
                    cls_vec_list.append(torch.zeros(768).to(self.device).tolist())
                else:
                    mean_pooling_vec_list.append(mean_pooling_output[valid_idx].tolist())
                    cls_vec_list.append(cls_output[valid_idx].tolist())
                    valid_idx += 1

        return mean_pooling_vec_list, cls_vec_list
    
    def text_to_embedding(self, text_list: list, image_to_text_list) -> tuple:
        model_path = os.path.join(self.model_folder, "Bert")
        angel = AnglE.from_pretrained(model_path, pooling_strategy='cls_avg').cuda()
        merged_text_list = []
        merged_text_vec_list = []

        batch_size = 64
        for i in tqdm(range(0, len(text_list), batch_size), desc="Processing text to embeddings."):
            # Get the current batch of text
            batch_text = text_list[i:i + batch_size]
            batch_image_text = image_to_text_list[i:i + batch_size]

            # Merge text and image descriptions for each item in the batch
            batch_merged_text = [t + img_text for t, img_text in zip(batch_text, batch_image_text)]
            merged_text_list.extend(batch_merged_text)

            # Generate embeddings for the current batch
            text_to_embedding_result = angel.encode(batch_merged_text, to_numpy=True)
            merged_text_vec_list.extend(text_to_embedding_result.tolist())

        return merged_text_list, merged_text_vec_list
    
    def user_to_embedding(self, user_info: list, batch_size: int = 64) -> list:
        # Load your encoder
        model_path = os.path.join(self.model_folder, "Bert")
        angle = AnglE.from_pretrained(model_path, pooling_strategy='cls_avg').cuda()

        embedding_list = []

        for i in tqdm(range(0, len(user_info), batch_size), desc="Encoding user embeddings"):
            batch_text = user_info[i:i + batch_size]
            embeddings = angle.encode(batch_text, to_numpy=True)
            embedding_list.extend(embeddings.tolist())

        return embedding_list

    def tokenize_text(self, combine_text_list: list) -> tuple:
        nlp = spacy.load("en_core_web_sm")

        n_nouns = []
        n_verbs = []
        n_adjectives = []

        for i in tqdm(range(len(combine_text_list)), desc="Tokenizing text."):

            merged_text = combine_text_list[i]

            doc = nlp(merged_text)

            nouns = [token.text for token in doc if token.pos_ == "NOUN"]
            verbs = [token.text for token in doc if token.pos_ == "VERB"]
            adjectives = [token.text for token in doc if token.pos_ == "ADJ"]

            n_nouns.append(nouns)
            n_verbs.append(verbs)
            n_adjectives.append(adjectives)
        return n_nouns,n_verbs,n_adjectives