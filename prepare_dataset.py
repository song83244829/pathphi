import json
import torch
import copy
from PIL import Image
from PIL import ImageFile
import os
from pathlib import Path

from phi3v_dataset import Phi3VDataset, Phi3VEvalDataset

def apply_chat_template(
    example,
    tokenizer,
):
    messages = example["messages"]
    example["text"] = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False)
    return example


def create_dataset(data_dir, processor):
    """
    Using dataset from preprocessed folder
    """
    data_path = Path(data_dir)
    train_dataset = Phi3VDataset(
        jsonl_file=str(data_path / 'quilt_1m_train.jsonl'),
        image_dir=str(data_path / 'quilt_1m'),
        processor=processor,
    )
    
    eval_dataset = Phi3VEvalDataset(
        jsonl_file=str(data_path / 'quilt_1m_val.jsonl'),
        image_dir=str(data_path / 'quilt_1m'),
        processor=processor,
    )

    return train_dataset, eval_dataset