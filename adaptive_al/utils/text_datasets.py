"""
Custom PyTorch Dataset for text classification tasks.

This module provides a Dataset class that handles tokenization of text data
for use with transformer models in text classification tasks.
"""

from typing import Any

import torch
from torch.utils.data import Dataset

class TextClassificationDataset(Dataset):
    """
    A custom PyTorch Dataset for text classification with transformer tokenization.

    This dataset takes raw text and labels, applies tokenization using a
    Hugging Face tokenizer, and returns properly formatted inputs for
    transformer models.

    Attributes:
        texts (list[str]): List of text samples.
        labels (list[int]): List of integer labels corresponding to each text.
        tokenizer (Any): Hugging Face tokenizer instance.
        tokenizer_kwargs (dict): Arguments passed to the tokenizer (e.g., max_length, padding).
    """

    def __init__(self, texts: list[str], labels: list[int], tokenizer: Any, tokenizer_kwargs: dict):
        """
        Initialize the text classification dataset.

        Args:
            texts (list[str]): List of text samples to classify.
            labels (list[int]): List of integer labels for each text.
            tokenizer (Any): Hugging Face tokenizer instance for text encoding.
            tokenizer_kwargs (dict): Keyword arguments for tokenization
                                    (e.g., max_length, padding, truncation).
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.tokenizer_kwargs = tokenizer_kwargs

    def __len__(self):
        """
        Return the total number of samples in the dataset.

        Returns:
            int: Number of text samples.
        """
        return len(self.texts)

    def __getitem__(self, idx):
        """
        Retrieve a single tokenized sample and its label.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple: A tuple containing:
                - inputs (dict): Dictionary with 'input_ids' and 'attention_mask' tensors.
                - label (torch.Tensor): Label as a long tensor.
        """
        text = self.texts[idx]
        label = self.labels[idx]


        encoding = self.tokenizer(text, **self.tokenizer_kwargs)



        inputs = {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze()
        }

        return inputs, torch.tensor(label, dtype=torch.long)