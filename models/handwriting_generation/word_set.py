import os
import xml.etree.ElementTree as ET

import torch
import math
import random
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as T
from PIL import Image, UnidentifiedImageError, ImageOps

import constants


class WordSet(Dataset):

    def __init__(self, dataset, object_image_size, theme='light'):
        data = []
        labels = []

        if dataset == 'IAM':
            images_path = 'dataset/words'
            labels_path = 'dataset/xml'
            for directory in os.listdir(images_path):
                folder_path = os.path.join(images_path, directory)
                for subdirectory in os.listdir(folder_path):
                    labels_xml_path = os.path.join(labels_path, f'{subdirectory}.xml')
                    tree = ET.parse(labels_xml_path)
                    root = tree.getroot()
                    handwritten_part = root.find('handwritten-part')
                    lines = handwritten_part.findall('line')
                    for line in lines:
                        words = line.findall('word')
                        for word in words:
                            word_id = word.get('id')
                            word_image_path = os.path.join(images_path, directory, subdirectory, f'{word_id}.png')
                            word_label = word.get('text')
                            data.append(word_image_path)
                            labels.append(word_label)

        if dataset == 'IIIT-HWS':
            ground_truths_file = open('dataset/iiit-hws/IIIT-HWS-90K.txt', 'r')
            ground_truth_lines = ground_truths_file.readlines()
            for i, line in enumerate(ground_truth_lines):
                line_split_parts = line.split()
                image_path = line_split_parts[0]
                image_label = line_split_parts[1]
                image_path = os.path.join('dataset/iiit-hws/Images_90K_Normalized', image_path)
                data.append(image_path)
                labels.append(image_label)
            ground_truths_file.close()

        self.data = np.array(data)
        self.labels = np.array(labels)
        self.object_image_size = object_image_size
        self.theme = theme
        self.vocab = self.get_vocab()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        try:
            image = Image.open(self.data[index])
        except UnidentifiedImageError:
            random_index = random.randint(0, len(self.data))
            image = Image.open(self.data[random_index])
        image = self.increase_contrast(image)
        image = self.crop_image(image)
        image_width, image_height = image.size
        if max(image_width, image_height) >= self.object_image_size:
            aspect_ratio = image_width / image_height
            if image_width >= image_height:
                new_image_width = self.object_image_size
                new_image_height = math.floor(self.object_image_size / aspect_ratio)
            else:
                new_image_height = self.object_image_size
                new_image_width = math.floor(self.object_image_size * aspect_ratio)
            image = image.resize((new_image_width, new_image_height))

        new_image_size = (self.object_image_size, self.object_image_size)
        new_image = Image.new("L", new_image_size, "white" if self.theme == "light" else "black")
        box = tuple((n - o) // 2 for n, o in zip(new_image_size, image.size))
        new_image.paste(image, box)
        transform = T.PILToTensor()
        image = torch.div(transform(new_image), 255)
        return (image, self.vocab['word_to_idx'][self.labels[index]])

    def get_image(self, word_label, image_size='in-line', line_height=125):
        indices = np.where(self.labels == word_label)[0]
        index = np.random.choice(indices)
        try:
            image = Image.open(self.data[index])
        except UnidentifiedImageError:
            index = np.random.choice(indices)
            image = Image.open(self.data[index])
        image = self.increase_contrast(image)
        image = self.crop_image(image)
        if image_size == 'in-line':
            image_width, image_height = image.size
            if image_height >= line_height:
                aspect_ratio = image_width / image_height
                new_image_height = line_height
                new_image_width = math.floor(line_height * aspect_ratio)
                image = image.resize((new_image_width, new_image_height))
            image_width, image_height = image.size
            new_image_size = (image_width, line_height)
            new_image = Image.new("L", new_image_size, "white")
            box = tuple((n - o) // 2 for n, o in zip(new_image_size, image.size))
            new_image.paste(image, box)
            image = new_image
            if constants.THEME == 'dark':
                image = ImageOps.invert(image)
        return image

    def get_vocab(self):
        idx_to_word = np.sort(np.unique(self.labels))
        word_to_idx = {}
        for idx, word in enumerate(idx_to_word):
            word_to_idx[word] = idx
        vocab = {
            'idx_to_word': idx_to_word,
            'word_to_idx': word_to_idx,
            'vocab_size': len(idx_to_word)
        }
        return vocab

    def increase_contrast(self, image):
        data = np.expand_dims(np.array(image), axis=2)
        rgb = data[:, :, :1]
        color = [160]
        black = [0]
        white = [255]
        mask = np.all(rgb > color, axis=-1)
        data[mask] = white
        data[np.logical_not(mask)] = black
        object_image = Image.fromarray(data[:, :, 0], 'L')
        return object_image

    def crop_image(self, image):
        data = np.array(image)
        color = 160
        rows_to_keep = np.any(data <= color, axis=1)
        cols_to_keep = np.any(data <= color, axis=0)
        data = data[rows_to_keep][:, cols_to_keep]
        object_image = Image.fromarray(data, 'L')
        return object_image