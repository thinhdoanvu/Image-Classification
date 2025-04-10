import os
import numpy as np
import random


class DatasetLoader:
    def __init__(self, train_dir, valid_dir):
        self.train_dir = train_dir
        self.valid_dir = valid_dir

        self.training_files = self._load_files(self.train_dir)
        self.validation_files = self._load_files(self.valid_dir)

        self.training_labels, self.classes = self._make_instances(self.training_files)
        self.validation_labels, _ = self._make_instances(self.validation_files, self.classes)

        self.num_classes = len(self.classes)

    def _load_files(self, directory):
        all_files = []
        for root, _, files in os.walk(directory):
            for file in files:
                if file.lower().endswith((".png", ".jpg", ".jpeg")):
                    all_files.append(os.path.join(root, file))
        return sorted(all_files)

    def _make_instances(self, files, classes=None):
        if classes is None:
            classes = []
        labels = []
        for file in files:
            clazz = file.split(os.sep)[-2]
            if clazz in classes:
                label = classes.index(clazz)
            else:
                label = len(classes)
                classes.append(clazz)
            labels.append(label)
        return np.array(labels), classes

    def get_training_data(self):
        return self.training_files, self.training_labels

    def get_validation_data(self):
        return self.validation_files, self.validation_labels

    def get_classes(self):
        return self.classes

    def get_num_classes(self):
        return self.num_classes

    def _create_pairs(self, x, digit_indices):
        pairs = []
        labels = []
        n = min([len(digit_indices[d]) for d in range(self.num_classes)]) - 1

        for d in range(self.num_classes):
            for i in range(n):
                z1, z2 = digit_indices[d][i], digit_indices[d][i + 1]
                pairs.append([x[z1], x[z2]])

                inc = random.randrange(1, self.num_classes)
                dn = (d + inc) % self.num_classes
                z1, z2 = digit_indices[d][i], digit_indices[dn][i]
                pairs.append([x[z1], x[z2]])

                labels += [1, 0]
        return np.array(pairs), np.array(labels, dtype='float32')

    def create_pairs_on_set(self, x, y):
        digit_indices = [np.where(y == i)[0] for i in range(self.num_classes)]
        return self._create_pairs(x, digit_indices)

    def create_pairs_on_training(self):
        return self.create_pairs_on_set(self.training_files, self.training_labels)

    def create_pairs_on_validation(self):
        return self.create_pairs_on_set(self.validation_files, self.validation_labels)
