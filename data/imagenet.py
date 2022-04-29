import os
import json
import torch
from PIL import Image
import torch.utils.data as data
from torchvision import transforms as torch_transforms


mean = [0.485, 0.456, 0.406]

std = [0.229, 0.224, 0.225]


class ImagenetDataset(data.Dataset):
    """
        Dataset class for the ImageNet dataset prepared for supervised learning.
        Both validation and training will have 1000 folders starting with 'n' (1 folder per class).
    """
    def __init__(self, args, is_training = True, is_evaluation = False):
        """
            :param args: arguments
            :param is_training: Training or validation mode
            :param is_evaluation: Evaluation mode
        """
        super(ImagenetDataset, self).__init__()
        root = args.data
        train_list = 'train.txt' # read imagenet from txt file
        self.is_training = is_training
        self.is_evaluation = is_evaluation
        self.size = (args.size, args.size)

        if self.is_training:
            data_file = os.path.join(root, train_list)
        else:
            data_file = os.path.join(root, 'val.txt')
        
        self.images = []

        with open(data_file, 'r') as lines:
            for line in lines:
                line = line.strip("\n")
                rgb_img_loc = os.path.join(root, line)
                assert os.path.isfile(rgb_img_loc), 'RGB file does not exist at: {}'.format(rgb_img_loc)
                self.images.append(rgb_img_loc)

        anno_file = os.path.join(root, 'class_to_idx.json')

        with open(anno_file) as json_file:
            self.class_to_id = json.load(json_file)

    def training_transform(self, size=(224, 224)):

        normalize = torch_transforms.Normalize(mean, std)
        aug_list = [
            torch_transforms.RandomResizedCrop(size),
            torch_transforms.RandomHorizontalFlip(),
            torch_transforms.ToTensor(),
            normalize
        ]  
        training_transforms = torch_transforms.Compose(aug_list)

        return training_transforms

    def evaluation_transform(self, size=(224, 224)):
        normalize = torch_transforms.Normalize(mean, std)
        aug_list = [
            torch_transforms.Resize(size),
            torch_transforms.RandomResizedCrop(size),
            torch_transforms.CenterCrop(224),
            torch_transforms.ToTensor(),
            normalize
        ]  
        validation_transforms = torch_transforms.Compose(aug_list)
        return validation_transforms

    def __getitem__(self, index):
        if self.is_training:
            transform = self.training_transform(size=self.size)
        else:
            transform = self.evaluation_transform(size=self.size)

        img_path = self.images[index]
        input_img = Image.open(img_path).convert('RGB')
        input_img = transform(input_img)
        data = {"image": input_img}        

        # get class name and transform to image id
        class_name = img_path.split('/')[4]
        target = self.class_to_id[class_name]

        # target is a 0-dimensional tensor
        target_tensor = torch.tensor(1, dtype=torch.long).fill_(target)

        data["label"] = target_tensor
        return data

    def __len__(self):
        return len(self.images)
