import os
import os.path as osp

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class DatasetLoader(Dataset):

    def __init__(self, setname, data_path):
        DATASET_DIR = data_path
        print('DATASET_DIR', DATASET_DIR)
        label_dict = {'2S1': 0, 'BMP2': 1, 'BRDM_2': 2, 'BTR70(SN_C71)': 3, 'BTR_60': 4, 'D7': 5,
                      'T62': 6, 'T72(SN_132)': 7, 'ZIL131': 8, 'ZSU_23_4': 9}
        # Set the path according to train, val and test
        if setname == 'train':
            THE_PATH = osp.join(DATASET_DIR, 'train')
            label_list = os.listdir(THE_PATH)
        elif setname == 'test':
            THE_PATH = osp.join(DATASET_DIR, 'test')
            label_list = os.listdir(THE_PATH)
        elif setname == 'val':
            THE_PATH = osp.join(DATASET_DIR, 'val')
            label_list = os.listdir(THE_PATH)
        elif setname in ['all', 'ood']:
            THE_PATH = DATASET_DIR
            label_list = os.listdir(THE_PATH)
        else:
            raise ValueError('Unkown setname.')

        data = []
        label = []
        # Get the images' paths and labels
        if setname in ['train', 'test']:
            for labelname in label_list:
                this_folder = osp.join(THE_PATH, labelname)
                this_folder_images = os.listdir(this_folder)
                for image_path in this_folder_images:
                    data.append(osp.join(this_folder, image_path))
                    label.append(int(label_dict[labelname]))
        else:
            for labelname in label_list:
                this_folder = osp.join(THE_PATH, labelname)
                this_folder_images = os.listdir(this_folder)
                for image_path in this_folder_images:
                    data.append(osp.join(this_folder, image_path))
                    label.append(11)

        self.data = data
        self.label = label
        self.num_class = len(set(label))

        # Transformation
        if setname == 'train':
            image_size = 90
            self.transform = transforms.Compose([
                transforms.Grayscale(3),
                transforms.Resize(image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

        else:
            image_size = 90
            self.transform = transforms.Compose([
                transforms.Grayscale(3),
                transforms.Resize(image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        path, label = self.data[i], self.label[i]
        image = self.transform(Image.open(path).convert('RGB'))
        return image, label
