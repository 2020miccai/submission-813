import os
import scipy.sparse as sp
import torch
from torch.utils.data import Dataset

from domainadapt.input.graph_sample import GraphSample


# Ignore warnings
# import warnings
# warnings.filterwarnings("ignore")


def make_dataset(root, mode):
    # assert mode in ['train', 'val', 'test']
    items = []
    if mode == 'train_lab':
        oth = 'train_unl'
        files_lab_list = os.listdir(os.path.join(root, 'train_lab'))
        files_unl_list = os.listdir(os.path.join(root, oth))
        ratio = int(files_unl_list.__len__() / files_lab_list.__len__())

        files_lab_list.sort()
        for it in range(len(files_lab_list)):
            item = os.path.join(os.path.join(root, 'train_lab'), files_lab_list[it])
            items.append(item)

        items = items * int(ratio)

    elif mode == 'val':
        files_list = os.listdir(os.path.join(root, 'valid'))
        files_list.sort()

        for it in range(len(files_list)):
            item = os.path.join(os.path.join(root, 'valid'), files_list[it])
            items.append(item)

    elif mode == 'test':
        files_list = os.listdir(os.path.join(root, 'test'))
        files_list.sort()

        for it in range(len(files_list)):
            item = os.path.join(os.path.join(root, 'test'), files_list[it])
            items.append(item)
    else:
        files_list = os.listdir(os.path.join(root, mode))
        files_list.sort()

        for it in range(len(files_list)):
            item = os.path.join(os.path.join(root, mode), files_list[it])
            items.append(item)

    return items


class GeometricDataset(Dataset):

    def __init__(self, mode, root_dir):
        """
        Args:
            mode: 'train', 'valid', or 'test'
            root_dir: path to the dataset
        """
        self.root_dir = root_dir
        self.files = make_dataset(root_dir, mode)
        self.mode = mode

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        files_path = self.files[index]
        # print(files_path)
        data = torch.load(files_path)

        # files_path = self.files[index]
        # # print(files_path)
        # temp = torch.load(files_path, encoding='latin1')
        #
        # # Enable this if you want to add Cortical thickness (X: 3 spectral coordinates + C: Sulcul depth + T: Cotical
        # # thickness)
        # # x = torch.cat((torch.FloatTensor(temp['X'][:, 0:3]), torch.FloatTensor(temp['C']),
        # # torch.FloatTensor(temp['T'])), 1)
        # #
        #
        # x = torch.cat((torch.FloatTensor(temp['X'][:, 0:3]),
        #                torch.FloatTensor(temp['C'])), 1)
        # e1, e2, e3 = sp.find(temp['A'])
        # edge_idx = torch.cat((torch.LongTensor(e2).unsqueeze(0), torch.LongTensor(e1).unsqueeze(0)),
        #                      0)  # index x,y of the sparse adj matrix
        # edge_wht = torch.LongTensor(e3).unsqueeze(
        #     0)  # the weights of the edges. Often used to construct the sparse matrix (adj)
        # gt = torch.LongTensor(temp['GT'])  # This is manual label
        # xyz = torch.FloatTensor(temp['EUC'])  # This is the xyz of the mesh node location in euclidean coordinates
        # face = torch.FloatTensor(temp['F'])  # This is the face of the mesh triangualtion
        # age = torch.FloatTensor(temp['AG'][0])  # This is the age of the subject
        # sx = torch.FloatTensor(temp['SX'])  # This is the gender of the subject
        # lab = torch.FloatTensor(temp['Y'])  # This is FreeSurfer label
        #
        # data = GraphSample(x=x, edge_idx=edge_idx, edge_wht=edge_wht, gt=gt, xyz=xyz, face=face, age=age, sx=sx,
        #                    lab=lab)
        return data
