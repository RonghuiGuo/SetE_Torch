import json
import os
import random
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# os.path.realpath(__file__)返回当前文件的绝对路径
# os.path.join(os.path.realpath(__file__), '..')
root_dir = os.path.abspath(os.path.join(os.path.realpath(__file__), '..'))
data_dir = os.path.join(root_dir, 'YAGO39K')
mode_list = ['train', 'vaild', 'test']
raw_list = [f'{data_dir}/raw/{mode}' for mode in mode_list]
cache_list = [f'{data_dir}/cache/{mode}.json' for mode in mode_list]
os.makedirs(f'{data_dir}/cache', exist_ok=True)


class YAGO39K_Dataset(Dataset):
    dataset = None
    # 指定一个类的方法为类方法，类方法的第一个参数必须是类对象，通常以cls作为第一个参数

    @classmethod
    def Merge_YAGO39K(cls, raw_dir, cache_path):
        # 读取概念、实体、关系id
        def read_str2id(path, sep='\t'):
            str2id = {}
            with open(path, 'r') as file:
                for line in tqdm(file.readlines()):
                    row = line.strip().split(sep)
                    if len(row) == 2:
                        str2id[row[0]] = int(row[1])
            return str2id

        # 读取二元/三元关系组
        def read_tuple(path, sep=' '):
            tuples = []
            with open(path, 'r') as file:
                for line in tqdm(file.readlines()):
                    row = line.strip().split(sep)
                    if len(row) >= 2:
                        row = list(map(int, row))
                        tuples.append(row)
            return tuples

        instance2id = read_str2id(f'{raw_dir}/instance2id.txt')
        concept2id = read_str2id(f'{raw_dir}/concept2id.txt')
        relation2id = read_str2id(f'{raw_dir}/relation2id.txt')

        instanceOf = read_tuple(f'{raw_dir}/instanceOf2id.txt')
        subClassOf = read_tuple(f'{raw_dir}/subClassOf2id.txt')
        triple = read_tuple(f'{raw_dir}/triple2id.txt')

        cls.dataset = {'instance2id': instance2id, 'concept2id': concept2id, 'relation2id': relation2id,
                       'instanceOf': instanceOf, 'subClassOf': subClassOf, 'triple': triple}
        with open(cache_path, 'w') as f:
            json.dump(cls.dataset, f)

    def __init__(self, raw_dir, cache_path, flag):
        # 加载dataset
        if YAGO39K_Dataset.dataset == None:
            if os.path.exists(cache_path):
                with open(cache_path, 'r') as f:
                    YAGO39K_Dataset.dataset = json.load(f)
            else:
                YAGO39K_Dataset.Merge_YAGO39K(raw_dir, cache_path)
        self.instanceID = list(self.dataset['instance2id'].values())
        self.conceptID = list(self.dataset['concept2id'].values())

        # 设置flag, 标记训练数据的类型（instanceOf/triple）
        self.flag = flag

    def __len__(self):
        return len(self.dataset[self.flag])

    def __getitem__(self, idx):
        sample_pos = self.dataset[self.flag][idx]
        sample_neg = self.GetNegativeSample(sample_pos)
        return [sample_pos, sample_neg]

    def GetNegativeSample(self, sample_pos):
        if len(sample_pos) == 2:  # instanceof, 换掉尾实体
            head, tail = sample_pos[0], sample_pos[1]
            # while [head, tail] in self.dataset['instanceOf']:
            tail = random.choice(self.conceptID)
            return [head, tail]

        elif len(sample_pos) == 3:  # triple, 换掉头/尾实体
            head, tail, relation = sample_pos[0], sample_pos[1], sample_pos[2]

            if random.random() < 0.5:  # 换头实体
                # while [head, tail, relation] in self.dataset['triple']:
                head = random.choice(self.instanceID)
                return [head, tail, relation]

            else:  # 换尾实体
                # while [head, tail, relation] in self.dataset['triple']:
                tail = random.choice(self.instanceID)
                return [head, tail, relation]


class YAGO39K_DataLoader():
    def __init__(self, raw_dir, cache_path, batch_size, shuffle):
        self.dataset, self.dataloader = {}, {}
        for item in ['instanceOf', 'triple']:
            self.dataset[item] = YAGO39K_Dataset(raw_dir, cache_path, item)
            self.dataloader[item] = DataLoader(
                self.dataset[item], batch_size=batch_size, shuffle=shuffle, collate_fn=self.collate_fn)

    def get_dataloader(self):
        instanceOf = [batch for batch in self.dataloader['instanceOf']]
        triple = [batch for batch in self.dataloader['triple']]
        dataloader = instanceOf + triple
        random.shuffle(dataloader)
        return dataloader

    def collate_fn(self, batch):

        batch_pos, batch_neg = [], []
        for item in batch:
            batch_pos.append(item[0])
            batch_neg.append(item[1])

        batch_pos = torch.LongTensor(batch_pos)
        batch_neg = torch.LongTensor(batch_neg)

        if batch_pos.shape[-1] == 2:  # instanceOf
            return {'flag': 'instanceOf',
                    'data_pos': [batch_pos[:, 0], batch_pos[:, 1]],
                    'data_neg': [batch_neg[:, 0], batch_neg[:, 1]]}

        elif batch_pos.shape[-1] == 3:  # triple
            return {'flag': 'triple',
                    'data_pos': [batch_pos[:, 0], batch_pos[:, 1], batch_pos[:, 2]],
                    'data_neg': [batch_neg[:, 0], batch_neg[:, 1], batch_neg[:, 2]]}


def get_dataloaders(batch_size, skip_list=[]):
    dataloader = []
    shuffle_list = [True, False, False]
    for i in range(len(raw_list)):
        if i in skip_list:
            dataloader.append(None)
        else:
            dataclass = YAGO39K_DataLoader(
                raw_list[i], cache_list[i], batch_size=batch_size, shuffle=shuffle_list[i])
            dataloader.append(dataclass.get_dataloader())
    return dataloader
