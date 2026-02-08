from torch.utils.data import Dataset
import logging
import torch
from utils.iotools import read_image
from utils.simple_tokenizer import SimpleTokenizer
from prettytable import PrettyTable
import random
import os


class BaseDataset(object):
    """
    Base class of text to image reid dataset
    """
    logger = logging.getLogger("ORBench.dataset")

    def show_dataset_info(self):
        num_train_pids, num_train_imgs, num_train_captions = len(
            self.train_id_container), len(self.train_annos), len(self.train)

        queries_num = 0
        queries = self.test['queries']
        for key in queries.keys():
            queries_num+=len(queries[key])

        num_test_pids, num_test_imgs, num_test_captions = len(self.test_id_container), len(self.test['gallery_paths']), queries_num

        # TODO use prettytable print comand line table

        self.logger.info(f"{self.__class__.__name__} Dataset statistics:")
        table = PrettyTable(['subset', 'ids', 'images', 'captions'])
        table.add_row(
            ['train', num_train_pids, num_train_imgs, num_train_captions])
        table.add_row(
            ['test', num_test_pids, num_test_imgs, num_test_captions])
        self.logger.info('\n' + str(table))


def tokenize(caption: str, tokenizer, text_length=77, truncate=True) -> torch.LongTensor:
    sot_token = tokenizer.encoder["<|startoftext|>"]
    eot_token = tokenizer.encoder["<|endoftext|>"]
    tokens = [sot_token] + tokenizer.encode(caption) + [eot_token]

    result = torch.zeros(text_length, dtype=torch.long)
    if len(tokens) > text_length:
        if truncate:
            tokens = tokens[:text_length]
            tokens[-1] = eot_token
        else:
            raise RuntimeError(
                f"Input {caption} is too long for context length {text_length}"
            )
    result[:len(tokens)] = torch.tensor(tokens)
    return result


class ImageTextDataset(Dataset):
    def __init__(self,
                 dataset,
                 transform=None,
                 text_length: int = 77,
                 truncate: bool = True):
        self.dataset = dataset
        self.transform = transform
        self.text_length = text_length
        self.truncate = truncate
        self.tokenizer = SimpleTokenizer()
        self.nir_paths, self.cp_paths, self.sk_paths = self.get_paths()

    def __len__(self):
        return len(self.dataset)

    def get_paths(self):
        dataset_root = self.dataset[0][2].split("ORBench")[0]
        nir_paths = {}
        for nir_identity in os.listdir(os.path.join(dataset_root,'ORBench','nir')):
            if nir_identity not in nir_paths.keys():
                nir_paths[nir_identity] = []
            identity_paths = os.listdir(os.path.join(dataset_root,'ORBench','nir',nir_identity))
            for path in identity_paths:
                nir_paths[nir_identity].append(os.path.join(dataset_root,'ORBench','nir',nir_identity,path))

        cp_paths = {}
        for cp_identity in os.listdir(os.path.join(dataset_root, 'ORBench','cp')):
            if cp_identity not in cp_paths.keys():
                cp_paths[cp_identity] = []
            identity_paths = os.listdir(os.path.join(dataset_root, 'ORBench','cp', cp_identity))
            for path in identity_paths:
                cp_paths[cp_identity].append(os.path.join(dataset_root,'ORBench','cp', cp_identity, path))

        sk_paths = {}
        for sk_identity in os.listdir(os.path.join(dataset_root, 'ORBench','sk')):
            if sk_identity not in sk_paths.keys():
                sk_paths[sk_identity] = []
            identity_paths = os.listdir(os.path.join(dataset_root, 'ORBench','sk', sk_identity))
            for path in identity_paths:
                sk_paths[sk_identity].append(os.path.join(dataset_root,'ORBench','sk', sk_identity, path))

        return nir_paths, cp_paths, sk_paths

    def random_sampling(self):
        print("Random Sampling Processing...")
        for i in range(len(self.dataset)):
            # Normalize path separators for cross-platform compatibility
            rgb_path = self.dataset[i][2].replace('\\', '/')
            # Extract identity from path: vis/0001/xxx.jpg -> 0001
            real_identity = rgb_path.split("/vis/")[-1].split("/")[0]
            self.dataset[i] = list(self.dataset[i])
            self.dataset[i][3] = random.sample(self.nir_paths[real_identity], 1)[0]
            self.dataset[i][4] = random.sample(self.cp_paths[real_identity], 1)[0]
            self.dataset[i][5] = random.sample(self.sk_paths[real_identity], 1)[0]
            self.dataset[i] = tuple(self.dataset[i])
        print("Random Sampling Completed!")

    def _build_random_masked_tokens_and_labels(self, tokens):
        """
        Masking some random tokens for Language Model task with probabilities as in the original BERT paper.
        :param tokens: list of int, tokenized sentence.
        :return: (list of int, list of int), masked tokens and related labels for MLM prediction
        """
        mask = self.tokenizer.encoder["<|mask|>"]
        token_range = list(range(1, len(self.tokenizer.encoder) - 3))  # 1 ~ 49405

        labels = []
        for i, token in enumerate(tokens):
            if 0 < token < 49405:
                prob = random.random()
                # mask token with 15% probability
                if prob < 0.15:
                    prob /= 0.15

                    # 80% randomly change token to mask token
                    if prob < 0.8:
                        tokens[i] = mask

                    # 10% randomly change token to random token
                    elif prob < 0.9:
                        tokens[i] = random.choice(token_range)

                    # -> rest 10% randomly keep current token

                    # append current token to output (we will predict these later)
                    labels.append(token)
                else:
                    # no masking token (will be ignored by loss function later)
                    labels.append(0)
            else:
                labels.append(0)

        if all(l == 0 for l in labels):
            # at least mask 1
            labels[1] = tokens[1]
            tokens[1] = mask

        return torch.tensor(tokens)

    def __getitem__(self, index):
        pid, image_id, rgb_path, nir_path, cp_path, sk_path, caption = self.dataset[index]
        rgb = read_image(rgb_path)
        nir = read_image(nir_path)
        cp = read_image(cp_path)
        sk = read_image(sk_path)
        if self.transform is not None:
            rgb = self.transform(rgb)
            nir = self.transform(nir)
            cp = self.transform(cp)
            sk = self.transform(sk)
        tokens = tokenize(caption, tokenizer=self.tokenizer, text_length=self.text_length, truncate=self.truncate)
        tokens = self._build_random_masked_tokens_and_labels(tokens.cpu().numpy())
        ret = {
            'pids': pid,
            'image_ids': image_id,
            'rgbs': rgb,
            'nirs': nir,
            'cps': cp,
            'sks': sk,
            'caption_ids': tokens,
        }
        return ret


class GalleryDataset(Dataset):
    def __init__(self, image_pids, img_paths, transform=None):
        self.image_pids = image_pids
        self.img_paths = img_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_pids)

    def __getitem__(self, index):
        pid, img_path = self.image_pids[index], self.img_paths[index]
        img = read_image(img_path)
        if self.transform is not None:
            img = self.transform(img)
        return pid, img


class SingleQueryVisionDataset(Dataset):
    def __init__(self,
                 queries,
                 transform=None):
        self.queries = queries
        self.transform = transform

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, index):
        query = self.queries[index]
        pid = query[0]
        img_path = query[1]
        img = read_image(img_path)
        if self.transform is not None:
            img = self.transform(img)
        return pid, img


class SingleQueryTextDataset(Dataset):
    def __init__(self,
                 queries,
                 text_length: int = 77,
                 truncate: bool = True):
        self.queries = queries
        self.text_length = text_length
        self.truncate = truncate
        self.tokenizer = SimpleTokenizer()

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, index):
        query = self.queries[index]
        pid = query[0]
        caption = query[1]
        caption = tokenize(caption, tokenizer=self.tokenizer, text_length=self.text_length, truncate=self.truncate)
        return pid, caption


class TwoQueryAllVisionDataset(Dataset):
    def __init__(self,
                 queries,
                 transform=None):
        self.queries = queries
        self.transform = transform

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, index):
        query = self.queries[index]
        pid = query[0]
        img1_path = query[1]
        img2_path = query[2]
        img1 = read_image(img1_path)
        img2 = read_image(img2_path)
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        return pid, img1, img2


class TwoQueryVisionTextDataset(Dataset):
    def __init__(self,
                 queries,
                 transform=None,
                 text_length: int = 77,
                 truncate: bool = True
                 ):
        self.queries = queries
        self.transform = transform
        self.text_length = text_length
        self.truncate = truncate
        self.tokenizer = SimpleTokenizer()

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, index):
        query = self.queries[index]
        pid = query[0]
        img_path = query[1]
        caption = query[2]
        img = read_image(img_path)
        caption = tokenize(caption, tokenizer=self.tokenizer, text_length=self.text_length, truncate=self.truncate)
        if self.transform is not None:
            img = self.transform(img)
        return pid, img, caption


class TwoQueryTextVisionDataset(Dataset):
    def __init__(self,
                 queries,
                 transform=None,
                 text_length: int = 77,
                 truncate: bool = True
                 ):
        self.queries = queries
        self.transform = transform
        self.text_length = text_length
        self.truncate = truncate
        self.tokenizer = SimpleTokenizer()

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, index):
        query = self.queries[index]
        pid = query[0]
        img_path = query[2]
        caption = query[1]
        img = read_image(img_path)
        caption = tokenize(caption, tokenizer=self.tokenizer, text_length=self.text_length, truncate=self.truncate)
        if self.transform is not None:
            img = self.transform(img)
        return pid, caption, img



class ThreeQueryAllVisionDataset(Dataset):
    def __init__(self,
                 queries,
                 transform=None):
        self.queries = queries
        self.transform = transform

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, index):
        query = self.queries[index]
        pid = query[0]
        img1_path = query[1]
        img2_path = query[2]
        img3_path = query[3]
        img1 = read_image(img1_path)
        img2 = read_image(img2_path)
        img3 = read_image(img3_path)
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            img3 = self.transform(img3)
        return pid, img1, img2, img3


class ThreeQueryVisionVisionTextDataset(Dataset):
    def __init__(self,
                 queries,
                 transform=None,
                 text_length: int = 77,
                 truncate: bool = True
                 ):
        self.queries = queries
        self.transform = transform
        self.text_length = text_length
        self.truncate = truncate
        self.tokenizer = SimpleTokenizer()

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, index):
        query = self.queries[index]
        pid = query[0]
        img1_path = query[1]
        img2_path = query[2]
        caption = query[3]
        img1 = read_image(img1_path)
        img2 = read_image(img2_path)
        caption = tokenize(caption, tokenizer=self.tokenizer, text_length=self.text_length, truncate=self.truncate)
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        return pid, img1, img2, caption


class ThreeQueryTextVisionVisionDataset(Dataset):
    def __init__(self,
                 queries,
                 transform=None,
                 text_length: int = 77,
                 truncate: bool = True
                 ):
        self.queries = queries
        self.transform = transform
        self.text_length = text_length
        self.truncate = truncate
        self.tokenizer = SimpleTokenizer()

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, index):
        query = self.queries[index]
        pid = query[0]
        img1_path = query[2]
        img2_path = query[3]
        caption = query[1]
        img1 = read_image(img1_path)
        img2 = read_image(img2_path)
        caption = tokenize(caption, tokenizer=self.tokenizer, text_length=self.text_length, truncate=self.truncate)
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        return pid, caption, img1, img2



class FourQueryVisionVisionVisionTextDataset(Dataset):
    def __init__(self,
                 queries,
                 transform=None,
                 text_length: int = 77,
                 truncate: bool = True
                 ):
        self.queries = queries
        self.transform = transform
        self.text_length = text_length
        self.truncate = truncate
        self.tokenizer = SimpleTokenizer()

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, index):
        query = self.queries[index]
        pid = query[0]
        img1_path = query[1]
        img2_path = query[2]
        img3_path = query[3]
        caption = query[4]
        img1 = read_image(img1_path)
        img2 = read_image(img2_path)
        img3 = read_image(img3_path)
        caption = tokenize(caption, tokenizer=self.tokenizer, text_length=self.text_length, truncate=self.truncate)
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            img3 = self.transform(img3)
        return pid,  img1, img2, img3, caption


class FourQueryTextVisionVisionVisionDataset(Dataset):
    def __init__(self,
                 queries,
                 transform=None,
                 text_length: int = 77,
                 truncate: bool = True
                 ):
        self.queries = queries
        self.transform = transform
        self.text_length = text_length
        self.truncate = truncate
        self.tokenizer = SimpleTokenizer()

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, index):
        query = self.queries[index]
        pid = query[0]
        img1_path = query[2]
        img2_path = query[3]
        img3_path = query[4]
        caption = query[1]
        img1 = read_image(img1_path)
        img2 = read_image(img2_path)
        img3 = read_image(img3_path)
        caption = tokenize(caption, tokenizer=self.tokenizer, text_length=self.text_length, truncate=self.truncate)
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            img3 = self.transform(img3)
        return pid, caption, img1, img2, img3


class ImageTextSketchDataset(Dataset):
    """
    Dataset for 3-modality training: RGB, TEXT, SK (Sketch).
    
    This is designed for datasets like CUHK-PEDES that only have
    image, text, and sketch modalities (no NIR or CP).
    
    For missing modalities (NIR, CP), we return the RGB image as placeholder
    so the model can handle them through missing-aware encoding.
    """
    def __init__(self,
                 dataset,
                 transform=None,
                 text_length: int = 77,
                 truncate: bool = True,
                 caption_dict=None):
        self.dataset = list(dataset)  # Make mutable for random sampling
        self.transform = transform
        self.text_length = text_length
        self.truncate = truncate
        self.tokenizer = SimpleTokenizer()
        # Optional: dictionary mapping rgb_path to list of captions for random sampling
        self.caption_dict = caption_dict or {}

    def __len__(self):
        return len(self.dataset)

    def set_caption_dict(self, caption_dict):
        """Set the caption dictionary for random sampling."""
        self.caption_dict = caption_dict

    def random_sampling(self):
        """
        For datasets that need random sampling of captions.
        If caption_dict is set, randomly selects a caption for each sample.
        """
        if not self.caption_dict:
            print("Random Sampling Processing (no-op for 3-modal dataset)...")
            print("Random Sampling Completed!")
            return
        
        print("Random Sampling Processing...")
        for i in range(len(self.dataset)):
            item = list(self.dataset[i])
            rgb_path = item[2]
            
            if rgb_path in self.caption_dict:
                captions = self.caption_dict[rgb_path]
                if isinstance(captions, list) and len(captions) > 0:
                    item[6] = random.choice(captions)
                elif isinstance(captions, str):
                    item[6] = captions
            
            self.dataset[i] = tuple(item)
        print("Random Sampling Completed!")

    def _build_random_masked_tokens_and_labels(self, tokens):
        """
        Masking some random tokens for Language Model task.
        """
        mask = self.tokenizer.encoder["<|mask|>"]
        token_range = list(range(1, len(self.tokenizer.encoder) - 3))

        labels = []
        for i, token in enumerate(tokens):
            if 0 < token < 49405:
                prob = random.random()
                if prob < 0.15:
                    prob /= 0.15
                    if prob < 0.8:
                        tokens[i] = mask
                    elif prob < 0.9:
                        tokens[i] = random.choice(token_range)
                    labels.append(token)
                else:
                    labels.append(0)
            else:
                labels.append(0)

        if all(l == 0 for l in labels):
            labels[1] = tokens[1]
            tokens[1] = mask

        return torch.tensor(tokens)

    def __getitem__(self, index):
        pid, image_id, rgb_path, nir_path, cp_path, sk_path, caption = self.dataset[index]
        
        # Load RGB image
        rgb = read_image(rgb_path)
        
        # Load NIR - for 3-modal datasets, this is a placeholder (same as RGB)
        # The model will handle this through missing-aware encoding
        nir = read_image(nir_path)
        
        # Load CP - for 3-modal datasets, this is a placeholder (same as RGB)
        cp = read_image(cp_path)
        
        # Load Sketch
        sk = read_image(sk_path)
        
        if self.transform is not None:
            rgb = self.transform(rgb)
            nir = self.transform(nir)
            cp = self.transform(cp)
            sk = self.transform(sk)
        
        tokens = tokenize(caption, tokenizer=self.tokenizer, text_length=self.text_length, truncate=self.truncate)
        tokens = self._build_random_masked_tokens_and_labels(tokens.cpu().numpy())
        
        ret = {
            'pids': pid,
            'image_ids': image_id,
            'rgbs': rgb,
            'nirs': nir,
            'cps': cp,
            'sks': sk,
            'caption_ids': tokens,
            # Flag to indicate which modalities are real vs placeholder
            'modality_available': {
                'RGB': True,
                'NIR': False,  # Placeholder
                'CP': False,   # Placeholder
                'SK': True,
                'TEXT': True,
            }
        }
        return ret
