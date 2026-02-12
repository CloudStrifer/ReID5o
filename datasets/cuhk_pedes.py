"""
CUHK-PEDES Dataset for Text-Image-Sketch Person Re-identification

This dataset contains:
- RGB images from CUHK-PEDES (imgs folder)
- Text captions (caption_all.json)
- Sketch images (from data/sketch/aliyun/CUHK/imgs/)

The dataset is organized as:
data/CUHK-PEDES/
    imgs/
        cam_a/
        cam_b/
        test_query/
        train_query/
        ...
    caption_all.json

caption_all.json format:
[
    {
        "id": 1,
        "file_path": "test_query/p10376_s14337.jpg",
        "captions": ["caption1", "caption2"]
    },
    ...
]

Sketch images are in:
data/sketch/aliyun/CUHK/imgs/
    cam_a/
    cam_b/
    test_query/
    ...
"""

import os
import os.path as op
import json
import random
from .bases import BaseDataset


class CUHK_PEDES(BaseDataset):
    """
    CUHK-PEDES Dataset with Sketch modality support.
    
    This dataset supports 3 modalities:
    - RGB: Original person images
    - TEXT: Textual descriptions (2 captions per image)
    - SK: Sketch images (generated separately)
    
    Note: This dataset does NOT have NIR or CP modalities.
    """
    dataset_dir = 'CUHK-PEDES'
    sketch_dir = 'sketch/aliyun/CUHK/imgs'  # Note: includes /imgs subfolder
    
    def __init__(self, root='', verbose=True):
        super(CUHK_PEDES, self).__init__()
        self.dataset_root = op.join(root, self.dataset_dir)
        self.sketch_root = op.join(root, self.sketch_dir)
        self.imgs_root = op.join(self.dataset_root, 'imgs')
        self.caption_path = op.join(self.dataset_root, 'caption_all.json')
        
        # Load captions
        self.captions_data = self._load_captions()
        
        # Build sketch path mapping
        self.sketch_paths = self._build_sketch_paths()
        
        # Split data into train/val/test based on file_path prefix
        self.train_annos, self.val_annos, self.test_annos = self._split_data()
        
        # Process annotations
        self.train, self.train_id_container = self._process_anno(self.train_annos, training=True)
        self.test, self.test_id_container = self._process_test_anno(self.test_annos)
        
        if verbose:
            self.logger.info("=> CUHK-PEDES Images and Captions are loaded")
            self.show_dataset_info()
    
    def _load_captions(self):
        """Load caption_all.json"""
        with open(self.caption_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    
    def _build_sketch_paths(self):
        """
        Build a mapping from image filename to sketch path.
        
        Sketch structure:
        data/sketch/aliyun/CUHK/imgs/
            cam_a/
                filename.jpg
            cam_b/
                filename.jpg
            test_query/
                filename.jpg
            ...
        """
        sketch_paths = {}
        
        if not op.exists(self.sketch_root):
            self.logger.warning(f"Sketch directory not found: {self.sketch_root}")
            return sketch_paths
        
        # Walk through all subdirectories (cam_a, cam_b, test_query, etc.)
        for subfolder in os.listdir(self.sketch_root):
            subfolder_path = op.join(self.sketch_root, subfolder)
            if not op.isdir(subfolder_path):
                continue
            
            for filename in os.listdir(subfolder_path):
                if filename.endswith(('.jpg', '.png', '.jpeg')):
                    # Key format: "subfolder/filename" (e.g., "cam_a/0001001.jpg")
                    key = f"{subfolder}/{filename}"
                    sketch_paths[key] = op.join(subfolder_path, filename)
                    # Also add just filename for flexible matching
                    sketch_paths[filename] = op.join(subfolder_path, filename)
        
        return sketch_paths
    
    def _get_sketch_path(self, file_path):
        """
        Get sketch path for a given image file_path.
        
        Args:
            file_path: Path like 'test_query/p10376_s14337.jpg'
            
        Returns:
            Sketch path or None if not found
        """
        file_path = file_path.replace('\\', '/')
        
        # Strategy 1: Direct match with full path
        if file_path in self.sketch_paths:
            return self.sketch_paths[file_path]
        
        # Strategy 2: Match with subfolder/filename
        parts = file_path.split('/')
        if len(parts) >= 2:
            subfolder = parts[-2]
            filename = parts[-1]
            key = f"{subfolder}/{filename}"
            if key in self.sketch_paths:
                return self.sketch_paths[key]
        
        # Strategy 3: Just filename match
        filename = op.basename(file_path)
        if filename in self.sketch_paths:
            return self.sketch_paths[filename]
        
        return None
    
    def _split_data(self):
        """
        Split data into train/val/test based on file_path prefix.
        """
        train_data = []
        val_data = []
        test_data = []
        
        for item in self.captions_data:
            file_path = item['file_path'].replace('\\', '/')
            
            if 'train' in file_path.lower():
                train_data.append(item)
            elif 'val' in file_path.lower():
                val_data.append(item)
            elif 'test' in file_path.lower():
                test_data.append(item)
            else:
                train_data.append(item)
        
        self.logger.info(f"CUHK-PEDES split: train={len(train_data)}, val={len(val_data)}, test={len(test_data)}")
        
        return train_data, val_data, test_data
    
    def _process_anno(self, annos, training=False):
        """Process annotations for training."""
        pid_container = set()
        dataset = []
        image_id = 0
        
        unique_pids = sorted(set(item['id'] for item in annos))
        pid_map = {pid: idx for idx, pid in enumerate(unique_pids)}
        
        for anno in annos:
            original_pid = anno['id']
            pid = pid_map[original_pid]
            pid_container.add(pid)
            
            file_path = anno['file_path'].replace('\\', '/')
            rgb_path = op.join(self.imgs_root, file_path)
            
            sk_path = self._get_sketch_path(file_path)
            if sk_path is None:
                sk_path = rgb_path
            
            nir_path = rgb_path
            cp_path = rgb_path
            
            captions = anno['captions']
            if isinstance(captions, list) and len(captions) > 0:
                caption = random.choice(captions)
            else:
                caption = captions if isinstance(captions, str) else ""
            
            dataset.append((pid, image_id, rgb_path, nir_path, cp_path, sk_path, caption))
            image_id += 1
        
        return dataset, pid_container
    
    def _process_test_anno(self, annos):
        """Process test annotations."""
        pid_container = set()
        
        query_items = []
        gallery_items = []
        
        for item in annos:
            file_path = item['file_path'].replace('\\', '/')
            if 'query' in file_path.lower():
                query_items.append(item)
            elif 'gallery' in file_path.lower():
                gallery_items.append(item)
            else:
                gallery_items.append(item)
        
        if len(query_items) == 0 and len(gallery_items) > 0:
            query_items = annos
            gallery_items = annos
        
        if len(gallery_items) == 0:
            gallery_items = annos
        
        all_items = list(set([item['id'] for item in query_items + gallery_items]))
        unique_pids = sorted(all_items)
        pid_map = {pid: idx for idx, pid in enumerate(unique_pids)}
        
        gallery_paths = []
        gallery_pids = []
        
        for item in gallery_items:
            pid = pid_map[item['id']]
            pid_container.add(pid)
            file_path = item['file_path'].replace('\\', '/')
            img_path = op.join(self.imgs_root, file_path)
            gallery_paths.append(img_path)
            gallery_pids.append(pid)
        
        queries = self._build_query_combinations(query_items, pid_map)
        
        dataset = {
            "gallery_pids": gallery_pids,
            "gallery_paths": gallery_paths,
            "queries": queries
        }
        
        return dataset, pid_container
    
    def _build_query_combinations(self, query_items, pid_map):
        """Build query combinations for CUHK-PEDES."""
        queries = {
            'NIR': [], 'CP': [], 'SK': [], 'TEXT': [],
            'NIR+CP': [], 'CP+NIR': [], 'NIR+SK': [], 'SK+NIR': [],
            'NIR+TEXT': [], 'TEXT+NIR': [], 'CP+SK': [], 'SK+CP': [],
            'CP+TEXT': [], 'TEXT+CP': [], 'SK+TEXT': [], 'TEXT+SK': [],
            'NIR+CP+SK': [], 'CP+NIR+SK': [], 'SK+NIR+CP': [],
            'NIR+CP+TEXT': [], 'CP+NIR+TEXT': [], 'TEXT+NIR+CP': [],
            'NIR+SK+TEXT': [], 'SK+NIR+TEXT': [], 'TEXT+NIR+SK': [],
            'CP+SK+TEXT': [], 'SK+CP+TEXT': [], 'TEXT+CP+SK': [],
            'NIR+CP+SK+TEXT': [], 'CP+NIR+SK+TEXT': [],
            'SK+NIR+CP+TEXT': [], 'TEXT+NIR+CP+SK': [],
        }
        
        added_sk_queries = set()
        
        for item in query_items:
            pid = pid_map[item['id']]
            
            file_path = item['file_path'].replace('\\', '/')
            rgb_path = op.join(self.imgs_root, file_path)
            sk_path = self._get_sketch_path(file_path)
            if sk_path is None:
                sk_path = rgb_path
            
            captions = item['captions']
            if isinstance(captions, list) and len(captions) > 0:
                for caption in captions:
                    queries['TEXT'].append((pid, caption))
                    queries['TEXT+SK'].append((pid, caption, sk_path))
                    queries['SK+TEXT'].append((pid, sk_path, caption))
                
                sk_key = (pid, sk_path)
                if sk_key not in added_sk_queries:
                    queries['SK'].append((pid, sk_path))
                    added_sk_queries.add(sk_key)
            else:
                caption = captions if isinstance(captions, str) else ""
                queries['TEXT'].append((pid, caption))
                
                sk_key = (pid, sk_path)
                if sk_key not in added_sk_queries:
                    queries['SK'].append((pid, sk_path))
                    added_sk_queries.add(sk_key)
                    
                queries['TEXT+SK'].append((pid, caption, sk_path))
                queries['SK+TEXT'].append((pid, sk_path, caption))
        
        return queries
    
    def random_sampling(self):
        """Random sampling for training."""
        if not hasattr(self, '_rgb_to_anno'):
            self._rgb_to_anno = {}
            for anno in self.train_annos:
                file_path = anno['file_path'].replace('\\', '/')
                rgb_path = op.join(self.imgs_root, file_path)
                self._rgb_to_anno[rgb_path] = anno
        
        print("Random Sampling Processing for CUHK-PEDES...")
        train_list = list(self.train)
        for i in range(len(train_list)):
            item = list(train_list[i])
            rgb_path = item[2]
            
            anno = self._rgb_to_anno.get(rgb_path)
            if anno:
                captions = anno['captions']
                if isinstance(captions, list) and len(captions) > 0:
                    caption = random.choice(captions)
                else:
                    caption = captions if isinstance(captions, str) else ""
                item[6] = caption
            
            train_list[i] = tuple(item)
        self.train = train_list
        print("Random Sampling Completed!")

    def show_dataset_info(self):
        """Show CUHK-PEDES specific statistics."""
        from prettytable import PrettyTable
        
        num_train_pids = len(self.train_id_container)
        num_train_imgs = len(self.train_annos)
        num_train_captions = sum(len(anno.get('captions', [])) for anno in self.train_annos)
        
        queries_num = 0
        queries = self.test['queries']
        for key, query_list in queries.items():
            if len(query_list) > 0:
                queries_num += len(query_list)
        
        num_test_pids = len(self.test_id_container)
        num_test_imgs = len(self.test['gallery_paths'])
        num_test_queries = queries_num
        
        available_queries = [k for k, v in queries.items() if len(v) > 0]
        
        self.logger.info(f"{self.__class__.__name__} Dataset statistics:")
        table = PrettyTable(['subset', 'ids', 'images', 'captions'])
        table.add_row(['train', num_train_pids, num_train_imgs, num_train_captions])
        table.add_row(['test', num_test_pids, num_test_imgs, num_test_queries])
        self.logger.info('\n' + str(table))
        self.logger.info(f"Available query modalities: {available_queries}")
        self.logger.info(f"Sketch images found: {len(self.sketch_paths) // 2}")  # Divide by 2 since we add both key formats


class CUHK_PEDES_ThreeModal(CUHK_PEDES):
    """CUHK-PEDES variant for 3-modal experiments."""
    
    def __init__(self, root='', verbose=True):
        super().__init__(root, verbose)
        self.num_modalities = 3
        self.available_modalities = ['RGB', 'TEXT', 'SK']
        self.missing_modalities = ['NIR', 'CP']
