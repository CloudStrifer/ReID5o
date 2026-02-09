"""
ICFG-PEDES Dataset for Text-Image-Sketch Person Re-identification

This dataset contains:
- RGB images from ICFG-PEDES (imgs folder)
- Text captions (ICFG-PEDES.json)
- Sketch images (from data/sketch/aliyun/ICFG)

The dataset is organized as:
data/ICFG-PEDES/
    imgs/
        train/
            0000/
            0001/
            ...
            1040/
        test/
            0000/
            0001/
            ...
            3059/
    ICFG-PEDES.json

ICFG-PEDES.json format:
[
    {
        "split": "train",
        "file_path": "test/0627/0627_010_05_0303afternoon_1591_0.jpg",
        "id": 0,
        "processed_tokens": [["A", "young", ...]],
        "captions": ["A young age woman has black shoulder-length hair..."]
    },
    ...
]

Sketch images are in:
data/sketch/aliyun/ICFG/
    train/
        0000/
        0001/
        ...
    test/
        0000/
        0001/
        ...
"""

import os
import os.path as op
import json
import random
from .bases import BaseDataset


class ICFG_PEDES(BaseDataset):
    """
    ICFG-PEDES Dataset with Sketch modality support.
    
    This dataset supports 3 modalities:
    - RGB: Original person images
    - TEXT: Textual descriptions (1 caption per image)
    - SK: Sketch images (generated separately)
    
    Note: This dataset does NOT have NIR or CP modalities.
    
    Dataset structure:
    - Train: ~34,674 images from 4,102 identities
    - Test: ~19,848 images from 3,060 identities (gallery + query)
    """
    dataset_dir = 'ICFG-PEDES'
    sketch_dir = 'sketch/aliyun/ICFG'
    
    def __init__(self, root='', verbose=True):
        super(ICFG_PEDES, self).__init__()
        self.dataset_root = op.join(root, self.dataset_dir)
        self.sketch_root = op.join(root, self.sketch_dir)
        self.imgs_root = op.join(self.dataset_root, 'imgs')
        self.anno_path = op.join(self.dataset_root, 'ICFG-PEDES.json')
        
        # Load annotations
        self.annotations = self._load_annotations()
        
        # Build sketch path mapping
        self.sketch_paths = self._build_sketch_paths()
        
        # Split data into train/test based on 'split' field
        self.train_annos, self.test_annos = self._split_data()
        
        # Process annotations
        self.train, self.train_id_container = self._process_anno(self.train_annos, training=True)
        self.test, self.test_id_container = self._process_test_anno(self.test_annos)
        
        if verbose:
            self.logger.info("=> ICFG-PEDES Images and Captions are loaded")
            self.show_dataset_info()
    
    def _load_annotations(self):
        """Load ICFG-PEDES.json"""
        with open(self.anno_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    
    def _build_sketch_paths(self):
        """
        Build a mapping from image file_path to sketch path.
        
        The sketch directory structure mirrors the imgs directory:
        data/sketch/aliyun/ICFG/
            train/
                0000/
                    0000_002_01_0303morning_0009_0.jpg
                    ...
            test/
                0000/
                    ...
        """
        sketch_paths = {}
        
        if not op.exists(self.sketch_root):
            self.logger.warning(f"Sketch directory not found: {self.sketch_root}")
            return sketch_paths
        
        # Walk through all directories
        for split in ['train', 'test']:
            split_path = op.join(self.sketch_root, split)
            if not op.exists(split_path):
                continue
                
            for identity_folder in os.listdir(split_path):
                identity_path = op.join(split_path, identity_folder)
                if not op.isdir(identity_path):
                    continue
                
                for filename in os.listdir(identity_path):
                    if filename.endswith(('.jpg', '.png', '.jpeg')):
                        # Create key matching the file_path in JSON
                        # e.g., "train/0000/0000_002_01_0303morning_0009_0.jpg"
                        key = f"{split}/{identity_folder}/{filename}"
                        sketch_paths[key] = op.join(identity_path, filename)
        
        return sketch_paths
    
    def _get_sketch_path(self, file_path):
        """
        Get sketch path for a given image file_path.
        
        Args:
            file_path: Path like 'train/0000/0000_002_01_0303morning_0009_0.jpg'
            
        Returns:
            Sketch path or None if not found
        """
        # Normalize path separators
        file_path = file_path.replace('\\', '/')
        
        # Direct match
        if file_path in self.sketch_paths:
            return self.sketch_paths[file_path]
        
        # Try matching just the filename
        filename = op.basename(file_path)
        for key, path in self.sketch_paths.items():
            if key.endswith(filename):
                return path
        
        return None
    
    def _split_data(self):
        """
        Split data into train/test based on 'split' field in JSON.
        """
        train_data = []
        test_data = []
        
        for item in self.annotations:
            split = item.get('split', 'train')
            if split == 'train':
                train_data.append(item)
            elif split == 'test':
                test_data.append(item)
            else:
                # Default to train if unknown
                train_data.append(item)
        
        self.logger.info(f"ICFG-PEDES split: train={len(train_data)}, test={len(test_data)}")
        
        return train_data, test_data
    
    def _process_anno(self, annos, training=False):
        """
        Process annotations for training.
        
        Returns dataset tuples: (pid, image_id, rgb_path, nir_path, cp_path, sk_path, caption)
        Note: nir_path and cp_path will be set to RGB as placeholder since ICFG-PEDES 
        doesn't have these modalities.
        """
        pid_container = set()
        dataset = []
        image_id = 0
        
        # Build pid mapping (original id to 0-indexed)
        # Note: ICFG-PEDES uses 'id' field which is already 0-indexed for each split
        unique_pids = sorted(set(item['id'] for item in annos))
        pid_map = {pid: idx for idx, pid in enumerate(unique_pids)}
        
        for anno in annos:
            original_pid = anno['id']
            pid = pid_map[original_pid]
            pid_container.add(pid)
            
            # RGB image path
            file_path = anno['file_path'].replace('\\', '/')
            rgb_path = op.join(self.imgs_root, file_path)
            
            # Sketch path (if available)
            sk_path = self._get_sketch_path(file_path)
            if sk_path is None:
                # If no sketch, use RGB as fallback
                sk_path = rgb_path
            
            # NIR and CP don't exist for this dataset - use RGB as placeholder
            # The model will handle missing modalities through the missing-aware encoding
            nir_path = rgb_path  # Placeholder
            cp_path = rgb_path   # Placeholder
            
            # Get caption (ICFG-PEDES has 1 caption per image in a list)
            captions = anno.get('captions', [])
            if isinstance(captions, list) and len(captions) > 0:
                caption = captions[0]  # Take the first caption
            else:
                caption = captions if isinstance(captions, str) else ""
            
            dataset.append((pid, image_id, rgb_path, nir_path, cp_path, sk_path, caption))
            image_id += 1
        
        return dataset, pid_container
    
    def _process_test_anno(self, annos):
        """
        Process test annotations.
        
        For ICFG-PEDES, we use:
        - Gallery: All test RGB images
        - Query: Text descriptions (and sketches)
        
        Returns dict with gallery and query information.
        """
        pid_container = set()
        
        # Build pid mapping for test set
        unique_pids = sorted(set(item['id'] for item in annos))
        pid_map = {pid: idx for idx, pid in enumerate(unique_pids)}
        
        # Process gallery (RGB images) - use all test images
        gallery_paths = []
        gallery_pids = []
        
        # Track unique images for gallery (avoid duplicates)
        seen_paths = set()
        
        for item in annos:
            pid = pid_map[item['id']]
            pid_container.add(pid)
            file_path = item['file_path'].replace('\\', '/')
            img_path = op.join(self.imgs_root, file_path)
            
            # Add to gallery (each unique image once)
            if img_path not in seen_paths:
                gallery_paths.append(img_path)
                gallery_pids.append(pid)
                seen_paths.add(img_path)
        
        # Process queries for different modality combinations
        queries = self._build_query_combinations(annos, pid_map)
        
        dataset = {
            "gallery_pids": gallery_pids,
            "gallery_paths": gallery_paths,
            "queries": queries
        }
        
        return dataset, pid_container
    
    def _build_query_combinations(self, query_items, pid_map):
        """
        Build query combinations for ICFG-PEDES.
        
        Since ICFG-PEDES only has RGB, TEXT, and SK, we create queries for:
        - TEXT (single modality text query)
        - SK (single modality sketch query)
        - TEXT+SK and SK+TEXT (two modality combinations)
        
        Note: NIR and CP queries will be empty.
        """
        queries = {
            # Single modalities (using existing naming for compatibility)
            'NIR': [],      # Not available - will be empty
            'CP': [],       # Not available - will be empty
            'SK': [],       # Sketch queries
            'TEXT': [],     # Text queries
            
            # Two modalities
            'NIR+CP': [], 'CP+NIR': [],
            'NIR+SK': [], 'SK+NIR': [],
            'NIR+TEXT': [], 'TEXT+NIR': [],
            'CP+SK': [], 'SK+CP': [],
            'CP+TEXT': [], 'TEXT+CP': [],
            'SK+TEXT': [], 'TEXT+SK': [],
            
            # Three modalities
            'NIR+CP+SK': [], 'CP+NIR+SK': [], 'SK+NIR+CP': [],
            'NIR+CP+TEXT': [], 'CP+NIR+TEXT': [], 'TEXT+NIR+CP': [],
            'NIR+SK+TEXT': [], 'SK+NIR+TEXT': [], 'TEXT+NIR+SK': [],
            'CP+SK+TEXT': [], 'SK+CP+TEXT': [], 'TEXT+CP+SK': [],
            
            # Four modalities
            'NIR+CP+SK+TEXT': [], 'CP+NIR+SK+TEXT': [],
            'SK+NIR+CP+TEXT': [], 'TEXT+NIR+CP+SK': [],
        }
        
        # Track which SK paths we've already added to avoid duplicates
        added_sk_queries = set()
        
        for item in query_items:
            pid = pid_map[item['id']]
            
            # Get paths
            file_path = item['file_path'].replace('\\', '/')
            rgb_path = op.join(self.imgs_root, file_path)
            sk_path = self._get_sketch_path(file_path)
            if sk_path is None:
                sk_path = rgb_path  # Fallback to RGB if no sketch found
            
            # Get caption
            captions = item.get('captions', [])
            if isinstance(captions, list) and len(captions) > 0:
                caption = captions[0]
            else:
                caption = captions if isinstance(captions, str) else ""
            
            # TEXT query: (pid, caption)
            queries['TEXT'].append((pid, caption))
            
            # SK query: (pid, sk_path) - add once per unique (pid, sk_path)
            # Always add SK queries regardless of whether it's a real sketch or fallback
            sk_key = (pid, sk_path)
            if sk_key not in added_sk_queries:
                queries['SK'].append((pid, sk_path))
                added_sk_queries.add(sk_key)
            
            # TEXT+SK: (pid, caption, sk_path)
            queries['TEXT+SK'].append((pid, caption, sk_path))
            
            # SK+TEXT: (pid, sk_path, caption)
            queries['SK+TEXT'].append((pid, sk_path, caption))
        
        return queries
    
    def random_sampling(self):
        """
        Random sampling for training.
        For ICFG-PEDES, each image has only 1 caption, so this is a no-op.
        But we keep it for API consistency.
        """
        print("Random Sampling Processing for ICFG-PEDES...")
        # ICFG-PEDES has only 1 caption per image, so no need to resample
        print("Random Sampling Completed!")

    def show_dataset_info(self):
        """Override to show ICFG-PEDES specific statistics."""
        from prettytable import PrettyTable
        
        num_train_pids = len(self.train_id_container)
        num_train_imgs = len(self.train_annos)
        num_train_captions = len(self.train_annos)  # 1 caption per image
        
        # Count test queries
        queries_num = 0
        queries = self.test['queries']
        for key, query_list in queries.items():
            if len(query_list) > 0:
                queries_num += len(query_list)
        
        num_test_pids = len(self.test_id_container)
        num_test_imgs = len(self.test['gallery_paths'])
        num_test_queries = queries_num
        
        # Count available modalities
        available_queries = [k for k, v in queries.items() if len(v) > 0]
        
        self.logger.info(f"{self.__class__.__name__} Dataset statistics:")
        table = PrettyTable(['subset', 'ids', 'images', 'captions'])
        table.add_row(['train', num_train_pids, num_train_imgs, num_train_captions])
        table.add_row(['test', num_test_pids, num_test_imgs, num_test_queries])
        self.logger.info('\n' + str(table))
        self.logger.info(f"Available query modalities: {available_queries}")
        self.logger.info(f"Sketch images found: {len(self.sketch_paths)}")


class ICFG_PEDES_ThreeModal(ICFG_PEDES):
    """
    ICFG-PEDES variant that explicitly handles 3 modalities: RGB, TEXT, SK.
    
    This class provides additional utilities for 3-modal experiments
    where NIR and CP are explicitly marked as missing.
    """
    
    def __init__(self, root='', verbose=True):
        super().__init__(root, verbose)
        self.num_modalities = 3  # RGB, TEXT, SK
        self.available_modalities = ['RGB', 'TEXT', 'SK']
        self.missing_modalities = ['NIR', 'CP']
