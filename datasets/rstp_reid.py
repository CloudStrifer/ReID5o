"""
RSTPReid Dataset for Text-Image-Sketch Person Re-identification

This dataset contains:
- RGB images from RSTPReid (imgs folder)
- Text captions (data_captions.json)
- Sketch images (from data/sketch/aliyun/RSTPReid/imgs)

The dataset is organized as:
data/RSTPReid/
    imgs/
        0000_c1_0004.jpg
        0000_c5_0022.jpg
        0000_c7_0015.jpg
        0001_c1_0003.jpg
        ...
    data_captions.json

data_captions.json format:
[
    {
        "id": 0,
        "img_path": "0000_c14_0031.jpg",
        "captions": ["caption1", "caption2"],
        "split": "train"
    },
    ...
]

Note: 
- Each pedestrian has 5 images
- The first 4 characters of the image name represent the pedestrian ID (e.g., "0000" in "0000_c1_0004.jpg")
- Each image has 2 text captions

Sketch images are in:
data/sketch/aliyun/RSTPReid/imgs/
    0000_c1_0004.jpg
    0000_c5_0022.jpg
    ...
"""

import os
import os.path as op
import json
import random
from .bases import BaseDataset


class RSTPReid(BaseDataset):
    """
    RSTPReid Dataset with Sketch modality support.
    
    This dataset supports 3 modalities:
    - RGB: Original person images
    - TEXT: Textual descriptions (2 captions per image)
    - SK: Sketch images (generated separately)
    
    Note: This dataset does NOT have NIR or CP modalities.
    
    Dataset structure:
    - 4,101 identities
    - 20,505 images (5 images per identity)
    - 41,010 captions (2 captions per image)
    - Split into train/val/test by 'split' field
    """
    dataset_dir = 'RSTPReid'
    sketch_dir = 'sketch/aliyun/RSTPReid/imgs'
    
    def __init__(self, root='', verbose=True):
        super(RSTPReid, self).__init__()
        self.dataset_root = op.join(root, self.dataset_dir)
        self.sketch_root = op.join(root, self.sketch_dir)
        self.imgs_root = op.join(self.dataset_root, 'imgs')
        self.anno_path = op.join(self.dataset_root, 'data_captions.json')
        
        # Load annotations
        self.annotations = self._load_annotations()
        
        # Build sketch path mapping
        self.sketch_paths = self._build_sketch_paths()
        
        # Split data into train/val/test based on 'split' field
        self.train_annos, self.val_annos, self.test_annos = self._split_data()
        
        # Process annotations
        self.train, self.train_id_container = self._process_anno(self.train_annos, training=True)
        self.test, self.test_id_container = self._process_test_anno(self.test_annos)
        
        if verbose:
            self.logger.info("=> RSTPReid Images and Captions are loaded")
            self.show_dataset_info()
    
    def _load_annotations(self):
        """Load data_captions.json"""
        with open(self.anno_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    
    def _build_sketch_paths(self):
        """
        Build a mapping from image filename to sketch path.
        
        The sketch directory structure is flat (all images in one folder):
        data/sketch/aliyun/RSTPReid/imgs/
            0000_c1_0004.jpg
            0000_c5_0022.jpg
            ...
        """
        sketch_paths = {}
        
        if not op.exists(self.sketch_root):
            self.logger.warning(f"Sketch directory not found: {self.sketch_root}")
            return sketch_paths
        
        for filename in os.listdir(self.sketch_root):
            if filename.endswith(('.jpg', '.png', '.jpeg')):
                # Key is just the filename (matching img_path in JSON)
                sketch_paths[filename] = op.join(self.sketch_root, filename)
        
        return sketch_paths
    
    def _get_sketch_path(self, img_path):
        """
        Get sketch path for a given image filename.
        
        Args:
            img_path: Image filename like '0000_c14_0031.jpg'
            
        Returns:
            Sketch path or None if not found
        """
        # Normalize path - extract just the filename
        filename = op.basename(img_path).replace('\\', '/')
        
        if filename in self.sketch_paths:
            return self.sketch_paths[filename]
        
        return None
    
    def _split_data(self):
        """
        Split data into train/val/test based on 'split' field in JSON.
        """
        train_data = []
        val_data = []
        test_data = []
        
        for item in self.annotations:
            split = item.get('split', 'train')
            if split == 'train':
                train_data.append(item)
            elif split == 'val':
                val_data.append(item)
            elif split == 'test':
                test_data.append(item)
            else:
                # Default to train if unknown
                train_data.append(item)
        
        self.logger.info(f"RSTPReid split: train={len(train_data)}, val={len(val_data)}, test={len(test_data)}")
        
        return train_data, val_data, test_data
    
    def _process_anno(self, annos, training=False):
        """
        Process annotations for training.
        
        Returns dataset tuples: (pid, image_id, rgb_path, nir_path, cp_path, sk_path, caption)
        Note: nir_path and cp_path will be set to RGB as placeholder since RSTPReid 
        doesn't have these modalities.
        """
        pid_container = set()
        dataset = []
        image_id = 0
        
        # Build pid mapping (original id to 0-indexed)
        unique_pids = sorted(set(item['id'] for item in annos))
        pid_map = {pid: idx for idx, pid in enumerate(unique_pids)}
        
        for anno in annos:
            original_pid = anno['id']
            pid = pid_map[original_pid]
            pid_container.add(pid)
            
            # RGB image path - img_path is just the filename
            img_path = anno['img_path'].replace('\\', '/')
            rgb_path = op.join(self.imgs_root, img_path)
            
            # Sketch path (if available)
            sk_path = self._get_sketch_path(img_path)
            if sk_path is None:
                # If no sketch, use RGB as fallback
                sk_path = rgb_path
            
            # NIR and CP don't exist for this dataset - use RGB as placeholder
            nir_path = rgb_path  # Placeholder
            cp_path = rgb_path   # Placeholder
            
            # Get caption (RSTPReid has 2 captions per image, randomly select one for training)
            captions = anno.get('captions', [])
            if isinstance(captions, list) and len(captions) > 0:
                caption = random.choice(captions)
            else:
                caption = captions if isinstance(captions, str) else ""
            
            dataset.append((pid, image_id, rgb_path, nir_path, cp_path, sk_path, caption))
            image_id += 1
        
        return dataset, pid_container
    
    def _process_test_anno(self, annos):
        """
        Process test annotations.
        
        For RSTPReid, we use:
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
            img_path = item['img_path'].replace('\\', '/')
            rgb_path = op.join(self.imgs_root, img_path)
            
            # Add to gallery (each unique image once)
            if rgb_path not in seen_paths:
                gallery_paths.append(rgb_path)
                gallery_pids.append(pid)
                seen_paths.add(rgb_path)
        
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
        Build query combinations for RSTPReid.
        
        Since RSTPReid only has RGB, TEXT, and SK, we create queries for:
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
            img_path = item['img_path'].replace('\\', '/')
            rgb_path = op.join(self.imgs_root, img_path)
            sk_path = self._get_sketch_path(img_path)
            if sk_path is None:
                sk_path = rgb_path  # Fallback to RGB if no sketch found
            
            # Get captions - RSTPReid has 2 captions per image
            captions = item.get('captions', [])
            if isinstance(captions, list) and len(captions) > 0:
                # Use all captions as separate queries
                for caption in captions:
                    # TEXT query: (pid, caption)
                    queries['TEXT'].append((pid, caption))
                    
                    # TEXT+SK: (pid, caption, sk_path)
                    queries['TEXT+SK'].append((pid, caption, sk_path))
                    
                    # SK+TEXT: (pid, sk_path, caption)
                    queries['SK+TEXT'].append((pid, sk_path, caption))
            else:
                caption = captions if isinstance(captions, str) else ""
                queries['TEXT'].append((pid, caption))
                queries['TEXT+SK'].append((pid, caption, sk_path))
                queries['SK+TEXT'].append((pid, sk_path, caption))
            
            # SK query: (pid, sk_path) - add once per unique (pid, sk_path)
            # Always add SK queries regardless of whether it's a real sketch or fallback
            sk_key = (pid, sk_path)
            if sk_key not in added_sk_queries:
                queries['SK'].append((pid, sk_path))
                added_sk_queries.add(sk_key)
        
        return queries
    
    def random_sampling(self):
        """
        Random sampling for training.
        For RSTPReid, we randomly select one of the 2 captions for each image.
        """
        # Build a quick lookup from rgb_path to original annotation
        if not hasattr(self, '_rgb_to_anno'):
            self._rgb_to_anno = {}
            for anno in self.train_annos:
                img_path = anno['img_path'].replace('\\', '/')
                rgb_path = op.join(self.imgs_root, img_path)
                self._rgb_to_anno[rgb_path] = anno
        
        print("Random Sampling Processing for RSTPReid...")
        train_list = list(self.train)
        for i in range(len(train_list)):
            item = list(train_list[i])
            rgb_path = item[2]
            
            # Find the original annotation
            anno = self._rgb_to_anno.get(rgb_path)
            if anno:
                captions = anno.get('captions', [])
                if isinstance(captions, list) and len(captions) > 0:
                    caption = random.choice(captions)
                else:
                    caption = captions if isinstance(captions, str) else ""
                item[6] = caption
            
            train_list[i] = tuple(item)
        self.train = train_list
        print("Random Sampling Completed!")

    def show_dataset_info(self):
        """Override to show RSTPReid specific statistics."""
        from prettytable import PrettyTable
        
        num_train_pids = len(self.train_id_container)
        num_train_imgs = len(self.train_annos)
        # Count total captions (2 per image)
        num_train_captions = sum(len(anno.get('captions', [])) for anno in self.train_annos)
        
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


class RSTPReid_ThreeModal(RSTPReid):
    """
    RSTPReid variant that explicitly handles 3 modalities: RGB, TEXT, SK.
    
    This class provides additional utilities for 3-modal experiments
    where NIR and CP are explicitly marked as missing.
    """
    
    def __init__(self, root='', verbose=True):
        super().__init__(root, verbose)
        self.num_modalities = 3  # RGB, TEXT, SK
        self.available_modalities = ['RGB', 'TEXT', 'SK']
        self.missing_modalities = ['NIR', 'CP']
