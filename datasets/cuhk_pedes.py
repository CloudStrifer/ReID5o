"""
CUHK-PEDES Dataset for Text-Image-Sketch Person Re-identification

This dataset contains:
- RGB images from CUHK-PEDES (imgs folder)
- Text captions (caption_all.json)
- Sketch images (from data/sketch/aliyun/CUHK)

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
data/sketch/aliyun/CUHK/
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
    sketch_dir = 'sketch/aliyun/CUHK'
    
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
        """Build a mapping from image filename to sketch path"""
        sketch_paths = {}
        
        if not op.exists(self.sketch_root):
            self.logger.warning(f"Sketch directory not found: {self.sketch_root}")
            return sketch_paths
        
        for subfolder in os.listdir(self.sketch_root):
            subfolder_path = op.join(self.sketch_root, subfolder)
            if not op.isdir(subfolder_path):
                continue
            
            for filename in os.listdir(subfolder_path):
                if filename.endswith(('.jpg', '.png', '.jpeg')):
                    # Create key: subfolder/filename (matching file_path structure in caption_all.json)
                    key = f"{subfolder}/{filename}"
                    sketch_paths[key] = op.join(subfolder_path, filename)
        
        return sketch_paths
    
    def _get_sketch_path(self, file_path):
        """
        Get sketch path for a given image file_path.
        
        Args:
            file_path: Path like 'test_query/p10376_s14337.jpg'
            
        Returns:
            Sketch path or None if not found
        """
        # The sketch might be organized differently
        # Try different matching strategies
        
        # Strategy 1: Direct match (subfolder/filename)
        parts = file_path.replace('\\', '/').split('/')
        if len(parts) >= 2:
            subfolder = parts[-2]  # e.g., 'test_query', 'cam_a'
            filename = parts[-1]   # e.g., 'p10376_s14337.jpg'
            key = f"{subfolder}/{filename}"
            if key in self.sketch_paths:
                return self.sketch_paths[key]
        
        # Strategy 2: Just filename match
        filename = op.basename(file_path)
        for key, path in self.sketch_paths.items():
            if key.endswith(filename):
                return path
        
        return None
    
    def _split_data(self):
        """
        Split data into train/val/test based on file_path prefix.
        
        CUHK-PEDES typically uses:
        - train: images from training set
        - val: images from validation set  
        - test_query/test_gallery: images for testing
        """
        train_data = []
        val_data = []
        test_data = []
        
        for item in self.captions_data:
            file_path = item['file_path'].replace('\\', '/')
            
            # Determine split based on file_path prefix
            if 'train' in file_path.lower():
                train_data.append(item)
            elif 'val' in file_path.lower():
                val_data.append(item)
            elif 'test' in file_path.lower():
                test_data.append(item)
            else:
                # Default to training if unclear
                train_data.append(item)
        
        self.logger.info(f"CUHK-PEDES split: train={len(train_data)}, val={len(val_data)}, test={len(test_data)}")
        
        return train_data, val_data, test_data
    
    def _process_anno(self, annos, training=False):
        """
        Process annotations for training.
        
        Returns dataset tuples: (pid, image_id, rgb_path, nir_path, cp_path, sk_path, caption)
        Note: nir_path and cp_path will be set to None/placeholder since CUHK-PEDES doesn't have these modalities.
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
            
            # RGB image path - the file_path in json is relative to imgs folder
            # e.g., "test_query/p10376_s14337.jpg" -> "data/CUHK-PEDES/imgs/test_query/p10376_s14337.jpg"
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
            
            # Get caption (randomly select one from the list)
            captions = anno['captions']
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
        
        Returns dict with gallery and query information.
        """
        pid_container = set()
        
        # Separate query and gallery based on file_path
        query_items = []
        gallery_items = []
        
        for item in annos:
            file_path = item['file_path'].replace('\\', '/')
            if 'query' in file_path.lower():
                query_items.append(item)
            elif 'gallery' in file_path.lower():
                gallery_items.append(item)
            else:
                # Default to gallery
                gallery_items.append(item)
        
        # If no explicit query/gallery split, create one
        if len(query_items) == 0 and len(gallery_items) > 0:
            # Use all test data, treating text as query and images as gallery
            query_items = annos
            gallery_items = annos
        
        # If no explicit gallery, use all items as gallery (text-to-image retrieval)
        if len(gallery_items) == 0:
            gallery_items = annos
        
        # Build pid mapping
        all_items = list(set([item['id'] for item in query_items + gallery_items]))
        unique_pids = sorted(all_items)
        pid_map = {pid: idx for idx, pid in enumerate(unique_pids)}
        
        # Process gallery (RGB images)
        gallery_paths = []
        gallery_pids = []
        
        for item in gallery_items:
            pid = pid_map[item['id']]
            pid_container.add(pid)
            file_path = item['file_path'].replace('\\', '/')
            img_path = op.join(self.imgs_root, file_path)
            gallery_paths.append(img_path)
            gallery_pids.append(pid)
        
        # Process queries for different modality combinations
        # For CUHK-PEDES with sketch, we have: TEXT, SK, TEXT+SK
        queries = self._build_query_combinations(query_items, pid_map)
        
        dataset = {
            "gallery_pids": gallery_pids,
            "gallery_paths": gallery_paths,
            "queries": queries
        }
        
        return dataset, pid_container
    
    def _build_query_combinations(self, query_items, pid_map):
        """
        Build query combinations for CUHK-PEDES.
        
        Since CUHK-PEDES only has RGB, TEXT, and SK, we create queries for:
        - TEXT (single modality text query)
        - SK (single modality sketch query)
        - TEXT+SK and SK+TEXT (two modality combinations)
        
        Note: NIR and CP queries will be empty or use placeholders.
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
                sk_path = rgb_path  # Fallback
            
            # Get caption
            captions = item['captions']
            if isinstance(captions, list) and len(captions) > 0:
                # Use all captions as separate queries
                for caption in captions:
                    # TEXT query: (pid, caption)
                    queries['TEXT'].append((pid, caption))
                    
                    # SK query: (pid, sk_path) - only add once per unique (pid, sk_path)
                    if sk_path != rgb_path:  # Only add if we have actual sketch
                        sk_key = (pid, sk_path)
                        if sk_key not in added_sk_queries:
                            queries['SK'].append((pid, sk_path))
                            added_sk_queries.add(sk_key)
                    
                    # TEXT+SK: (pid, caption, sk_path)
                    queries['TEXT+SK'].append((pid, caption, sk_path))
                    
                    # SK+TEXT: (pid, sk_path, caption)
                    queries['SK+TEXT'].append((pid, sk_path, caption))
            else:
                caption = captions if isinstance(captions, str) else ""
                queries['TEXT'].append((pid, caption))
                if sk_path != rgb_path:
                    sk_key = (pid, sk_path)
                    if sk_key not in added_sk_queries:
                        queries['SK'].append((pid, sk_path))
                        added_sk_queries.add(sk_key)
                queries['TEXT+SK'].append((pid, caption, sk_path))
                queries['SK+TEXT'].append((pid, sk_path, caption))
        
        return queries
    
    def random_sampling(self):
        """
        Random sampling for training.
        For CUHK-PEDES, we randomly select a caption for each image.
        
        Note: This is called by the training dataset at each epoch.
        """
        # Build a quick lookup from rgb_path to original annotation
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
            
            # Find the original annotation
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
        """Override to show CUHK-PEDES specific statistics."""
        from prettytable import PrettyTable
        
        num_train_pids = len(self.train_id_container)
        num_train_imgs = len(self.train_annos)
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


class CUHK_PEDES_ThreeModal(CUHK_PEDES):
    """
    CUHK-PEDES variant that explicitly handles 3 modalities: RGB, TEXT, SK.
    
    This class provides additional utilities for 3-modal experiments
    where NIR and CP are explicitly marked as missing.
    """
    
    def __init__(self, root='', verbose=True):
        super().__init__(root, verbose)
        self.num_modalities = 3  # RGB, TEXT, SK
        self.available_modalities = ['RGB', 'TEXT', 'SK']
        self.missing_modalities = ['NIR', 'CP']
