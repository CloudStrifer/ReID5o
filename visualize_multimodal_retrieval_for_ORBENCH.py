"""
Multi-Modal Retrieval Visualization Script for ORBench
This script visualizes and compares retrieval results across specific modality combinations:
- TEXT
- TEXT+NIR
- TEXT+NIR+SK
- TEXT+NIR+SK+CP
"""

import torch
import os
import os.path as op
import torch.nn.functional as F
from datasets import build_dataloader
from utils.checkpoint import Checkpointer
from model import build_model
from utils.iotools import load_train_configs, read_image
from utils.simple_tokenizer import SimpleTokenizer
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import gridspec
from typing import List, Tuple, Optional, Dict
import textwrap
import numpy as np


class MultiModalRetriever:
    """Enhanced retriever for multi-modal comparison"""

    def __init__(self, config_path: str, device: str = "cuda"):
        self.device = device
        self.args = self._load_config(config_path)
        self.model = self._build_model()
        self.transform = self._build_transform()
        self.tokenizer = SimpleTokenizer()

    def _load_config(self, config_path: str):
        args = load_train_configs(config_path)
        args.batch_size = 1024
        args.training = False
        return args

    def _build_model(self):
        saved_config_path = op.join(self.args.output_dir, 'configs.yaml')
        if op.exists(saved_config_path):
            saved_args = load_train_configs(saved_config_path)
            saved_args.training = True
            saved_data_loaders = build_dataloader(saved_args)
            train_num_classes = saved_data_loaders[-1]
            print(f"Loading model with {train_num_classes} classes from training config")
            model = build_model(self.args, num_classes=train_num_classes)
        else:
            print(f"Config file not found, using default 1000 classes")
            model = build_model(self.args, num_classes=1000)

        checkpointer = Checkpointer(model)
        checkpointer.load(f=op.join(self.args.output_dir, 'best.pth'))
        return model.to(self.device).eval()

    def _build_transform(self):
        mean = [0.48145466, 0.4578275, 0.40821073]
        std = [0.26862954, 0.26130258, 0.27577711]
        return T.Compose([
            T.Resize((384, 128)),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])

    def tokenize(self, caption: str, text_length: int = 77) -> torch.LongTensor:
        if not caption:
            return torch.zeros(text_length, dtype=torch.long).to(self.device)

        sot_token = self.tokenizer.encoder["<|startoftext|>"]
        eot_token = self.tokenizer.encoder["<|endoftext|>"]
        tokens = [sot_token] + self.tokenizer.encode(caption) + [eot_token]

        result = torch.zeros(text_length, dtype=torch.long)
        if len(tokens) > text_length:
            tokens = tokens[:text_length]
            tokens[-1] = eot_token

        result[:len(tokens)] = torch.tensor(tokens)
        return result

    def extract_gallery_features(self, gallery_loader) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
        gids, gfeats, gpaths = [], [], []
        print(f"Extracting gallery features...")
        for batch in gallery_loader:
            pid, img = batch
            img_paths = gallery_loader.dataset.img_paths[len(gpaths):len(gpaths) + img.size(0)]
            with torch.no_grad():
                img_feat = self.model.encode_rgb_cls(img.to(self.device))
            gids.append(pid.view(-1))
            gfeats.append(img_feat)
            gpaths.extend(img_paths)

        gids = torch.cat(gids, 0)
        gfeats = F.normalize(torch.cat(gfeats, 0), p=2, dim=1)
        print(f"Gallery features extracted: {gfeats.shape[0]} images")
        return gids, gfeats, gpaths

    def extract_query_features(self,
                               nir_path: Optional[str] = None,
                               cp_path: Optional[str] = None,
                               sk_path: Optional[str] = None,
                               text: Optional[str] = None) -> torch.Tensor:
        use_missing_aware = getattr(self.model, 'use_missing_aware', False)
        use_completion = getattr(self.model, 'use_cross_modal_completion', False) and \
                         getattr(self.model, 'use_completion_inference', False)
        model_dtype = getattr(self.model, 'dtype', torch.float16)

        modalities = []
        if nir_path and op.exists(nir_path): modalities.append(('nir', nir_path))
        if cp_path and op.exists(cp_path): modalities.append(('cp', cp_path))
        if sk_path and op.exists(sk_path): modalities.append(('sk', sk_path))
        if text and text.strip(): modalities.append(('text', text))

        if not modalities:
            raise ValueError("At least one modality input is required!")

        with torch.no_grad():
            # 单模态逻辑：触发跨模态特征补全 (Cross-modal Completion)
            if len(modalities) == 1:
                modality_name, data = modalities[0]
                if use_missing_aware:
                    if modality_name == 'text':
                        txt_tensor = self.tokenize(data).to(self.device).unsqueeze(0)
                        img_feat, _ = self.model.encode_text_embeds_with_missing_aware(txt_tensor, is_present=True)
                        img_feat = img_feat.float()
                    else:
                        img_tensor = self.transform(read_image(data)).to(self.device).unsqueeze(0)
                        encoder_method = getattr(self.model, f'encode_{modality_name}_embeds_with_missing_aware')
                        embed_out = encoder_method(img_tensor, is_present=True)
                        img_feat = embed_out[:, 0, :].float()

                    if use_completion and modality_name != 'rgb':
                        mod_name_map = {'nir': 'NIR', 'cp': 'CP', 'sk': 'SK', 'text': 'TEXT'}
                        modality_mask = [False, False, False, False, False]
                        modality_idx_map = {'nir': 1, 'cp': 2, 'sk': 3, 'text': 4}
                        modality_mask[modality_idx_map[modality_name]] = True

                        available_features = {mod_name_map[modality_name]: img_feat}
                        try:
                            completed = self.model.complete_missing_features(available_features, modality_mask)
                            if 'RGB' in completed:
                                generated_rgb = completed['RGB'].float()
                                img_feat = 0.7 * img_feat + 0.3 * generated_rgb
                        except Exception as e:
                            print(f"Completion failed: {e}")
                else:
                    encoder_method = getattr(self.model, f'encode_{modality_name}_cls')
                    img_feat = encoder_method(img_tensor).float()

                fusion_feats = img_feat

            # 多模态逻辑：使用占位符保持 Transformer 结构对齐
            else:
                embeds = []
                dummy_img = torch.zeros(1, 3, 384, 128).to(self.device)
                dummy_txt = torch.zeros(1, 77, dtype=torch.long).to(self.device)

                if use_missing_aware:
                    # 1. NIR 占位与提取
                    if nir_path and op.exists(nir_path):
                        img = self.transform(read_image(nir_path)).to(self.device).unsqueeze(0)
                        embed_out = self.model.encode_nir_embeds_with_missing_aware(img, is_present=True)
                    else:
                        embed_out = self.model.encode_nir_embeds_with_missing_aware(dummy_img, is_present=False)
                    embeds.append(embed_out.to(dtype=model_dtype))

                    # 2. CP 占位与提取
                    if cp_path and op.exists(cp_path):
                        img = self.transform(read_image(cp_path)).to(self.device).unsqueeze(0)
                        embed_out = self.model.encode_cp_embeds_with_missing_aware(img, is_present=True)
                    else:
                        embed_out = self.model.encode_cp_embeds_with_missing_aware(dummy_img, is_present=False)
                    embeds.append(embed_out.to(dtype=model_dtype))

                    # 3. SK 占位与提取
                    if sk_path and op.exists(sk_path):
                        img = self.transform(read_image(sk_path)).to(self.device).unsqueeze(0)
                        embed_out = self.model.encode_sk_embeds_with_missing_aware(img, is_present=True)
                    else:
                        embed_out = self.model.encode_sk_embeds_with_missing_aware(dummy_img, is_present=False)
                    embeds.append(embed_out.to(dtype=model_dtype))

                    # 4. TEXT 占位与提取
                    if text and text.strip():
                        txt = self.tokenize(text).to(self.device).unsqueeze(0)
                        _, embed_out = self.model.encode_text_embeds_with_missing_aware(txt, is_present=True)
                    else:
                        _, embed_out = self.model.encode_text_embeds_with_missing_aware(dummy_txt, is_present=False)
                    embeds.append(embed_out.to(dtype=model_dtype))

                combined = torch.cat(embeds, dim=1).to(dtype=model_dtype)
                fusion_feats = self.model.mm_fusion(combined, combined, combined)
                fusion_feats = fusion_feats.float()

        return F.normalize(fusion_feats, p=2, dim=1)

    def retrieve(self, query_features: torch.Tensor, gallery_features: torch.Tensor,
                 topk: int = 10) -> Tuple[torch.Tensor, torch.Tensor]:
        similarity = query_features @ gallery_features.t()
        scores, indices = torch.topk(similarity, k=topk, dim=1, largest=True, sorted=True)
        return indices.cpu(), scores.cpu()


def plot_multimodal_comparison(
        query_data: Dict,
        modality_results: List[Dict],
        gallery_paths: List[str],
        gallery_ids: torch.Tensor,
        topk: int = 10,
        save_path: str = "multimodal_comparison.png"
):
    num_settings = len(modality_results)

    # 缩小 figsize 的宽度乘数 (1.8)，让画框变窄，消除内部留白
    fig = plt.figure(figsize=(1.8 * (topk + 2), 3.5 * num_settings))
    gs = gridspec.GridSpec(num_settings, topk + 2, figure=fig, wspace=0.05, hspace=0.4)

    query_id = query_data.get('query_id', -1)

    for row_idx, result in enumerate(modality_results):
        modality_name = result['name']
        indices = result['indices'][0]
        scores = result['scores'][0]

        # 1. 绘制左侧的 Query 标题
        ax_title = fig.add_subplot(gs[row_idx, 0])
        ax_title.axis('off')
        ax_title.text(0.5, 0.5, f"{modality_name}\nQuery",
                      ha='center', va='center', fontsize=12, weight='bold',
                      bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))

        # 2. 绘制 Query 内容（文字或堆叠的多模态图像）
        ax_query = fig.add_subplot(gs[row_idx, 1])
        ax_query.axis('off')

        query_elements = []
        if 'TEXT' in modality_name.upper() and query_data.get('text'):
            query_elements.append(('text', query_data['text']))
        # 按照顺序加入提供的视觉模态
        if 'NIR' in modality_name.upper() and query_data.get('nir_path') and op.exists(query_data['nir_path']):
            query_elements.append(('nir', query_data['nir_path']))
        if 'SK' in modality_name.upper() and query_data.get('sk_path') and op.exists(query_data['sk_path']):
            query_elements.append(('sk', query_data['sk_path']))
        if 'CP' in modality_name.upper() and query_data.get('cp_path') and op.exists(query_data['cp_path']):
            query_elements.append(('cp', query_data['cp_path']))

        # 渲染查询模态
        if len(query_elements) == 1 and query_elements[0][0] == 'text':
            wrapped_text = textwrap.fill(query_elements[0][1][:100] + "...", width=20)
            ax_query.text(0.5, 0.5, wrapped_text, ha='center', va='center',
                          fontsize=9, wrap=True)
        else:
            n_images = len([e for e in query_elements if e[0] != 'text'])
            if n_images > 0:
                # 均分左侧框的高度，堆叠渲染图片
                img_height = 1.0 / max(n_images, 1)
                img_idx = 0
                for elem_type, elem_data in query_elements:
                    if elem_type != 'text':
                        try:
                            img = Image.open(elem_data).resize((60, 120))
                            y_pos = img_idx * img_height
                            ax_query.imshow(np.array(img),
                                            extent=[0, 1, 1 - y_pos - img_height, 1 - y_pos])
                            ax_query.text(0.5, 1 - y_pos - img_height / 2, elem_type.upper(),
                                          ha='center', va='center', fontsize=8,
                                          bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                            img_idx += 1
                        except Exception as e:
                            print(f"Error loading {elem_type}: {e}")

        ax_query.set_xlim(0, 1)
        ax_query.set_ylim(0, 1)

        # 3. 绘制检索结果图片 (红绿边框，无文字)
        matched_count = 0
        for col_idx in range(topk):
            ax_result = fig.add_subplot(gs[row_idx, col_idx + 2])

            # 隐藏坐标轴刻度（不隐藏边框）
            ax_result.set_xticks([])
            ax_result.set_yticks([])

            img_idx = indices[col_idx].item()
            img_path = gallery_paths[img_idx]
            img_id = gallery_ids[img_idx].item()
            print(img_id)
            try:
                img = Image.open(img_path).resize((100, 200))
                ax_result.imshow(img)

                # 判断检索是否正确
                is_match = (img_id == query_id)
                if is_match:
                    matched_count += 1
                    border_color = '#00FF00'  # 鲜艳的绿色
                else:
                    border_color = '#FF0000'  # 鲜艳的红色

                # 将边框设置为对应的颜色并加粗
                for spine in ax_result.spines.values():
                    spine.set_edgecolor(border_color)
                    spine.set_linewidth(4)  # 控制边框粗细

            except Exception as e:
                ax_result.text(0.5, 0.5, f"Error\n{str(e)[:20]}",
                               ha='center', va='center', fontsize=8)

        match_rate = matched_count / topk * 100
        print(f"{modality_name}: Matched {matched_count}/{topk} ({match_rate:.1f}%)")

    plt.suptitle(f"ORBench Retrieval Comparison", fontsize=16, weight='bold', y=0.995)

    # 强制控制布局的水平间距
    plt.tight_layout(w_pad=0.2, h_pad=1.0)
    plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
    print(f"\n✓ Visualization saved to: {save_path}")
    plt.close()


def main():
    # 修改为你在 ORBench 上训练好的模型 configs.yaml 路径
    config_path = 'logs/configs.yaml'
    topk = 10

    print("=" * 80)
    print("ORBench Multi-Modal Retrieval Visualization")
    print("=" * 80)

    print("\n[1/5] Initializing model...")
    retriever = MultiModalRetriever(config_path)

    print("\n[2/5] Building data loaders...")
    data_loaders = build_dataloader(retriever.args)
    test_gallery_loader = data_loaders[0]

    print("\n[3/5] Extracting gallery features...")
    gallery_ids, gallery_features, gallery_paths = retriever.extract_gallery_features(test_gallery_loader)

    # 设置你的 ORBench 查询图片路径
    query_data = {
        'query_id': 845,  # 这个要和292行代码中，打印出来的img_id一致，才对。（也就是说，真实ID是603，但是查询的ID是会乱序，你要根据这个img_id一致才行
        'text': "A middle-aged man, of average build and height, sports a short, cropped hairstyle in jet black. At this moment, he is clad in a long-sleeved, hooded jacket in black, complemented by black trousers that reach down to his ankles. On his feet, he wears a pair of black casual shoes that harmonize with the overall tone of his attire.",
        'nir_path': "./data/ORBench/nir/0605/0605_sysu_0117_cam3_0001_nir.jpg",
        'sk_path': "./data/ORBench/sk/0605/0605_sysu_0117_back_0_sketch.jpg",
        'cp_path': "./data/ORBench/cp/0605/0605_sysu_0117_back_0_colorpencil.jpg"
    }

    # 严格按照需求，仅保留 TEXT，TEXT+NIR，TEXT+NIR+SK，TEXT+NIR+SK+CP 四个递进组合
    modality_combinations = [
        {'name': 'TEXT',
         'text': query_data.get('text'),
         'nir_path': None,
         'sk_path': None,
         'cp_path': None},

        {'name': 'TEXT+NIR',
         'text': query_data.get('text'),
         'nir_path': query_data.get('nir_path'),
         'sk_path': None,
         'cp_path': None},

        {'name': 'TEXT+NIR+SK',
         'text': query_data.get('text'),
         'nir_path': query_data.get('nir_path'),
         'sk_path': query_data.get('sk_path'),
         'cp_path': None},

        {'name': 'TEXT+NIR+SK+CP',
         'text': query_data.get('text'),
         'nir_path': query_data.get('nir_path'),
         'sk_path': query_data.get('sk_path'),
         'cp_path': query_data.get('cp_path')},
    ]

    print(f"\n[4/5] Performing retrieval for {len(modality_combinations)} modality combinations...")
    modality_results = []

    for idx, combo in enumerate(modality_combinations):
        print(f"  [{idx + 1}/{len(modality_combinations)}] {combo['name']}...", end=' ')

        try:
            query_features = retriever.extract_query_features(
                nir_path=combo.get('nir_path'),
                cp_path=combo.get('cp_path'),
                sk_path=combo.get('sk_path'),
                text=combo.get('text')
            )

            indices, scores = retriever.retrieve(query_features, gallery_features, topk=topk)

            modality_results.append({
                'name': combo['name'],
                'indices': indices,
                'scores': scores
            })
            print("✓")
        except Exception as e:
            print(f"✗ Error: {e}")

    print("\n[5/5] Creating visualization...")
    plot_multimodal_comparison(
        query_data=query_data,
        modality_results=modality_results,
        gallery_paths=gallery_paths,
        gallery_ids=gallery_ids,
        topk=topk,
        save_path='orbench_multimodal_retrieval_comparison.png'
    )

    print("\n" + "=" * 80)
    print("✓ ORBench retrieval visualization completed!")
    print("=" * 80)


if __name__ == "__main__":
    main()