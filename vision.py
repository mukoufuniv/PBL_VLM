# vision.py (全面改訂版)

import torch
import time
from PIL import ImageDraw, ImageFont
from transformers import AutoProcessor, AutoModelForCausalLM,AutoConfig
import numpy as np
from datetime import datetime
import os

import config
from database import save_detection

def get_bbox_center(bbox):
    """バウンディングボックスの中心座標を計算する"""
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2, (y1 + y2) / 2)

class VLM_Detector:
    def __init__(self):
        """モデルの読み込みと、追跡用オブジェクトリストの初期化"""
        print("VLMモデルの読み込みを開始します...")
        self.dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            self.dtype = torch.bfloat16
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用デバイス: {self.device}, データ型: {self.dtype}")

        self.model = AutoModelForCausalLM.from_pretrained(
            config.MODEL_ID, torch_dtype=self.dtype, trust_remote_code=True
        ).to(self.device)
        self.processor = AutoProcessor.from_pretrained(config.MODEL_ID, trust_remote_code=True)
        
        # ★★★ 追跡中のオブジェクトを管理するリスト ★★★
        self.tracked_objects = []
        
        print("モデルの読み込みが完了しました。")

    def run_detection(self, image):
        """画像から物体を検出し、追跡と消失検知を行う"""
        original_image = image.copy()
        
        task_prompt = '<DENSE_REGION_CAPTION>'
        inputs = self.processor(
            text=task_prompt, images=original_image, return_tensors="pt", do_rescale=False
        ).to(self.device, self.dtype)
        
        # --- 推論実行 ---
        generated_ids = self.model.generate(
            input_ids=inputs["input_ids"], pixel_values=inputs["pixel_values"],
            max_new_tokens=1024, num_beams=3, do_sample=False,
        )
        
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        results = self.processor.post_process_generation(
            generated_text, task=task_prompt, image_size=(original_image.width, original_image.height)
        )
        
        current_detections = []
        if task_prompt in results and isinstance(results[task_prompt], dict):
            detections = results[task_prompt]
            bboxes = detections.get('bboxes', [])
            labels = detections.get('labels', [])
            for box, label in zip(bboxes, labels):   
                # ホワイトリストに含まれるキーワードがラベルに含まれている場合のみを対象とする
                if any(keyword in label for keyword in config.WHITELIST_KEYWORDS):
                    current_detections.append({'bbox': box, 'label': label, 'center': get_bbox_center(box)})

        # ★★★ ここからがトラッキングのメインロジック ★★★
        
        # 1. 追跡中のオブジェクトと現在の検出結果をマッチング
        unmatched_detections = []
        for det in current_detections:
            matched = False
            for tracked_obj in self.tracked_objects:
                dist = np.linalg.norm(np.array(det['center']) - np.array(tracked_obj['center']))
                if dist < config.OBJECT_TRACKING_THRESHOLD_PIXELS:
                    # マッチした場合、情報を更新して追跡を継続
                    tracked_obj.update({
                        'bbox': det['bbox'],
                        'center': det['center'],
                        'label': det['label'], # ラベルも最新に更新
                        'unseen_frames': 0,
                        'last_seen_image': original_image,
                        'last_seen_time': datetime.now()
                    })
                    matched = True
                    tracked_obj['matched_in_frame'] = True
                    break
            if not matched:
                unmatched_detections.append(det)

        # 2. 追跡中のオブジェクトで、今回マッチしなかったものを処理
        disappeared_objects = []
        for tracked_obj in self.tracked_objects:
            if not tracked_obj.get('matched_in_frame', False):
                tracked_obj['unseen_frames'] += 1
            # 一定フレーム数見失ったら「消失」と判断
            if tracked_obj['unseen_frames'] > config.FRAMES_TO_CONSIDER_DISAPPEARED:
                disappeared_objects.append(tracked_obj)
        
        # 3. 消失したオブジェクトをDBに保存し、追跡リストから削除
        for obj in disappeared_objects:
            print(f"  [Disappeared] {obj['label']} を最後に検出。DBに保存します。")
            timestamp_str = obj['last_seen_time'].strftime('%Y-%m-%d %H:%M:%S')
            image_filename = obj['last_seen_time'].strftime('%Y%m%d%H%M%S_%f') + '.jpg'
            image_path = os.path.join(config.HISTORY_DIR, image_filename)
            obj['last_seen_image'].save(image_path, 'JPEG')
            save_detection(timestamp_str, image_path, obj['label'], obj['bbox'])
            self.tracked_objects.remove(obj)

        # 4. 今回新たに検出されたオブジェクトを追跡リストに追加
        for det in unmatched_detections:
            print(f"  [New Object] {det['label']} の追跡を開始します。")
            self.tracked_objects.append({
                'bbox': det['bbox'],
                'center': det['center'],
                'label': det['label'],
                'unseen_frames': 0,
                'last_seen_image': original_image,
                'last_seen_time': datetime.now(),
            })

        # 5. 追跡中のオブジェクトを描画用に準備
        draw = ImageDraw.Draw(image)
        for tracked_obj in self.tracked_objects:
            tracked_obj['matched_in_frame'] = False # 次のフレームのためにリセット
            box = tracked_obj['bbox']
            label = tracked_obj['label']
            color = config.COLORS[self.tracked_objects.index(tracked_obj) % len(config.COLORS)]
            draw.rectangle(box, outline=color, width=3)
            try:
                font = ImageFont.truetype("meiryo.ttc", 16)
            except IOError:
                font = ImageFont.load_default()
            draw.text((box[0], box[1] - 20), label, fill=color, font=font)
        
        return image