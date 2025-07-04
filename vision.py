# vision.py

import torch
import time
from PIL import ImageDraw, ImageFont
from transformers import AutoProcessor, AutoModelForCausalLM
import config
from database import save_detection # データベース保存関数をインポート

class VLM_Detector:
    def __init__(self):
        """モデルの読み込みと設定を初期化時に行う"""
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
        print("モデルの読み込みが完了しました。")

    def run_detection(self, image):
        """画像から物体を検出し、描画とログ保存を行う"""
        original_image = image.copy()
        draw = ImageDraw.Draw(image)
        
        task_prompt = '<DENSE_REGION_CAPTION>'
        inputs = self.processor(
            text=task_prompt, images=original_image, return_tensors="pt", do_rescale=False
        ).to(self.device, self.dtype)
        
        print("推論を開始します...")
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start_time = time.time()

        generated_ids = self.model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            num_beams=3,
            do_sample=False,
        )

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end_time = time.time()
        print(f"純粋な推論時間: {end_time - start_time:.4f} 秒")
        
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        results = self.processor.post_process_generation(
            generated_text, task=task_prompt, image_size=(original_image.width, original_image.height)
        )
        
        if task_prompt in results and isinstance(results[task_prompt], dict):
            detections = results[task_prompt]
            bboxes = detections.get('bboxes', [])
            labels = detections.get('labels', [])

            for i, (box, label) in enumerate(zip(bboxes, labels)):
                save_detection(original_image, label, box)
                
                color = config.COLORS[i % len(config.COLORS)]
                draw.rectangle(box, outline=color, width=3)
                try:
                    font = ImageFont.truetype("meiryo.ttc", 16)
                except IOError:
                    font = ImageFont.load_default()
                text_position = [box[0], box[1] - 20 if box[1] > 20 else box[1] + 5]
                draw.text(tuple(text_position), label, fill=color, font=font)
        else:
            print("オブジェクトは検出されませんでした。")
            
        return image