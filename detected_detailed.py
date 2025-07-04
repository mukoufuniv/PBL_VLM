import torch
import time
import requests
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoProcessor, AutoModelForCausalLM 
import numpy as np
import cv2

# --- 設定 ---
dtype = torch.float16 if torch.cuda.is_available() else torch.float32
if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
    dtype = torch.bfloat16
print(f"使用するデータ型: {dtype}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用デバイス: {device}")

model_id = 'microsoft/Florence-2-large' 

model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    torch_dtype=dtype,
    trust_remote_code=True
).to(device)
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

COLORS = ['red', 'blue', 'green', 'yellow', 'purple', 'orange']

def run_detailed_detection(image):
    # 詳細情報取得用のプロンプト
    task_prompt = '<DENSE_REGION_CAPTION>'
    
    inputs = processor(text=task_prompt, images=image, return_tensors="pt", do_rescale=False).to(device, dtype)
    
    # 時間計測
    print("推論を開始します...")
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start_time = time.time()

    generated_ids = model.generate(
      input_ids=inputs["input_ids"],
      pixel_values=inputs["pixel_values"],
      max_new_tokens=1024,
      num_beams=3,
      do_sample=False
    )

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    end_time = time.time()
    
    print(f"純粋な推論時間: {end_time - start_time:.4f} 秒")
    
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    results = processor.post_process_generation(generated_text, task=task_prompt, image_size=(image.width, image.height))
    draw = ImageDraw.Draw(image)
    
    if task_prompt in results and isinstance(results[task_prompt], dict):
        detections = results[task_prompt]
        bboxes = detections.get('bboxes', [])
        labels = detections.get('labels', [])

        for i, (box, label) in enumerate(zip(bboxes, labels)):
            color = COLORS[i % len(COLORS)]
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


# --- メインの処理 ---
if __name__ == '__main__':
    try:
        image = Image.open('test_image.png').convert("RGB")
    except FileNotFoundError:
        print("エラー: 'test_image.png' が見つかりません。画像を同じフォルダに置いてください。")
        exit()
    
    print("\nモデルのロードと設定が完了しました。")
    print("詳細情報取得タスクの推論速度を計測します...")
    print("="*40)

    # --- 修正点: ウォームアップと本実行のロジックを追加 ---
    print("--- 1回目 (ウォームアップ実行) ---")
    _ = run_detailed_detection(image.copy())

    print("\n" + "="*40 + "\n")

    print("--- 2回目 (速度計測の本実行) ---")
    result_image = run_detailed_detection(image.copy())
    
    result_image.show()
    
    result_image.save('result_detailed.jpg')
    print("詳細な検出結果を 'result_detailed.jpg' に保存しました。")