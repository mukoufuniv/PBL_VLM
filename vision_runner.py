# vision_runner.py

import cv2
from PIL import Image
import time

import config
import database
from vision import VLM_Detector

def run_vision_process():
    """Webカメラを起動し、映像の解析とDBへの保存を続ける"""
    database.init_db()
    detector = VLM_Detector()
    
    cap = cv2.VideoCapture(config.WEBCAM_DEVICE_ID)
    if not cap.isOpened():
        print(f"エラー: カメラデバイスID {config.WEBCAM_DEVICE_ID} を開けませんでした。")
        return

    print(">>> 映像解析プロセスを開始しました。このウィンドウは閉じないでください。 <<<")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("エラー: フレームを読み込めませんでした。")
            break

        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        # 検出処理を実行（この中でDBへの保存も行われる）
        detector.run_detection(pil_image)
        
        # 指定したFPSになるように待機
        time.sleep(1.0 / config.PROCESSING_FPS)

    cap.release()

if __name__ == '__main__':
    run_vision_process()