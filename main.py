# main.py

import cv2
import numpy as np
from PIL import Image
import time

import config
import database
from vision import VLM_Detector

def main():
    """プログラムのメイン処理（Webカメラ版）"""
    # 1. 初期化
    database.init_db()
    detector = VLM_Detector()
    
    # Webカメラを開く
    cap = cv2.VideoCapture(config.WEBCAM_DEVICE_ID)
    if not cap.isOpened():
        print(f"エラー: カメラデバイスID {config.WEBCAM_DEVICE_ID} を開けませんでした。")
        return

    print("\nWebカメラのフィードを開始します。ウィンドウを選択して 'q' キーを押すと終了します。")
    
    # FPS制御用の設定
    last_processed_time = 0
    frame_interval = 1.0 / config.PROCESSING_FPS

    # 2. メインループ
    while True:
        # カメラからフレームを読み込む
        ret, frame = cap.read()
        if not ret:
            print("エラー: フレームを読み込めませんでした。")
            break

        display_frame = frame  # 表示用のフレームを準備

        current_time = time.time()
        # 指定した間隔が経過した場合のみ推論を実行
        if (current_time - last_processed_time) >= frame_interval:
            last_processed_time = current_time
            
            print("-" * 20)
            # OpenCV(BGR)からPillow(RGB)へ画像を変換
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            
            # 物体検出を実行
            processed_image_pil = detector.run_detection(pil_image)
            
            # 表示用にPillow(RGB)からOpenCV(BGR)へ画像を変換
            display_frame = cv2.cvtColor(np.array(processed_image_pil), cv2.COLOR_RGB2BGR)

        # 画面に映像を表示
        cv2.imshow('VLM Live Feed', display_frame)

        # 'q'キーが押されたらループを抜ける
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    # 3. 後処理
    print("アプリケーションを終了します。")
    cap.release()
    cv2.destroyAllWindows()

    # 終了時に「グラス」を検索するデモ
    database.search_for_object('glasses')

if __name__ == '__main__':
    main()