import cv2
from PIL import Image
import time

import config
import database
from vision import VLM_Detector
import numpy as np

VIDEO_STREAM_URL = 'http://192.168.11.6:8080/video'

def run_vision_process():
    """Webカメラを起動し、映像の解析とDBへの保存を続ける"""
    database.init_db()
    detector = VLM_Detector()
    cap = None
    # ステップ1: URLが設定されていれば、まずURLへの接続を試みる
    if VIDEO_STREAM_URL:
        print(f"URLへの接続を試みます: {VIDEO_STREAM_URL}")
        cap = cv2.VideoCapture(VIDEO_STREAM_URL)
    
    # ステップ2: URLでの接続に失敗したか、URLが未設定の場合、ローカルカメラを試みる
    if not cap or not cap.isOpened():
        if VIDEO_STREAM_URL:
            print("URLへの接続に失敗しました。ローカルカメラを試みます。")
        else:
            print("ローカルカメラへの接続を試みます。")
        
        cap = cv2.VideoCapture(config.WEBCAM_DEVICE_ID)

    # 最終チェック: 両方の方法で接続に失敗した場合
    if not cap or not cap.isOpened():
        print("--------------------------------------------------")
        print("エラー: カメラに接続できませんでした。")
        print("URLとローカルカメラの両方への接続に失敗しました。")
        print("  - スマートフォンとPCの接続、またはUSBカメラの接続を確認してください。")
        print("--------------------------------------------------")
        return

    print(">>> 映像解析プロセスを開始しました。 <<<")
    print(">>> 映像ウィンドウを選択して 'q' キーを押すと終了します。 <<<")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("エラー: フレームを読み込めませんでした。")
            break

        # OpenCV(BGR)からPillow(RGB)へ画像を変換
        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        # 検出処理を実行し、描画済みの画像を受け取る
        # vision.pyのrun_detectionは元々、描画後のimageを返す仕様になっています
        processed_image_pil = detector.run_detection(pil_image)
        
        # 表示用にPillow(RGB)からOpenCV(BGR)へ画像を変換
        display_frame = cv2.cvtColor(np.array(processed_image_pil), cv2.COLOR_RGB2BGR)

        # "Live Vision Feed"という名前のウィンドウに映像を表示
        cv2.imshow('Live Vision Feed', display_frame)

        # 'q'キーが押されたらループを抜ける
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        # 指定したFPSになるように待機 (この行は元からありますが、ループの最後に移動)
        time.sleep(1.0 / config.PROCESSING_FPS)

    # 後処理
    print("アプリケーションを終了します。")
    cap.release()
    cv2.destroyAllWindows()
    
if __name__ == '__main__':
    run_vision_process()