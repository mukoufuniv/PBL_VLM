# config.py

# --- 定数設定 ---
DB_NAME = 'memory_log.db'
HISTORY_DIR = 'history'
MODEL_ID = 'microsoft/Florence-2-large'
COLORS = ['red', 'blue', 'green', 'yellow', 'purple', 'orange']

# --- Webカメラ設定を追加 ---
WEBCAM_DEVICE_ID = 0  # 0は通常、PC内蔵のデフォルトカメラを指します
PROCESSING_FPS = 3    # 1秒あたりに推論を実行する回数（フレーム数）