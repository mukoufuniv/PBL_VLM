# config.py

# --- 定数設定 ---
DB_NAME = 'memory_log.db'
HISTORY_DIR = 'history'
MODEL_ID = 'microsoft/Florence-2-large'
COLORS = ['red', 'blue', 'green', 'yellow', 'purple', 'orange']

# --- Webカメラ設定を追加 ---
WEBCAM_DEVICE_ID = 0  # 0は通常、PC内蔵のデフォルトカメラを指します
PROCESSING_FPS = 3    # 1秒あたりに推論を実行する回数（フレーム数）
# 同一物体と見なす中心点の最大距離（ピクセル単位）
OBJECT_TRACKING_THRESHOLD_PIXELS = 75 

# 何フレーム見失ったら「消失した」と判断するか
FRAMES_TO_CONSIDER_DISAPPEARED = 5 

BLACKLIST_KEYWORDS = [
    'wall', 'floor', 'ceiling', 'window', 'door', 
    'light', 'shadow', 'table', 'desk', 'chair', 'shelf',
    'background', 
    
    # ↓ ここに人物関連のキーワードを追加・修正します
    'person', 'man', 'woman', 'people', 'boy', 'girl', 'face', 'human' 
]