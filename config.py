# config.py

# --- 定数設定 ---
DB_NAME = 'memory_log.db'
HISTORY_DIR = 'history'
MODEL_ID = 'microsoft/Florence-2-large'
COLORS = ['red', 'blue', 'green', 'yellow', 'purple', 'orange']

# --- Webカメラ設定を追加 ---
WEBCAM_DEVICE_ID = 0  
PROCESSING_FPS = 3    
OBJECT_TRACKING_THRESHOLD_PIXELS = 75 

FRAMES_TO_CONSIDER_DISAPPEARED = 5 


# 記録したいオブジェクトのキーワードを列挙します。
# 例として、鍵、カップ、眼鏡、スマートフォンを登録します。
WHITELIST_KEYWORDS = [
    'key', 'cup', 'glasses', 'smartphone', 'wallet'
]

