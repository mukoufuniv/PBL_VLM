# database.py (修正版)

import sqlite3
import os
from datetime import datetime
import config
import shutil

# init_dbは変更なし
def init_db():
    if not os.path.exists(config.HISTORY_DIR):
        os.makedirs(config.HISTORY_DIR)
        
    conn = sqlite3.connect(config.DB_NAME)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS detections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            label TEXT NOT NULL,
            bbox_coords TEXT,
            image_path TEXT
        )
    ''')
    conn.commit()
    conn.close()
    print(f"データベース '{config.DB_NAME}' を準備しました。")


# ★★★ save_detection関数を修正 ★★★
# この関数はDBへの記録に専念させ、引数で全ての情報を受け取るようにします。
def save_detection(timestamp_str, image_path, label, bbox):
    """検出情報をデータベースに記録する"""
    conn = sqlite3.connect(config.DB_NAME)
    cursor = conn.cursor()
    
    bbox_str = ','.join(map(str, [int(coord) for coord in bbox]))
    
    cursor.execute(
        "INSERT INTO detections (timestamp, label, bbox_coords, image_path) VALUES (?, ?, ?, ?)",
        (timestamp_str, label, bbox_str, image_path)
    )
    
    conn.commit()
    conn.close()
    # vision.py側でメッセージを出すので、ここでのprintは不要
    # print(f"  [DB Save] {label}")

# search_for_objectは変更なし
def search_for_object(keyword):
    """キーワードに一致するオブジェクトの履歴を検索して返す"""
    conn = sqlite3.connect(config.DB_NAME)
    # 結果を辞書型で受け取れるようにする
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    cursor.execute(
        "SELECT timestamp, label, image_path FROM detections WHERE label LIKE ? ORDER BY timestamp DESC", 
        ('%' + keyword + '%',)
    )
    
    results = cursor.fetchall()
    conn.close()
    return results

def clear_db():
    """データベースの全レコードと、historyフォルダ内の全画像ファイルを削除する"""
    try:
        # 1. データベースの全レコードを削除
        conn = sqlite3.connect(config.DB_NAME)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM detections")
        conn.commit()
        conn.close()
        print("データベースの全レコードを削除しました。")

        # 2. historyフォルダの中身を全て削除
        history_path = config.HISTORY_DIR
        if os.path.exists(history_path):
            # 一度フォルダを削除して、再度作成するのが簡単で確実
            shutil.rmtree(history_path)
            os.makedirs(history_path)
            print("historyフォルダ内の全画像ファイルを削除しました。")
        
        return True, "データベースと履歴ファイルを正常にクリアしました。"
        
    except Exception as e:
        print(f"データベースのクリア中にエラーが発生しました: {e}")
        return False, f"エラーが発生しました: {e}"