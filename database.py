# database.py

import sqlite3
import os
from datetime import datetime
import config

def init_db():
    """データベースを初期化し、テーブルと画像保存用フォルダを作成する"""
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

def save_detection(original_image, label, bbox):
    """検出した情報と画像をデータベースに保存する"""
    now = datetime.now()
    timestamp_str = now.strftime('%Y-%m-%d %H:%M:%S')
    image_filename = now.strftime('%Y%m%d%H%M%S_%f') + '.jpg'
    image_path = os.path.join(config.HISTORY_DIR, image_filename)
    
    original_image.save(image_path, 'JPEG')
    
    conn = sqlite3.connect(config.DB_NAME)
    cursor = conn.cursor()
    bbox_str = ','.join(map(str, [int(coord) for coord in bbox]))
    cursor.execute(
        "INSERT INTO detections (timestamp, label, bbox_coords, image_path) VALUES (?, ?, ?, ?)",
        (timestamp_str, label, bbox_str, image_path)
    )
    conn.commit()
    conn.close()
    print(f"  [DB Save] {label}")

def search_for_object(keyword):
    """キーワードに一致するオブジェクトの履歴を検索して表示する"""
    print(f"\n--- '{keyword}' の検索結果 ---")
    conn = sqlite3.connect(config.DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT timestamp, label, image_path FROM detections WHERE label LIKE ? ORDER BY timestamp DESC", ('%' + keyword + '%',))
    results = cursor.fetchall()
    conn.close()
    
    if not results:
        print("見つかりませんでした。")
        return

    for row in results:
        print(f"  日時: {row[0]}, 内容: {row[1]}, 画像: {row[2]}")
    print("-" * (len(keyword) + 16))