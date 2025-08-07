# app.py (修正版)

import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import database
import os
import config # config.pyをインポート

# --- アプリの基本設定 ---
st.set_page_config(page_title="忘れ物捜索アプリ", layout="wide")
st.title("🕵️ 忘れ物捜索アプリ")
st.write("過去にカメラが認識した物体を検索できます。")
# サイドバーに管理機能を追加
st.sidebar.title("管理メニュー")

if st.sidebar.button("データベースをクリアする"):
    # ボタンが押されたら、確認のための状態をセッションに保存
    st.session_state.confirm_delete = True

# 確認状態にある場合のみ、確認メッセージとボタンを表示
if 'confirm_delete' in st.session_state and st.session_state.confirm_delete:
    st.sidebar.warning("本当に全ての検出履歴と画像ファイルを削除しますか？この操作は元に戻せません。")
    
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        if st.button("はい、削除します", type="primary"):
            success, message = database.clear_db()
            if success:
                st.success(message)
            else:
                st.error(message)
            # 処理が終わったら確認状態をリセット
            st.session_state.confirm_delete = False
            # 画面をリフレッシュして結果を反映
            st.rerun() 

    with col2:
        if st.button("キャンセル"):
            # キャンセルされたら確認状態をリセット
            st.session_state.confirm_delete = False
            st.info("クリア処理はキャンセルされました。")
            # 画面をリフレッシュ
            st.rerun() 


st.sidebar.markdown("---") # 区切り線
# --- 検索フォーム ---
search_term = st.text_input("探している物体のキーワードを入力してください（例: key, cup）", "")

if st.button("検索する"):
    if search_term:
        # ★★★ 修正点1: データベースからBBOXの座標も取得する ★★★
        # database.pyのsearch_for_objectを改造して、bbox_coordsも返すようにします。
        # 今回は一時的にapp.py側で直接DBにアクセスして実装します。
        conn = database.sqlite3.connect(config.DB_NAME)
        conn.row_factory = database.sqlite3.Row
        cursor = conn.cursor()
        cursor.execute(
            "SELECT timestamp, label, image_path, bbox_coords FROM detections WHERE label LIKE ? ORDER BY timestamp DESC", 
            ('%' + search_term + '%',)
        )
        results = cursor.fetchall()
        conn.close()
        
        if not results:
            st.warning("その物体は見つかりませんでした。")
        else:
            st.success(f"「{search_term}」の検索結果: {len(results)}件")
            
            # --- ★★★ 修正点2: BBOX表示のチェックボックスを追加 ★★★
            show_bbox = st.checkbox("検出した位置（バウンディングボックス）を表示する", value=True)
            
            # 検索結果を表示
            for row in results:
                st.markdown("---")
                
                if os.path.exists(row['image_path']):
                    image = Image.open(row['image_path'])
                    
                    # チェックボックスがONの場合のみBBOXを描画
                    if show_bbox:
                        draw = ImageDraw.Draw(image)
                        try:
                            # DBからBBOX座標を文字列として取得し、数値のリストに変換
                            bbox_str = row['bbox_coords']
                            bbox = [float(coord) for coord in bbox_str.split(',')]
                            
                            # BBOXを描画
                            draw.rectangle(bbox, outline='lime', width=5)
                            
                            # ラベルも描画
                            font = ImageFont.truetype("meiryo.ttc", 32)
                            draw.text((bbox[0], bbox[1] - 40), row['label'], fill='lime', font=font)
                        except (ValueError, TypeError, IOError):
                            st.error("BBOXの描画に失敗しました。")

                    # --- ★★★ 修正点3: 画像を大きく表示 ★★★
                    # st.imageを中央のメインカラムに配置して大きく表示
                    st.image(image, caption=f"検出日時: {row['timestamp']}", use_column_width='always')
                    
                else:
                    st.error(f"画像ファイルが見つかりません: {row['image_path']}")
    else:
        st.info("検索キーワードを入力してください。")