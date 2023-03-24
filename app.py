# ①基本ライブラリ
import streamlit as st
import re
# モデル用ライブラリ
import transformers
import torch


# ②メインパネル
# ⭐️タイトル
st.title("Text Topic Classifier for TOEIC's passages")
st.caption('AIモデルによるTOEIC長文問題のトピック分類')

# ③以下を⭐️サイドバーに表示
st.sidebar.markdown("## [トピック分類したい]")
st.sidebar.markdown("#### ファイルをアップロード")
# txtファイルアップロード→文章の表示
uploaded_file = st.sidebar.file_uploader("Choose a file", type="txt")

if uploaded_file is not None:
    file_contents = uploaded_file.read()

    # 改行文字を削除する
    file_contents_modif_1 = file_contents.decode('utf-8').replace('\n', '')
    file_contents_modif_2 = ''.join(file_contents_modif_1)
    
    st.markdown("### <アップロードした文書>を表示")
    st.write(file_contents_modif_1)
    


# トピックのリスト表示
st.sidebar.markdown("## [テーマ・トピックス]")
st.sidebar.markdown("#### [0]文化・芸術")
st.sidebar.markdown("#### [1]レクリエーション・学術")
st.sidebar.markdown("#### [2]旅行・アクティビティ")
st.sidebar.markdown("#### [3]人事関係")
st.sidebar.markdown("#### [4]ホテル・レストラン・観光施設")
st.sidebar.markdown("#### [5]公共施設・オフィス・公共機関")
st.sidebar.markdown("#### [6]社内連絡・会話")
st.sidebar.markdown("#### [7]セールス・マネジメント")
st.sidebar.markdown("#### [8]顧客対応")
st.sidebar.markdown("#### [9]告知・手続き関係")
st.sidebar.markdown("#### [10]広告・プロモーション。出版物")
st.sidebar.markdown("#### [11]パートナーシップ(取引先)")




# ④⭐️予測モデル
# 学習モデルのアップロード
# model_path = 'src/t_model_path_15'
model_path = './src/t_model_path_15'

from transformers import AlbertTokenizer, AlbertForSequenceClassification

model = AlbertForSequenceClassification.from_pretrained(model_path, num_labels=12)
tokenizer = AlbertTokenizer.from_pretrained(model_path)

# 推論モデル
from transformers import pipeline

classifier = pipeline('text-classification', model=model.to('cpu'), tokenizer=tokenizer)


# ⑤⭐️予測値の結果出力
# 推論の実行 
result = classifier(file_contents_modif_2)

st.write('#### [Process for prediction]') 
st.write('### Result of classification:')
st.write(str(result),'です!')




