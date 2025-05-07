import streamlit as st
import os
from dotenv import load_dotenv
from openai import AzureOpenAI

# 環境変数の読み込み
load_dotenv()

# Streamlitアプリのタイトル設定
st.title("Azure OpenAI APIデモアプリ")

# サイドバーに説明を追加
st.sidebar.header("このアプリについて")
st.sidebar.write("このアプリはAzure OpenAI APIを使用した簡単なデモです。")
st.sidebar.write("テキスト生成と埋め込みベクトルの生成を試すことができます。")

# タブの作成
tab1, tab2 = st.tabs(["テキスト生成", "埋め込みベクトル"])

# クライアントの初期化
@st.cache_resource
def get_client():
    return AzureOpenAI(
        api_key=os.environ['OPENAI_API_KEY'],  
        api_version=os.environ['OPENAI_API_VERSION'],
        azure_endpoint=os.environ['OPENAI_API_ENDPOINT']
    )

try:
    client = get_client()

    # テキスト生成タブ
    with tab1:
        st.header("テキスト生成")
        system_prompt = st.text_area(
            "システムプロンプト", 
            value="あなたは役立つアシスタントです。", 
            height=100
        )
        user_prompt = st.text_area(
            "ユーザープロンプト", 
            value="こんにちは。簡単な自己紹介をしてください。", 
            height=150
        )
        
        if st.button("テキスト生成"):
            with st.spinner("生成中..."):
                response = client.chat.completions.create(
                    model=os.environ['OPENAI_ENGINE'],
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    max_tokens=500
                )
                st.write("### 生成結果")
                st.write(response.choices[0].message.content)
    
    # 埋め込みベクトルタブ
    with tab2:
        st.header("埋め込みベクトル生成")
        embedding_text = st.text_area(
            "テキスト入力", 
            value="こんにちは、世界", 
            height=150
        )
        
        if st.button("埋め込みベクトル生成"):
            with st.spinner("生成中..."):
                response = client.embeddings.create(
                    input=embedding_text,
                    model=os.environ['OPENAI_EMBEDDING_MODEL']
                )
                embedding_vector = response.data[0].embedding
                
                st.write(f"### 埋め込みベクトル (次元数: {len(embedding_vector)})")
                st.write("最初の10要素:")
                st.write(embedding_vector[:10])
                
                # ベクトルの可視化（簡易的）
                st.write("### ベクトル可視化（最初の20要素）")
                st.bar_chart(embedding_vector[:20])

except Exception as e:
    st.error(f"エラーが発生しました: {str(e)}")
    st.error("環境変数が正しく設定されているか確認してください。") 