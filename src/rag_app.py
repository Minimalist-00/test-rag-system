import os
import logging
import re
import streamlit as st
from dotenv import load_dotenv
from openai import AzureOpenAI
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery

# ロギングの設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 環境変数の読み込み
load_dotenv()

# Streamlitアプリのタイトル設定
st.set_page_config(page_title="シンプルRAGデモ", page_icon="🔍", layout="wide")
st.title("シンプルRAGデモアプリ")

# サイドバーに説明を追加
st.sidebar.header("このアプリについて")
st.sidebar.write("このアプリはAzure OpenAI APIとAzure AI Searchを使用したシンプルなRAGデモです。")
st.sidebar.write("ドキュメントに対する質問と回答ができます。")

# 必要な環境変数の確認
required_vars = [
    'OPENAI_API_KEY', 
    'OPENAI_API_VERSION', 
    'OPENAI_API_ENDPOINT',
    'OPENAI_ENGINE',
    'OPENAI_EMBEDDING_MODEL',
    'SEARCH_ENDPOINT',
    'SEARCH_API_KEY',
    'SEARCH_INDEX_NAME'
]

# 環境変数のチェック
missing_vars = []
for var in required_vars:
    if var not in os.environ or not os.environ[var]:
        missing_vars.append(var)
        logger.error(f"環境変数 {var} が設定されていません")

if missing_vars:
    st.error(f"次の環境変数が設定されていません: {', '.join(missing_vars)}")
else:
    try:
        # OpenAIクライアントの初期化
        openai_client = AzureOpenAI(
            api_key=os.environ['OPENAI_API_KEY'],
            api_version=os.environ['OPENAI_API_VERSION'],
            azure_endpoint=os.environ['OPENAI_API_ENDPOINT']
        )
        
        # Azure AI Search設定
        search_endpoint = os.environ['SEARCH_ENDPOINT']
        search_key = os.environ['SEARCH_API_KEY']
        index_name = os.environ['SEARCH_INDEX_NAME']
        
        # 検索クライアントの初期化
        search_client = SearchClient(
            endpoint=search_endpoint,
            index_name=index_name,
            credential=AzureKeyCredential(search_key)
        )
        
        # アプリの設定
        if "messages" not in st.session_state:
            st.session_state.messages = []
            # システムメッセージを追加
            st.session_state.messages.append({
                "role": "system", 
                "content": """あなたは、ドキュメントに対する質問をする際に支援する優秀なアシスタントです。
                提供された情報源のみを使用して回答してください。十分な情報がない場合は、わからないと答えてください。"""
            })
        
        # 埋め込みを生成する関数
        def generate_embeddings(text, text_limit=8000):
            # テキスト整形（改行や空白を削除）
            text = re.sub(r'\s+', ' ', text).strip()
            text = re.sub(r'[\n\r]+', ' ', text).strip()
            if len(text) > text_limit:
                logger.warning(f"テキストが上限（{text_limit}文字）を超えたため、切り捨てます")
                text = text[:text_limit]
            
            try:
                logger.info(f"埋め込み生成開始: テキスト長={len(text)}")
                response = openai_client.embeddings.create(
                    input=text,
                    model=os.environ['OPENAI_EMBEDDING_MODEL']
                )
                embeddings = response.data[0].embedding
                logger.info(f"埋め込み生成完了: 次元数={len(embeddings)}")
                return embeddings
            except Exception as e:
                logger.error(f"埋め込み生成エラー: {str(e)}")
                raise
        
        # ベクトル検索を実行する関数
        def search_documents(query_text, search_type="ハイブリッド検索", top_k=3):
            try:
                logger.info(f"検索実行: クエリ='{query_text}', 検索タイプ={search_type}, 取得数={top_k}")
                
                # 埋め込みベクトルの生成
                vector = generate_embeddings(query_text)
                
                # 検索タイプに応じた検索設定
                if search_type == "ベクトル検索":
                    search_text = None
                else:
                    search_text = query_text
                
                # 検索実行
                if search_type in ["ベクトル検索", "ハイブリッド検索"]:
                    results = search_client.search(
                        search_text=search_text,
                        vector_queries=[
                            VectorizedQuery(
                                vector=vector,
                                k_nearest_neighbors=top_k,
                                fields="text_vector"
                            )
                        ],
                        top=top_k
                    )
                else:  # フルテキスト検索
                    results = search_client.search(
                        search_text=search_text,
                        top=top_k
                    )
                
                return results
            except Exception as e:
                logger.error(f"検索エラー: {str(e)}")
                st.error(f"検索中にエラーが発生しました: {str(e)}")
                return []
        
        # サイドバーの設定
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 検索設定")
        top_k = st.sidebar.number_input("検索結果数", min_value=1, max_value=10, value=3)
        search_type = st.sidebar.radio("検索方法", ("ハイブリッド検索", "ベクトル検索", "フルテキスト検索"))
        temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.0, value=0.0, step=0.1)
        
        # チャット履歴の表示
        for message in st.session_state.messages:
            if message["role"] != "system":  # システムメッセージは表示しない
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
        
        # ユーザー入力
        if prompt := st.chat_input("質問を入力してください"):
            # ユーザーメッセージをチャット履歴に追加
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # 最新のユーザーメッセージを表示
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # 検索処理
            with st.spinner("検索中..."):
                search_results = search_documents(prompt, search_type, top_k)
                
                # 検索結果をコンテキストとしてまとめる
                context = ""
                for i, result in enumerate(search_results):
                    # 検索結果からフィールドを抽出（インデックスの構造に合わせて調整が必要）
                    try:
                        title = result.get("title", "不明なタイトル")
                        content = result.get("content", result.get("chunk", "内容なし"))
                        score = result["@search.score"]
                        
                        context += f"[ドキュメント {i+1}]\n"
                        context += f"タイトル: {title}\n"
                        context += f"スコア: {score}\n"
                        context += f"内容: {content}\n\n"
                    except Exception as e:
                        logger.error(f"検索結果の処理エラー: {str(e)}")
                        context += f"[ドキュメント {i+1}] エラー: {str(e)}\n\n"
            
            # 検索結果の表示（サイドバーの折りたたみセクション）
            with st.sidebar.expander("検索結果", expanded=False):
                st.markdown(context)
            
            # 回答生成
            with st.spinner("回答を生成中..."):
                # プロンプトの作成
                messages = [
                    {"role": "system", "content": """あなたは、ドキュメントに対する質問をする際に支援する優秀なアシスタントです。
                    以下の情報源のみを使用して回答してください。十分な情報がない場合は、わからないと答えてください。
                    ユーザーの質問に対して、情報源の内容に基づいて簡潔に回答してください。"""},
                    {"role": "user", "content": f"以下の情報源を元に質問に答えてください:\n\n{context}\n\n質問: {prompt}"}
                ]
                
                try:
                    # 回答の生成
                    response = openai_client.chat.completions.create(
                        model=os.environ['OPENAI_ENGINE'],
                        messages=messages,
                        temperature=temperature,
                        max_tokens=800
                    )
                    
                    answer = response.choices[0].message.content
                    
                    # アシスタントの回答をチャット履歴に追加
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                    
                    # 回答を表示
                    with st.chat_message("assistant"):
                        st.markdown(answer)
                except Exception as e:
                    logger.error(f"回答生成エラー: {str(e)}")
                    st.error(f"回答の生成中にエラーが発生しました: {str(e)}")
        
        # フッター情報
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 設定情報")
        with st.sidebar.expander("API情報", expanded=False):
            st.markdown(f"**OpenAI モデル**: {os.environ['OPENAI_ENGINE']}")
            st.markdown(f"**埋め込みモデル**: {os.environ['OPENAI_EMBEDDING_MODEL']}")
            st.markdown(f"**検索インデックス**: {index_name}")
    
    except Exception as e:
        logger.error(f"アプリケーションエラー: {str(e)}")
        st.error(f"エラーが発生しました: {str(e)}") 