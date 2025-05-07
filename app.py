import os
import re
import logging
from dotenv import load_dotenv
import streamlit as st
from openai import AzureOpenAI
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 環境変数の読み込み
load_dotenv()
search_endpoint = os.environ['SEARCH_ENDPOINT']
search_key = os.environ['SEARCH_API_KEY']
indexnametemp = os.environ['SEARCH_INDEX_NAME']
top_k_temp = 3  # 検索結果の上位何件を表示するか

# API設定情報のログ出力（デバッグ用）
logger.info(f"OPENAI_API_ENDPOINT: {os.environ.get('OPENAI_API_ENDPOINT')}")
logger.info(f"EMBEDDING_API_ENDPOINT: {os.environ.get('EMBEDDING_API_ENDPOINT')}")
logger.info(f"EMBEDDING_MODEL: {os.environ.get('EMBEDDING_MODEL')}")

# 会話用クライアント
chat_client = AzureOpenAI(
    api_key=os.environ['OPENAI_API_KEY'],
    api_version=os.environ['OPENAI_API_VERSION'],
    azure_endpoint=os.environ['OPENAI_API_ENDPOINT']
)

# 埋め込み生成用クライアント - 共通クライアントを使用
embedding_client = AzureOpenAI(
    api_key=os.environ.get('EMBEDDING_API_KEY', os.environ['OPENAI_API_KEY']),
    api_version=os.environ['OPENAI_API_VERSION'],
    azure_endpoint=os.environ.get('EMBEDDING_API_ENDPOINT', os.environ['OPENAI_API_ENDPOINT'])
)

openai_engine = os.environ['OPENAI_ENGINE']
openai_embedding_model = os.environ['EMBEDDING_MODEL']

# タイトルや本文、クエリに対して埋め込みを生成する関数
def generate_embeddings(text, text_limit=7000):
    # テキスト整形（改行や空白を削除）
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'[\n\r]+', ' ', text).strip()
    if len(text) > text_limit:
        logging.warning("トークン数が上限を超えたため、テキストを切り捨てます。")
        text = text[:text_limit]

    try:
        logger.info(f"埋め込み生成: モデル={openai_embedding_model}, テキスト長={len(text)}")
        response = embedding_client.embeddings.create(input=text, model=openai_embedding_model)
        embeddings = response.data[0].embedding
        return embeddings
    except Exception as e:
        logger.error(f"埋め込み生成エラー: {str(e)}")
        raise

# ベクトルインデックスに対してクエリを実行する関数
def query_vector_index(query, searchtype, top_k_parameter, search_client):
    try:
        vector = generate_embeddings(query)
        
        # ベクトル検索のときは、search_textをNoneにする
        if searchtype == "ベクトル検索":
            search_text = None
        else:
            search_text = query

        # ベクトル検索またはハイブリッド検索の場合
        if searchtype == "ベクトル検索" or searchtype == "ハイブリッド検索":
            results = search_client.search(
                search_text=search_text,
                vector_queries=[
                    VectorizedQuery(
                        kind="vector",
                        vector=vector,
                        k_nearest_neighbors=top_k_parameter,
                        fields="text_vector"
                    )
                ],
            )
        # フルテキスト検索の場合
        else:
            results = search_client.search(search_text=search_text, top=top_k_parameter)

        return results
    except Exception as e:
        logger.error(f"検索エラー: {str(e)}")
        st.error(f"検索中にエラーが発生しました: {str(e)}")
        return []

def main():
    # ページの設定
    st.set_page_config(page_title="RAG Sample Application", page_icon="💬", layout="wide")

    # チャット履歴の初期化
    if "messages" not in st.session_state:
        st.session_state['messages'] = []

    # クリアボタン押下時にチャット履歴とプロンプトをリセット
    if st.sidebar.button("Clear Chat"):
        st.session_state['messages'] = []
        promptall = ""

    # 検索関連のパラメータ設定
    st.sidebar.markdown("### Azure AI Search 関連パラメータ")
    top_k_parameter = st.sidebar.text_input("検索結果対象ドキュメント数", top_k_temp)
    indexname = st.sidebar.text_input("インデックス名", indexnametemp)
    search_type = st.sidebar.radio("検索方法", ("フルテキスト検索", "ベクトル検索", "ハイブリッド検索"))

    # ChatGPT関連のパラメータ設定
    st.sidebar.markdown("### Azure OpenAI 関連パラメータ")
    Temperature_temp = st.sidebar.slider("Temperature", 0.0, 1.0, 0.0, 0.01)

    # システムロールの定義
    SystemRole = st.sidebar.text_area("System Role",
"""###前提条件
あなたは、ナレッジやドキュメントに対する質問をする際に支援する優秀なアシスタントです。
###制約
・回答には役割(userやassistantなど)の情報を含めないでください。
・Sources(情報源)にリストされている事実のみを使用して回答してください。
・十分な情報がない場合は、わからないと回答してください。
・Sources(情報源)を使用しない回答は生成しないでください。
・ユーザーへの質問によって明確化が必要な場合は、質問してください。""")

    if SystemRole:
        # 既にsystemロールのメッセージがあるか確認し、なければ追加
        if not any(message["role"] == "system" for message in st.session_state.messages):
            st.session_state.messages.append({"role": "system", "content": SystemRole})

    # 検索クライアントの生成
    credential = AzureKeyCredential(search_key)
    search_client = SearchClient(
        endpoint=search_endpoint,
        index_name=indexname,
        credential=credential
    )

    # デバッグ情報の表示（開発モード）
    with st.sidebar.expander("デバッグ情報", expanded=False):
        st.text(f"Embedding Model: {openai_embedding_model}")
        st.text(f"API Endpoint: {os.environ.get('EMBEDDING_API_ENDPOINT', os.environ['OPENAI_API_ENDPOINT'])}")
        st.text(f"API Version: {os.environ['OPENAI_API_VERSION']}")

    # ユーザーの入力を受け取り、検索と応答生成を行う
    if user_input := st.chat_input("プロンプトを入力してください"):
        try:
            results = query_vector_index(user_input, search_type, top_k_parameter, search_client)

            prompt_source = ""
            for result in results:
                Score = result['@search.score']
                filepath = result['title']
                chunk_id = re.search(r'(?<=pages_).*', result['chunk_id']).group(0)
                content = result['chunk']

                prompt_source += f"#filepath: {filepath}\n\n#chunk_id: {chunk_id}\n\n#score: {Score}\n\n#content: {content}\n\n"

            promptall = "###Soruces(情報源): \n\n" + prompt_source + "###質問： \n\n" + user_input
            message_temp = st.session_state.messages + [{"role": "user", "content": promptall}]

            with st.sidebar.expander("検索結果の表示"):
                st.markdown(prompt_source)

            with st.spinner("ChatGPTが回答を生成しています"):
                output = chat_client.chat.completions.create(
                    model=openai_engine,
                    messages=message_temp,
                    temperature=Temperature_temp,
                    max_tokens=1000,
                    frequency_penalty=0,
                    presence_penalty=0,
                )

            # チャット履歴にユーザー入力とAI出力を追加
            st.session_state.messages.append({"role": "user", "content": user_input})
            st.session_state.messages.append({"role": "assistant", "content": output.choices[0].message.content})
        except Exception as e:
            st.error(f"エラーが発生しました: {str(e)}")
            logger.error(f"メイン処理エラー: {str(e)}")

    # チャット履歴の表示
    messages = st.session_state.get('messages', [])
    for message in messages:
        if message['role'] == 'assistant':
            with st.chat_message('assistant'):
                st.markdown(message['content'])
        elif message['role'] == 'user':
            with st.chat_message('user'):
                st.markdown(message['content'])
        else:
            pass

if __name__ == '__main__':
    main()
