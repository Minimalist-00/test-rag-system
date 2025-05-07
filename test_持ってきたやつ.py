import os
import re
import logging
from dotenv import load_dotenv
import streamlit as st
from openai import AzureOpenAI
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient  
from azure.search.documents.models import VectorizedQuery

load_dotenv()
search_endpoint = os.environ['SEARCH_ENDPOINT']
search_key = os.environ['SEARCH_API_KEY']
indexnametemp = os.environ['SEARCH_INDEX_NAME']
top_k_temp=3 #検索結果の上位何件を表示するか
client = AzureOpenAI(
  api_key=os.environ['OPENAI_API_KEY'],  
  api_version=os.environ['OPENAI_API_VERSION'],
  azure_endpoint = os.environ['OPENAI_API_ENDPOINT']
)

openai_engine = os.environ['OPENAI_ENGINE']
openai_embedding_model = os.environ['OPENAI_EMBEDDING_MODEL']

# Function to generate embeddings for title and content fields, also used for query embeddings
def generate_embeddings(text, text_limit=7000):
    # Clean up text (e.g. line breaks, )    
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'[\n\r]+', ' ', text).strip()
    if len(text) > text_limit:
        logging.warning("Token limit reached exceeded maximum length, truncating...")
        text = text[:text_limit]

    response = client.embeddings.create(input=text, model=openai_embedding_model)
    embeddings = response.data[0].embedding
    return embeddings
    
def query_vector_index(query, searchtype, top_k_parameter, search_client):
    vector = generate_embeddings(query)
    # 検索方法がベクトル検索の場合は、search_textをNoneにする
    if searchtype == "ベクトル検索":
        search_text = None
    # 検索方法がvector_only以外の場合は、search_textにqueryを設定する
    else:
        search_text = query
    
    if searchtype == "ベクトル検索" or searchtype == "ハイブリッド検索":
        results = search_client.search(search_text=search_text, 
                                        vector_queries=[
                                            VectorizedQuery(
                                            kind="vector", vector=vector, k_nearest_neighbors=top_k_parameter, fields="text_vector"
                                            )
                                        ],)
    # 検索方法がフルテキスト検索の場合
    else:
        results = search_client.search(search_text=search_text, top=top_k_parameter)

    return results

def main():
    # Set page title and icon
    st.set_page_config(page_title="RAG Sample Application", page_icon="💬", layout="wide")

    # チャット履歴の初期化
    if "messages" not in st.session_state:
        st.session_state['messages'] = []

    # クリアボタンを押した場合、チャットとst.text_input,promptallをクリアする。
    if st.sidebar.button("Clear Chat"):
        st.session_state['messages'] = []
        promptall = ""

    # Set Search parameters in sidebar
    st.sidebar.markdown("### Azure AI Search 関連パラメータ")

    # 検索結果の上位何件を対象とするかを設定する。top_k_parameterの設定。テキストボックスで指定する。
    top_k_parameter = st.sidebar.text_input("検索結果対象ドキュメント数", top_k_temp)

    # インデックスの名前をテキストボックスで指定する。indexnameの設定
    indexname = st.sidebar.text_input("インデックス名", indexnametemp)

    # 検索方法を選択する。フルテキスト検索 or ベクトル検索 or ハイブリッド検索。
    search_type = st.sidebar.radio("検索方法", ("フルテキスト検索", "ベクトル検索", "ハイブリッド検索"))

    # Set ChatGPT parameters in sidebar
    st.sidebar.markdown("### Azure OpenAI 関連パラメータ")
    Temperature_temp = st.sidebar.slider("Temperature", 0.0, 1.0, 0.0, 0.01)

    # Define system role in text area
    SystemRole = st.sidebar.text_area("System Role",
"""###前提条件
あなたは、ナレッジやドキュメントに対する質問をする際に支援する優秀なアシスタントです。
###制約
    ・回答には役割(userやassistantなど)の情報を含めないでください。
    ・Sources(情報源)にリストされている事実のみを使用して回答してください。
    ・十分な情報がない場合は、わからないと回答してください。
    ・Sources(情報源)を使用しない回答は生成しないでください
    ・ユーザーへの質問によって明確化が必要な場合は、質問してください。""")

    # Add system role to session state
    if SystemRole:
        #既にroleがsystemのメッセージがある場合は、追加しない。ない場合は追加する。
        if not any(message["role"] == "system" for message in st.session_state.messages):
            st.session_state.messages.append({"role": "system", "content": SystemRole})

    #検索クライアントを作成する
    credential = AzureKeyCredential(search_key)
    search_client = SearchClient(endpoint=search_endpoint,
                                 index_name=indexname,
                                 credential=credential)

    # ユーザからの入力を取得する
    if user_input := st.chat_input("プロンプトを入力してください"):
        #検索する。search_fieldsはcontentを対象に検索する
        results = query_vector_index(user_input, search_type, top_k_parameter, search_client)
            
        # 変数を初期化する
        prompt_source = ""

        # resultsから各resultの結果を変数prompt_sourceに代入する。filepathとcontentの情報を代入する。
        for result in results:
            Score = result['@search.score']
            filepath = result['title']
            chunk_id = re.search(r'(?<=pages_).*', result['chunk_id']).group(0)
            content = result['chunk']
    
            # 変数prompt_sourceに各変数の値を追加する
            prompt_source += f"#filepath: {filepath}\n\n  #chunk_id: {chunk_id}\n\n #score: {Score}\n\n #content: {content}\n\n"

        # プロンプトを作成する
        promptall="###Soruces(情報源): \n\n" + prompt_source + "###質問： \n\n" + user_input
        message_temp = []
        message_temp = st.session_state.messages + [{"role": "user", "content": promptall}]

         # expanderを作成する
        with st.sidebar.expander("検索結果の表示"):
            # マークダウンを表示する
            st.markdown(prompt_source)

        with st.spinner("ChatGPTが回答を生成しています"):
            output =client.chat.completions.create(
                model=openai_engine,
                messages=message_temp,
                temperature=Temperature_temp,
                max_tokens=1000,
                frequency_penalty=0,
                presence_penalty=0,
            )
       
        # Add ChatGPT response to conversation
        st.session_state.messages.append({"role": "user", "content": user_input})
        st.session_state.messages.append({"role": "assistant", "content": output.choices[0].message.content})

    # チャット履歴の表示
    messages = st.session_state.get('messages', [])
    for message in messages:
        #roleがassistantだったら、assistantのchat_messageを使う
        if message['role'] == 'assistant':
            with st.chat_message('assistant'):
                st.markdown(message['content'])
        #roleがuserだったら、userのchat_messageを使う
        elif message['role'] == 'user':
            with st.chat_message('user'):
                st.markdown(message['content'])
        else: # 何も出力しない  
            pass
    
if __name__ == '__main__':
    main()