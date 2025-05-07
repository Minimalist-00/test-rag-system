# test_api_connection.py
import os
import sys
import logging
from dotenv import load_dotenv
from openai import AzureOpenAI

# ロギングの設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_connection():
    # .env ファイルから環境変数を読み込む
    load_dotenv()
    
    # 必要な環境変数の確認
    required_vars = [
        'OPENAI_API_KEY', 
        'OPENAI_API_VERSION', 
        'OPENAI_API_ENDPOINT',
        'OPENAI_ENGINE',
        'OPENAI_EMBEDDING_MODEL'
    ]
    
    missing_vars = []
    for var in required_vars:
        if var not in os.environ or not os.environ[var]:
            missing_vars.append(var)
    
    if missing_vars:
        logger.error(f"次の環境変数が設定されていません: {', '.join(missing_vars)}")
        return False
    
    # 環境変数のロギング
    logger.info(f"API キー: {os.environ['OPENAI_API_KEY'][:4]}...{os.environ['OPENAI_API_KEY'][-4:]}")
    logger.info(f"API バージョン: {os.environ['OPENAI_API_VERSION']}")
    logger.info(f"API エンドポイント: {os.environ['OPENAI_API_ENDPOINT']}")
    logger.info(f"Chat エンジン: {os.environ['OPENAI_ENGINE']}")
    logger.info(f"埋め込みモデル: {os.environ['OPENAI_EMBEDDING_MODEL']}")
    
    try:
        # Azure OpenAI クライアントの初期化
        client = AzureOpenAI(
            api_key=os.environ['OPENAI_API_KEY'],  
            api_version=os.environ['OPENAI_API_VERSION'],
            azure_endpoint=os.environ['OPENAI_API_ENDPOINT']
        )
        
        # 埋め込みテスト
        logger.info("埋め込みAPIの接続をテスト中...")
        embedding_response = client.embeddings.create(
            input="こんにちは、世界",
            model=os.environ['OPENAI_EMBEDDING_MODEL']
        )
        embedding_vector = embedding_response.data[0].embedding
        logger.info(f"埋め込みベクトル生成成功: 次元数 {len(embedding_vector)}")
        
        # チャット完了テスト
        logger.info("チャット完了APIの接続をテスト中...")
        chat_response = client.chat.completions.create(
            model=os.environ['OPENAI_ENGINE'],
            messages=[
                {"role": "system", "content": "あなたは役立つアシスタントです。"},
                {"role": "user", "content": "こんにちは。今日の天気は？"}
            ],
            max_tokens=100
        )
        logger.info(f"チャット完了API接続成功: '{chat_response.choices[0].message.content[:50]}...'")
        
        logger.info("すべてのAPIテストが成功しました！")
        return True
        
    except Exception as e:
        logger.error(f"API接続エラー: {str(e)}")
        return False

if __name__ == "__main__":
    result = test_connection()
    sys.exit(0 if result else 1)