# Azure OpenAI API と Azure AI Search を使用した RAG デモ

このリポジトリには、Azure OpenAI API と Azure AI Search を使用したシンプルな RAG（Retrieval Augmented Generation）システムのデモが含まれています。

## 機能

- Azure OpenAI API を使用したテキスト生成と埋め込み生成
- Azure AI Search を使用したベクトル検索とハイブリッド検索
- Streamlit を使用したインタラクティブなチャットインターフェース

## セットアップ

### 必要条件

- Python 3.8 以上
- Azure アカウントと以下のサービス：
  - Azure OpenAI
  - Azure AI Search

### インストール

1. リポジトリをクローンまたはダウンロードします

2. 必要なライブラリをインストールします：

```
pip install -r requirements.txt
```

3. `.env`ファイルを作成し、必要な環境変数を設定します：

```
# Azure OpenAI API設定
OPENAI_API_KEY=your_api_key
OPENAI_API_VERSION=your_api_version (例: 2023-05-15)
OPENAI_API_ENDPOINT=your_endpoint_url (例: https://your-resource.openai.azure.com/)
OPENAI_ENGINE=your_deployment_name (例: gpt-35-turbo)
OPENAI_EMBEDDING_MODEL=your_embedding_model_name (例: text-embedding-ada-002)

# Azure AI Search設定
SEARCH_ENDPOINT=your_search_endpoint (例: https://your-search.search.windows.net)
SEARCH_API_KEY=your_search_api_key
SEARCH_INDEX_NAME=your_index_name
```

## 使い方

### API の接続テスト

API への接続テストを実行するには：

```
python test_api_connection.py
```

### シンプルなデモアプリ

テキスト生成と埋め込み生成のデモアプリを起動するには：

```
streamlit run simple_app.py
```

### RAG アプリ

RAG（検索拡張生成）デモアプリを起動するには：

```
streamlit run rag_app.py
```

## 注意点

- 使用する前に、Azure AI Search にインデックスが作成され、ドキュメントがインポートされていることを確認してください
- インデックスには `text_vector` フィールドが含まれている必要があります
- インデックスの構造に応じて、`rag_app.py` の検索結果処理部分を調整する必要がある場合があります
