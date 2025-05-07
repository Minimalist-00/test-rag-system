# Mac での RAG デモシステムの実行方法

このガイドでは、Mac で Azure OpenAI API と Azure AI Search を使用した RAG デモシステムをセットアップして実行する手順を説明します。

## 前提条件

- macOS
- Python 3.8 以上
- pip（Python パッケージマネージャー）
- Azure アカウントと関連サービス（Azure OpenAI、Azure AI Search）

## セットアップ手順

### 1. リポジトリのクローン

ターミナルを開き、以下のコマンドを実行してリポジトリをクローンします：

```bash
git clone https://github.com/yourusername/test-rag-system.git
cd test-rag-system
```

または、既にリポジトリをダウンロードしている場合は、そのディレクトリに移動してください。

### 2. Python 仮想環境のセットアップ（推奨）

隔離された環境でアプリケーションを実行するために、仮想環境を作成します：

```bash
# 仮想環境の作成
python3 -m venv test_rag_env

# 仮想環境のアクティベート
source test_rag_env/bin/activate
```

### 3. 依存パッケージのインストール

必要なパッケージをインストールします：

```bash
pip install -r requirements.txt
```

### 4. 環境変数の設定

`.env`ファイルを作成して必要な環境変数を設定します：

```bash
touch .env
```

お好みのテキストエディタで`.env`ファイルを開き、以下の内容を追加します：

```
# Azure OpenAI API設定
OPENAI_API_KEY=your_api_key
OPENAI_API_VERSION=your_api_version
OPENAI_API_ENDPOINT=your_endpoint_url
OPENAI_ENGINE=your_deployment_name
OPENAI_EMBEDDING_MODEL=your_embedding_model_name

# Azure AI Search設定
SEARCH_ENDPOINT=your_search_endpoint
SEARCH_API_KEY=your_search_api_key
SEARCH_INDEX_NAME=your_index_name
```

各変数を実際の値に置き換えてください。

### 5. アプリケーションの実行

以下のコマンドを実行してアプリケーションを起動します：

```bash
# APIの接続テスト
python src/test_api_connection.py

# シンプルなデモアプリ
streamlit run src/simple_app.py

# RAGアプリ
streamlit run src/rag_app.py
```

アプリケーションが起動すると、ブラウザで自動的に開かれます（通常は`http://localhost:8501`）。

## トラブルシューティング

### 依存関係のエラー

特定のパッケージのインストールでエラーが発生した場合は、以下のコマンドを試してください：

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 環境変数の問題

環境変数が正しく読み込まれない場合は、アプリケーションを実行する前に以下のコマンドを試してください：

```bash
export $(grep -v '^#' .env | xargs)
```

### Python 実行エラー

Python バージョンの問題が発生した場合は、`python3`コマンドを使用してください：

```bash
python3 src/test_api_connection.py
```

## Windows 環境からの移行に関する注意点

1. パス区切り文字：Windows ではバックスラッシュ（`\`）、Mac ではスラッシュ（`/`）が使用されます
2. 環境変数の設定：Windows の`set`コマンドの代わりに、Mac では`export`コマンドを使用します
3. 実行可能ファイルの権限：必要に応じて、実行可能ファイルに権限を付与します（`chmod +x filename`）

## 仮想環境の終了

作業が終わったら、以下のコマンドで仮想環境を終了できます：

```bash
deactivate
```
