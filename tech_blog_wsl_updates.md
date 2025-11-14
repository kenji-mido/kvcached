# kvcached の WSL2 対応: GPU 共有を Windows 環境でも実現

## はじめに

LLM（Large Language Model）の推論サービングにおいて、GPU メモリの効率的な利用は重要な課題です。本記事では、GPU 共有を実現する革新的なライブラリ「kvcached」と、その WSL2 環境への対応について解説します。

## kvcached とは

kvcached（KV cache daemon）は、共有 GPU 上で LLM のサービングやトレーニングを行うための KV キャッシュライブラリです。OS スタイルの**仮想メモリ**抽象化を LLM システムに導入することで、**エラスティックでオンデマンド**な KV キャッシュ割り当てを実現し、動的なワークロードにおける GPU 利用率を向上させます。

### 核心技術

kvcached の革新性は、GPU の仮想アドレッシングと物理メモリ割り当てを KV キャッシュレイヤーで分離する点にあります。これにより以下が可能になります：

- **初期段階**: 仮想メモリのみを予約
- **実使用時**: キャッシュが実際に使用されるタイミングで物理 GPU メモリを割り当て
- **結果**: オンデマンド割り当てと柔軟な共有により、動的・混合ワークロードにおける GPU メモリ利用率が向上

### 主要機能

1. **エラスティック KV キャッシュ**: 実負荷に応じて KV メモリを動的に割り当て・回収
2. **GPU 仮想メモリ**: ランタイムマッピングを介して論理 KV と物理 GPU メモリを分離
3. **メモリ制御 CLI**: kvcached CLI でメモリ制限を強制
4. **フロントエンドルーターとスリープモード**: リクエストをターゲットモデルにルーティングし、アイドル時にモデルをスリープ
5. **主要サービングエンジンのサポート**: SGLang と vLLM に統合

### ユースケース

kvcached は以下のような実用的なシナリオで威力を発揮します：

- **マルチ LLM サービング**: 複数の LLM が GPU メモリを弾力的に共有し、静的なメモリ分割なしで同時デプロイ可能
- **サーバーレス LLM**: 必要時のみ KV キャッシュを割り当て、オンデマンドでモデルの起動・停止をサポート
- **複合 AI システム**: 限られたハードウェア上で、パイプライン内の専門モデル（検索、推論、要約など）間でメモリを弾力的に割り当て
- **GPU ワークロード共存**: トレーニングジョブ、ファインチューニング、ビジョンモデルなど、他の GPU ワークロードと LLM 推論を共存可能

### 性能上の利点

ベンチマーク結果によると、A100-80G GPU 上で 3 つの `Llama-3.1-8B` モデルを断続的なピークを持つワークロードでサービングする場合、kvcached は現行のサービングエンジンと比較して **2〜28 倍の TTFT（Time To First Token）削減**を実現します。この性能向上は、LLM サービングにおける**大幅なコスト削減**に直結します。

## WSL2 対応アップデート

今回の `wsl-updates` ブランチでは、kvcached を Windows の WSL2（Windows Subsystem for Linux 2）環境で動作させるための包括的な対応が行われました。

### アップデート内容

#### 1. 完全なセットアップガイドの追加（SETUP_GUIDE.md）

409 行にわたる詳細なセットアップガイドが新規追加されました。このガイドには以下が含まれます：

**前提条件**
- WSL2 with CUDA 12.8（`/usr/local/cuda-12.8` にインストール）
- Python 3.x
- NVIDIA GPU とドライバ
- `nvidia-smi` が動作すること

**環境構築手順**
- vLLM 仮想環境の作成
- SGLang 仮想環境の作成
- kvcached のインストールと統合
- WSL2 固有の環境変数設定

**両エンジンのサポート**
- **SGLang**: kvcached と完全互換、すべてのエラスティックメモリ機能が動作
- **vLLM**: `--no-enable-prefix-caching` フラグを使用することで kvcached と完全互換

**必須環境変数**
```bash
export ENABLE_KVCACHED=true
export KVCACHED_AUTOPATCH=1
export CUDA_HOME=/usr/local/cuda-12.8
export LIBRARY_PATH=/usr/lib/wsl/lib:$LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH
```

#### 2. 仮想メモリアドレスの WSL2 最適化（constants.hpp）

WSL2 環境でのマルチプロセス互換性を確保するため、仮想アドレス開始位置を大幅に変更しました：

**変更前**
```cpp
static constexpr size_t kStartAddr = 0x1f0'000'000'000;  // 33.8TB
```

**変更後**
```cpp
static constexpr size_t kStartAddr = 0x2'000'000'00;  // 8GB (WSL2 compatibility)
```

この変更により、WSL2 の制約の中で安定した動作が可能になりました。

### WSL2 固有の技術的課題と解決策

#### 課題 1: CUDA ライブラリパス

**問題**: デフォルトでは CUDA ライブラリが見つからない
**解決策**: WSL2 固有のライブラリパス `/usr/lib/wsl/lib` を明示的に指定

```bash
export LIBRARY_PATH=/usr/lib/wsl/lib:$LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH
```

#### 課題 2: CUDA Graph のキャプチャエラー

**問題**: SGLang で CUDA グラフキャプチャ時に "requires grad" エラー
**解決策**: `--disable-cuda-graph` フラグを使用

```bash
python3 -m sglang.launch_server \
  --model openai-community/gpt2 \
  --disable-radix-cache \
  --disable-cuda-graph \
  --port 30000
```

#### 課題 3: vLLM のフロントエンドマルチプロセッシング

**問題**: デフォルトの設定では API リクエストがタイムアウト
**解決策**: `--disable-frontend-multiprocessing` フラグを使用

```bash
vllm serve openai-community/gpt2 \
  --disable-frontend-multiprocessing \
  --no-enable-prefix-caching \
  --port 8000
```

#### 課題 4: プレフィックスキャッシング非対応

**問題**: kvcached はプレフィックスキャッシングをまだサポートしていない
**解決策**: vLLM 0.11.0 ではデフォルトで有効なため、明示的に無効化

```bash
# vLLM
vllm serve model_name --no-enable-prefix-caching

# SGLang
python -m sglang.launch_server --model model_name --disable-radix-cache
```

### 検証済みバージョン

- **vLLM**: 0.11.0
- **SGLang**: 0.5.5.post2
- **PyTorch**: 2.5.1+cu121
- **CUDA**: 12.8 (WSL2)

## 実用例

### SGLang での起動

```bash
# 環境変数設定
export ENABLE_KVCACHED=true
export KVCACHED_AUTOPATCH=1
export CUDA_HOME=/usr/local/cuda-12.8
export LIBRARY_PATH=/usr/lib/wsl/lib:$LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH

# 仮想環境アクティベート
source ~/Work/kvcached/kvcached/vllm-venv/bin/activate

# サーバー起動
python3 -m sglang.launch_server \
  --model openai-community/gpt2 \
  --disable-radix-cache \
  --disable-cuda-graph \
  --trust-remote-code \
  --port 30000 \
  --host 0.0.0.0
```

### 期待されるログ出力

```
[kvcached][INFO] Successfully patched sglang: elastic_allocator, elastic_memory_pool, scheduler_memory_leak
[kvcached][INFO] Init kvcached KV cache allocator: num_layers=12, mem_size_per_layer=363MB, total_mem_size=8735MB
[kvcached][INFO] VirtualKV Cache is allocated. #tokens: 248479, K size: 7.99 GB, V size: 7.99 GB
[kvcached][INFO] Physical KV Cache limits by --mem-fraction-static: #tokens: 248479, K size: 4.27 GB, V size: 4.27 GB
INFO:     Application startup complete.
```

### API テスト

```bash
curl -s -X POST http://127.0.0.1:30000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "openai-community/gpt2",
    "prompt": "Once upon a time",
    "max_tokens": 20,
    "temperature": 0
  }'
```

## まとめ

今回の WSL2 対応アップデートにより、kvcached は Windows 環境でも利用可能になりました。主なポイントは以下の通りです：

1. **包括的なドキュメント**: SETUP_GUIDE.md により、WSL2 環境での完全なセットアップ手順が提供されました
2. **メモリアドレス最適化**: 仮想アドレス開始位置を 33.8TB から 8GB に変更し、WSL2 のマルチプロセス互換性を確保
3. **エンジン対応**: SGLang と vLLM の両方で動作確認済み
4. **トラブルシューティング**: WSL2 固有の課題とその解決策を詳細に文書化

kvcached を使用することで、以下のような革新的な GPU 活用が可能になります：

- 複数の LLM が同一 GPU 上でメモリを動的に共有
- オンデマンドなモデルの起動・停止によるサーバーレス展開
- トレーニングと推論のワークロード共存
- 2〜28 倍の TTFT 削減による大幅なコスト削減

Windows ユーザーも WSL2 を通じて、この革新的な GPU 共有技術の恩恵を受けられるようになりました。

## 参考リンク

- [kvcached GitHub リポジトリ](https://github.com/ovg-project/kvcached)
- [技術ブログ（英語）](https://yifanqiao.notion.site/Solve-the-GPU-Cost-Crisis-with-kvcached-289da9d1f4d68034b17bf2774201b141)
- [論文: Towards Efficient and Practical GPU Multitasking](https://arxiv.org/abs/2508.08448)
- [論文: Prism - Multi-LLM Serving](https://arxiv.org/abs/2505.04021)
