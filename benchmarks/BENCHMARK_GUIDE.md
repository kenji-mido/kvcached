# KVcached ベンチマークガイド

本ドキュメントでは、KVcached の有効/無効を比較するベンチマーク手順を説明します。

---

## KVcached の本質

### なぜ複数モデルを同時に動かせるのか

**従来の vLLM/SGLang では:**

- 各モデルが KV cache を事前に確保（`gpu-memory-utilization` × GPU メモリ）
- 2モデルで 0.5 × 2 = 100% → 3モデル目は OOM で起動不可

**KVcached では:**

- **Model weights**: 物理 GPU メモリに常駐（必須、従来と同じ）
- **KV cache**: 共有プールからオンデマンド割り当て（ここが違う）

### 極端な例: 16GB GPU で 10 モデル同時運用

```
┌─────────────────────────────────────────────────────────────┐
│                      16GB GPU                               │
│                                                             │
│  Model weights (10モデル × 0.5GB each) = 5GB [常駐]         │
│  ├─ Model 1 (0.5GB) ✓ ロード済                              │
│  ├─ Model 2 (0.5GB) ✓ ロード済                              │
│  ├─ ...                                                     │
│  └─ Model 10 (0.5GB) ✓ ロード済                             │
│                                                             │
│  KVCache Pool: 11GB [共有・オンデマンド]                    │
│  ┌─────────────────────────────────────────────────────────┐│
│  │ ████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ ││
│  │ 1.5GB 使用中 (Model 2, 7 がアクティブ)                  ││
│  │ 9.5GB 空き (他モデルのリクエストに使用可能)             ││
│  └─────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────┘
```

スパースなリクエストパターン（同時アクティブ 1-2 モデル）なら、
KVCache 消費は 1-2GB 程度 → **10モデル同時運用が可能**

### 制約

| 項目 | オーバーコミット | 説明 |
|------|------------------|------|
| Model weights | ❌ 不可 | 物理 GPU メモリに載る必要あり |
| KV cache | ✅ 可能 | 仮想メモリ管理、オンデマンド割り当て |

- 同時リクエストの KVCache 消費がプールサイズを超えると OOM
- Model weights の合計が GPU メモリを超えるとロード不可

### Without KVcached での選択肢

GPU メモリが限られている場合、以下のどちらかを選ぶ必要がある：

| 方式 | TTFT | GPU メモリ効率 | 運用 |
|------|------|----------------|------|
| 全モデル事前ロード | ~50ms | ❌ 悪い（各モデルがKVcache予約） | 限られたモデル数 |
| オンデマンドロード | 5,000-30,000ms | ✅ 良い | モデル切替時に待ち |

**KVcached なら両立可能**: 全モデル常駐 + KVcache 共有 → TTFT ~50ms + メモリ効率 ✅

---

## KVcached の動作原理

### WITH KVcached (有効時)

```
┌─────────────────────────────────────────────────────────────────────┐
│                            GPU (16 GB)                              │
│                                                                     │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │              Shared KVCache Pool (4.5 GB)                     │  │
│  │  ┌─────────────────────┐  ┌─────────────────────┐             │  │
│  │  │ Llama KV Blocks     │  │ Qwen KV Blocks      │             │  │
│  │  │ (allocated on       │  │ (allocated on       │             │  │
│  │  │  demand)            │  │  demand)            │             │  │
│  │  └─────────────────────┘  └─────────────────────┘             │  │
│  │                    ▲ Dynamic Sharing ▲                        │  │
│  └────────────────────┼─────────────────┼────────────────────────┘  │
│                       │                 │                           │
│  ┌────────────────────┴──┐  ┌──────────┴─────────────────┐         │
│  │ vLLM Server (Llama)   │  │ vLLM Server (Qwen)         │         │
│  │ Model Weights: 2.3 GB │  │ Model Weights: 1.0 GB      │         │
│  │ Port: 12346           │  │ Port: 12347                │         │
│  └───────────────────────┘  └────────────────────────────┘         │
│                                                                     │
│  Total GPU Used: ~11.5 GB                                           │
└─────────────────────────────────────────────────────────────────────┘

特徴:
- KV キャッシュメモリは共有プールで動的に割り当て
- Llama がアイドル時、Qwen がより多くの KV ブロックを使用可能
- 仮想メモリマッピングにより柔軟な割り当てを実現
```

### WITHOUT KVcached (無効時)

```
┌─────────────────────────────────────────────────────────────────────┐
│                            GPU (16 GB)                              │
│                                                                     │
│  ┌─────────────────────────────┐  ┌─────────────────────────────┐  │
│  │ vLLM Server (Llama)         │  │ vLLM Server (Qwen)          │  │
│  │                             │  │                             │  │
│  │ Model Weights: 2.3 GB       │  │ Model Weights: 1.0 GB       │  │
│  │                             │  │                             │  │
│  │ ┌─────────────────────────┐ │  │ ┌─────────────────────────┐ │  │
│  │ │ Reserved KV Cache       │ │  │ │ Reserved KV Cache       │ │  │
│  │ │ (Pre-allocated)         │ │  │ │ (Pre-allocated)         │ │  │
│  │ │                         │ │  │ │                         │ │  │
│  │ │ 5.5 GB                  │ │  │ │ 5.5 GB                  │ │  │
│  │ │ (35% × 16GB)            │ │  │ │ (35% × 16GB)            │ │  │
│  │ │                         │ │  │ │                         │ │  │
│  │ │ ████████░░░░░░░░░░░░░░░ │ │  │ │ ██████░░░░░░░░░░░░░░░░░ │ │  │
│  │ │ (Often underutilized)   │ │  │ │ (Often underutilized)   │ │  │
│  │ └─────────────────────────┘ │  │ └─────────────────────────┘ │  │
│  │                             │  │                             │  │
│  │ Port: 12346                 │  │ Port: 12347                 │  │
│  └─────────────────────────────┘  └─────────────────────────────┘  │
│                                                                     │
│  Total GPU Used: ~15.5 GB                                           │
│  ⚠️  各サーバーが固定メモリを予約 - 共有不可                        │
└─────────────────────────────────────────────────────────────────────┘

特徴:
- KV キャッシュメモリはモデルごとに分離・事前割り当て
- モデルがアイドル時もメモリは無駄に予約されたまま
- OOM を避けるため gpu-memory-utilization を慎重に設定する必要あり
```

### KVcached が効果的なシナリオ

**1. 開発・テスト環境**

```
┌─────────────────────────────────────────────────────────────────┐
│ 1台の GPU で複数モデルを同時に試したい                          │
│                                                                 │
│ Without KVcached:                                               │
│   - gpu-memory-utilization を慎重に計算                        │
│   - 3つ目のモデルを追加 → OOM で起動失敗                        │
│                                                                 │
│ With KVcached:                                                  │
│   - 共有プールから動的割り当て                                  │
│   - モデル追加が容易（使用中の分だけメモリ消費）                │
└─────────────────────────────────────────────────────────────────┘
```

**2. マルチテナント推論**

```
┌─────────────────────────────────────────────────────────────────┐
│ 複数のアプリ/ユーザーが GPU リソースを共有                      │
│                                                                 │
│ Without KVcached:                                               │
│   - 各テナントに固定メモリを割り当て                            │
│   - テナント A がアイドルでもメモリは解放されない               │
│   - リソースの無駄                                              │
│                                                                 │
│ With KVcached:                                                  │
│   - アイドルテナントの KV cache は他テナントが使用可能          │
│   - 効率的なリソース共有                                        │
└─────────────────────────────────────────────────────────────────┘
```

**3. スパースなリクエストパターン**

```
┌─────────────────────────────────────────────────────────────────┐
│ 各モデルへのリクエストが散発的（同時アクティブが少ない）        │
│                                                                 │
│ 例: 翻訳モデル + 要約モデル + QA モデル                         │
│     → 同時に使われることは稀                                    │
│                                                                 │
│ Without KVcached:                                               │
│   - 3モデル分の KV cache を常に予約 → GPU メモリ不足            │
│   - または、オンデマンドロード → TTFT 5〜30秒                   │
│                                                                 │
│ With KVcached:                                                  │
│   - 全モデルのweightsをロード済み                               │
│   - KV cache はアクティブなリクエストにのみ割り当て             │
│   - TTFT ~50ms を維持                                           │
└─────────────────────────────────────────────────────────────────┘
```

### 比較サマリー

```
                        WITH KVcached          WITHOUT KVcached
                        ─────────────          ────────────────
メモリモデル:           共有プール             モデルごとに分離
割り当て方式:           オンデマンド           事前割り当て
GPU メモリ使用量:       11.5 GB                15.5 GB
メモリ効率:             ████████████░░ 74%     ██████████████ 100%
                        (必要に応じて拡張)     (固定、未使用分は無駄)

TTFT:                   38.79 ms               41.73 ms
                        (メモリ競合が少ない)   (競合が多い)

モデル追加:             ✅ 容易                ❌ 困難
                        (プールが伸縮)         (割当を再計算必要)

アイドルモデル:         ✅ メモリ解放          ❌ メモリ無駄
                        (他モデルが使用可能)   (未使用のまま)
```

### ベンチマーク結果（実測値）

```
┌────────────────────┬───────────────┬────────────────┬─────────────┐
│ 指標               │ KVcached ON   │ KVcached OFF   │ 改善率      │
├────────────────────┼───────────────┼────────────────┼─────────────┤
│ Peak GPU Memory    │ 11.49 GB      │ 15.54 GB       │ -26.1%      │
│ Peak KVCache       │ 0.60 GB       │ N/A (固定)     │ 動的       │
│ TTFT               │ 38.79 ms      │ 41.73 ms       │ +7.0%       │
│ ITL                │ 6.57 ms       │ 6.76 ms        │ +2.8%       │
└────────────────────┴───────────────┴────────────────┴─────────────┘

テスト条件: Llama-3.2-1B + Qwen2.5-0.5B、各20リクエスト @ 2 req/s
```

### 実運用での考慮事項

KVcached なしで複数モデルを運用する場合、以下の2つの選択肢があります：

**Option 1: 全モデルを事前ロード（本ベンチマークの方式）**

```
┌─────────────────────────────────────────────────────────────────┐
│ - gpu-memory-utilization を低く設定 (0.35 × 2 = 0.70)          │
│ - 各モデルの KV cache が小さくなる                              │
│ - 同時リクエスト数が制限される                                  │
│ - TTFT: ~50ms (モデルはロード済み)                              │
└─────────────────────────────────────────────────────────────────┘
```

**Option 2: オンデマンドでモデルをロード（メモリ待ちロジック）**

```
┌─────────────────────────────────────────────────────────────────┐
│ - リクエスト来る → メモリ不足 → 他モデルをアンロード            │
│ - → モデルをロード (数秒〜数十秒)                               │
│ - TTFT: 5,000ms〜30,000ms+ (モデルロード時間が加算)             │
│ - ユーザー体験が大幅に悪化                                      │
└─────────────────────────────────────────────────────────────────┘
```

**KVcached: 両方の利点を実現**

```
┌─────────────────────────────────────────────────────────────────┐
│ - 全モデルが常にロード済み (weights in GPU)                     │
│ - KV cache はオンデマンドで共有プールから割り当て               │
│ - TTFT: ~50ms (常に即座に応答可能)                              │
│ - メモリ効率: 25% 削減                                          │
└─────────────────────────────────────────────────────────────────┘
```

**実際の比較:**

| シナリオ | TTFT | GPU メモリ |
|----------|------|------------|
| KVcached ON | ~50ms | 11.6 GB |
| KVcached OFF (事前ロード) | ~50ms | 15.5 GB |
| KVcached OFF (オンデマンド) | **5,000〜30,000ms** | ~8 GB (1モデル時) |

本ベンチマークは「KVcached OFF + 事前ロード」と比較していますが、GPU メモリが限られた実環境では「オンデマンドロード」になる可能性が高く、その場合 **TTFT は 100倍以上悪化** します。

---

## 前提条件

### 必要なソフトウェア
- CUDA Toolkit 13.0+
- Python 3.12+
- kvcached (ビルド済み)
- vLLM 0.14+

### インストール手順

```bash
# CUDA Toolkit のインストール（Ubuntu 24.04）
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt install cuda-toolkit-13-0

# Python 開発ヘッダー
sudo apt install python3.12-dev

# 環境変数設定
export PATH=/usr/local/cuda-13.0/bin:$PATH
export CUDA_HOME=/usr/local/cuda-13.0

# kvcached ビルド
cd /path/to/kvcached
pip install -e . --no-build-isolation --no-cache-dir

# autopatch 用 .pth ファイルのコピー（重要！）
python tools/dev_copy_pth.py
```

### autopatch の確認

KVcached の autopatch が正しく設定されているか確認：

```bash
# .pth ファイルの存在確認
python tools/dev_copy_pth.py --check

# vLLM 起動後、ログに以下が表示されれば有効
# [kvcached] Applying 6 patches for vllm
# [kvcached] Successfully patched vllm: ...
```

---

## ベンチマーク1: シングルモデル比較

シングルモデルでの KVcached オーバーヘッド確認。

### サーバー起動

**KVcached あり:**

```bash
export PATH=/usr/local/cuda-13.0/bin:$PATH
export CUDA_HOME=/usr/local/cuda-13.0
export ENABLE_KVCACHED=true
export KVCACHED_AUTOPATCH=1

python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Llama-3.2-1B \
  --port 12346 \
  --no-enable-prefix-caching \
  --gpu-memory-utilization 0.8 \
  --max-model-len 4096
```

**KVcached なし:**

```bash
export PATH=/usr/local/cuda-13.0/bin:$PATH
export CUDA_HOME=/usr/local/cuda-13.0
export ENABLE_KVCACHED=false

python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Llama-3.2-1B \
  --port 12346 \
  --gpu-memory-utilization 0.8 \
  --max-model-len 4096
```

### ベンチマーク実行

```bash
vllm bench serve \
  --model meta-llama/Llama-3.2-1B \
  --port 12346 \
  --num-prompts 50 \
  --request-rate 5
```

### 期待される結果

シングルモデルでは KVcached の有無で性能差はほぼなし（オーバーヘッドなし）。

---

## ベンチマーク2: 複数モデル同時実行比較

KVcached の真のメリットを確認するベンチマーク。

### サーバー起動（2モデル）

**KVcached あり:**

```bash
# ターミナル1: Model A
export PATH=/usr/local/cuda-13.0/bin:$PATH
export CUDA_HOME=/usr/local/cuda-13.0
export ENABLE_KVCACHED=true
export KVCACHED_AUTOPATCH=1
export KVCACHED_IPC_NAME=BENCH

python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Llama-3.2-1B \
  --port 12346 \
  --no-enable-prefix-caching \
  --gpu-memory-utilization 0.35 \
  --max-model-len 4096

# ターミナル2: Model B
export PATH=/usr/local/cuda-13.0/bin:$PATH
export CUDA_HOME=/usr/local/cuda-13.0
export ENABLE_KVCACHED=true
export KVCACHED_AUTOPATCH=1
export KVCACHED_IPC_NAME=BENCH

python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-0.5B \
  --port 12347 \
  --no-enable-prefix-caching \
  --gpu-memory-utilization 0.35 \
  --max-model-len 4096
```

**KVcached なし:**

```bash
# ターミナル1: Model A
export PATH=/usr/local/cuda-13.0/bin:$PATH
export CUDA_HOME=/usr/local/cuda-13.0
export ENABLE_KVCACHED=false

python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Llama-3.2-1B \
  --port 12346 \
  --gpu-memory-utilization 0.35 \
  --max-model-len 4096

# ターミナル2: Model B
export PATH=/usr/local/cuda-13.0/bin:$PATH
export CUDA_HOME=/usr/local/cuda-13.0
export ENABLE_KVCACHED=false

python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-0.5B \
  --port 12347 \
  --gpu-memory-utilization 0.35 \
  --max-model-len 4096
```

### ベンチマーク実行（同時実行）

```bash
# 両モデルに同時にリクエストを送信
vllm bench serve --model meta-llama/Llama-3.2-1B --port 12346 --num-prompts 30 --request-rate 5 &
vllm bench serve --model Qwen/Qwen2.5-0.5B --port 12347 --num-prompts 30 --request-rate 5 &
wait
```

### 期待される結果（参考値）

| モデル | 指標 | KVcached あり | KVcached なし | 改善 |
|--------|------|---------------|---------------|------|
| Qwen2.5-0.5B | TTFT (mean) | ~45ms | ~57ms | ~20%改善 |
| Llama-3.2-1B | TTFT (mean) | ~77ms | ~95ms | ~19%改善 |

---

## 重要なパラメータ

### 環境変数

| 変数 | 説明 | 値 |
|------|------|-----|
| `ENABLE_KVCACHED` | KVcached 有効化 | `true` / `false` |
| `KVCACHED_AUTOPATCH` | 自動パッチ有効化 | `1` |
| `KVCACHED_IPC_NAME` | 複数プロセス間の共有メモリ名 | 任意の文字列 |
| `CUDA_HOME` | CUDA インストールパス | `/usr/local/cuda-13.0` |

### vLLM オプション

| オプション | 説明 |
|------------|------|
| `--no-enable-prefix-caching` | KVcached 使用時は必須 |
| `--gpu-memory-utilization` | GPU メモリ使用率（複数モデル時は合計100%以下に） |
| `--max-model-len` | 最大シーケンス長（メモリ使用量に影響） |

### ベンチマーククライアント (vllm bench serve)

```bash
vllm bench serve \
  --model <モデル名> \
  --port <ポート番号> \
  --num-prompts <リクエスト数> \
  --request-rate <リクエスト/秒>
```

| オプション | 説明 | 例 |
|------------|------|-----|
| `--model` | ベンチマーク対象のモデル名 | `meta-llama/Llama-3.2-1B` |
| `--port` | サーバーのポート番号 | `12346` または `8080`（ルーター経由） |
| `--num-prompts` | 送信するリクエスト総数 | `30`, `50`, `100` |
| `--request-rate` | 1秒あたりのリクエスト数 | `5`, `10` |

**複数モデル同時ベンチマーク:**

```bash
# バックグラウンドで並列実行
vllm bench serve --model meta-llama/Llama-3.2-1B --port 8080 --num-prompts 30 --request-rate 5 &
vllm bench serve --model Qwen/Qwen2.5-0.5B --port 8080 --num-prompts 30 --request-rate 5 &
wait  # 両方の完了を待つ
```

---

## トラブルシューティング

### "No available memory for the cache blocks" エラー
- `--gpu-memory-utilization` を下げる
- `--max-model-len` を小さくする

### 2つ目のモデルが起動しない
- `KVCACHED_IPC_NAME` が両プロセスで同じか確認
- GPU メモリ使用量を確認（`nvidia-smi`）
- 最初のモデルが完全に起動してから2つ目を起動

### nvcc が見つからない

```bash
export PATH=/usr/local/cuda-13.0/bin:$PATH
export CUDA_HOME=/usr/local/cuda-13.0
```

---

## ベンチマーク3: コントローラーを使用した複数モデル管理

コントローラーを使用すると、YAML 設定ファイルで複数モデルを宣言的に管理し、統一された HTTP エンドポイント経由でアクセスできます。

### 設定ファイルの準備

`controller/example-config.yaml` を編集します：

```yaml
kvcached:
  kvcached_gpu_utilization: 0.95
  kvcached_page_prealloc_enabled: true
  kvcached_min_reserved_pages: 5
  kvcached_max_reserved_pages: 10
  kvcached_sanity_check: false
  kvcached_log_level: INFO

router:
  enable_router: true
  router_port: 8080
  router_host: localhost

sleep_manager:
  idle_threshold_seconds: 300
  check_interval_seconds: 60
  auto_sleep_enabled: false
  wakeup_on_request: true
  min_sleep_duration: 80

launch_delay_seconds: 60  # モデル間の起動待ち時間

instances:
  - name: llama
    model: meta-llama/Llama-3.2-1B
    engine: vllm
    using_venv: true
    venv_path: /path/to/your/venv  # 実際のパスに変更
    kvcached_env:
      - "ENABLE_KVCACHED=true"
      - "KVCACHED_AUTOPATCH=1"
      - "KVCACHED_IPC_NAME=CONTROLLER"  # 重要: 全インスタンスで同じ値
    engine_env:
      - "VLLM_USE_V1=1"
      - "VLLM_ATTENTION_BACKEND=FLASH_ATTN"
      - "CUDA_HOME=/usr/local/cuda-13.0"
      - "PATH=/usr/local/cuda-13.0/bin:/usr/bin:/bin"
    engine_args:
      - "--disable-log-requests"
      - "--no-enable-prefix-caching"
      - "--host=localhost"
      - "--port=12346"
      - "--gpu-memory-utilization=0.35"
      - "--max-model-len=4096"
  - name: qwen
    model: Qwen/Qwen2.5-0.5B
    engine: vllm
    using_venv: true
    venv_path: /path/to/your/venv  # 実際のパスに変更
    kvcached_env:
      - "ENABLE_KVCACHED=true"
      - "KVCACHED_AUTOPATCH=1"
      - "KVCACHED_IPC_NAME=CONTROLLER"  # 重要: 全インスタンスで同じ値
    engine_env:
      - "VLLM_USE_V1=1"
      - "VLLM_ATTENTION_BACKEND=FLASH_ATTN"
      - "CUDA_HOME=/usr/local/cuda-13.0"
      - "PATH=/usr/local/cuda-13.0/bin:/usr/bin:/bin"
    engine_args:
      - "--disable-log-requests"
      - "--no-enable-prefix-caching"
      - "--host=localhost"
      - "--port=12347"
      - "--gpu-memory-utilization=0.35"
      - "--max-model-len=4096"
```

### 重要な設定ポイント

- **`KVCACHED_IPC_NAME`**: 全インスタンスで同じ値を設定（共有メモリセグメント名）
- **`launch_delay_seconds`**: 最初のモデルが完全に起動してから次を起動するための待ち時間
- **`gpu-memory-utilization`**: 複数モデルの合計が 100% 以下になるよう設定

### コントローラーでの KVcached 有効/無効切り替え

**KVcached を有効にする場合:**

```yaml
kvcached_env:
  - "ENABLE_KVCACHED=true"
  - "KVCACHED_AUTOPATCH=1"
  - "KVCACHED_IPC_NAME=CONTROLLER"
```

**KVcached を無効にする場合:**

```yaml
kvcached_env:
  - "ENABLE_KVCACHED=false"
# または kvcached_env を空にする:
# kvcached_env: []
```

設定変更後はコントローラーの再起動が必要です（後述の「コントローラー再起動」参照）。

### コントローラー起動

```bash
cd controller

# コントローラー起動（バックエンド + フロントエンド）
python launch.py --config example-config.yaml

# 別ターミナルでフロントエンド（ルーター）を起動する場合
# ※ venv 環境の python を明示的に指定
/path/to/your/venv/bin/python frontend.py --config_path example-config.yaml --port 8080
```

### コントローラー再起動

設定変更後やトラブル時にコントローラーを再起動する手順：

```bash
cd controller

# 1. 全セッションを終了
python launch.py --kill-all

# 2. 再起動
python launch.py --config example-config.yaml

# セッション一覧で確認
python launch.py --list-sessions
# または
tmux list-sessions
```

**注意**: `can't find window: 0` というメッセージは無害なので無視してください。

### tmux セッションの確認

コントローラーは各モデルを tmux セッションで管理します：

```bash
# セッション一覧
tmux list-sessions

# 特定モデルのログ確認
tmux attach -t kvcached-llama
tmux attach -t kvcached-qwen

# tmux からデタッチ: Ctrl+B, D
```

### KVcached 状態の確認

`kvctl` コマンドで KVcached の状態を確認できます：

```bash
# セグメント一覧と使用量
kvctl list

# 特定の IPC セグメント
kvctl list CONTROLLER

# 出力例:
# IPC                             Limit         Used      %
# CONTROLLER                    3.09 GB       0.00 B   0.0 %

# 継続的に監視
kvctl watch CONTROLLER

# curses TUI で監視
kvtop CONTROLLER
```

**注意**: `kvctl list` で "No active KVCached segments found" と表示される場合は、`.pth` ファイルがコピーされていない可能性があります。`python tools/dev_copy_pth.py` を実行してコントローラーを再起動してください。

### ベンチマーク実行（ルーター経由）

統一エンドポイント（ポート 8080）経由で両モデルにリクエストを送信：

```bash
# 両モデルに同時にベンチマーク実行
vllm bench serve --model meta-llama/Llama-3.2-1B --port 8080 --num-prompts 30 --request-rate 5 &
vllm bench serve --model Qwen/Qwen2.5-0.5B --port 8080 --num-prompts 30 --request-rate 5 &
wait
```

### ベンチマークモニタリング（kvbench）

ベンチマーク実行中にリアルタイムでメトリクスを監視するには、別ターミナルで `kvbench` を使用：

```bash
# 基本的な使用方法
kvbench --endpoints localhost:12346,localhost:12347 --ipc CONTROLLER

# CSV ファイルに出力（後で分析用）
kvbench --endpoints localhost:12346,localhost:12347 --ipc CONTROLLER -o bench_results.csv

# リフレッシュ間隔を変更（デフォルト: 1秒）
kvbench --endpoints localhost:12346,localhost:12347 -r 0.5
```

**表示される情報:**
- GPU Memory: 全体のGPUメモリ使用量
- KVCache Segments: IPCセグメントごとのKVキャッシュ使用量
- Model Endpoints: 各モデルのメトリクス
  - Requests: 実行中/待機中/完了リクエスト数
  - Tokens: prompt/generation トークン数
  - Rate: トークン生成レート (tokens/s)、リクエストレート (req/s)
  - Latency: ITL (Inter-Token Latency)、E2E レイテンシ

**KVCache Segments の読み方:**

```
KVCache Segments
BENCHMARK                    4.50 GB
[################--------------------------------------------] 0.74 GB (16.4%)
```

| 値 | 説明 |
|----|------|
| **BENCHMARK** | IPC 名（共有メモリセグメントの識別子） |
| **4.50 GB** | KVCache プールの総サイズ（事前確保された仮想アドレス空間） |
| **0.74 GB** | 現在使用中の物理メモリ（実際に消費している GPU メモリ） |
| **16.4%** | 使用率 (0.74 / 4.50) |

```
┌─────────────────────────────────────────────────────────────┐
│           仮想アドレス空間 (4.50 GB)                        │
│  ┌─────────────────────────────────────────────────────────┐│
│  │ 予約済みだが未割当 (3.76 GB)                            ││
│  │ (将来のリクエストに使用可能)                            ││
│  ├─────────────────┬───────────────────────────────────────┤│
│  │ 実際に使用中    │                                       ││
│  │ (0.74 GB)       │                                       ││
│  │ ████████████████│                                       ││
│  └─────────────────┴───────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────┘
         ▲                              ▲
         │                              │
    物理 GPU メモリ               仮想のみ
    を消費                       (GPU メモリ未使用)
```

**動作中の変化:**

```
アイドル:   [##------------------------------------------] 0.1 GB / 4.5 GB
アクティブ: [##################--------------------------] 0.8 GB / 4.5 GB
ピーク:     [##########################------------------] 1.2 GB / 4.5 GB
完了後:     [####----------------------------------------] 0.2 GB / 4.5 GB
```

バーはリクエストの増減に応じて伸縮します。これが KVcached のオンデマンド割り当ての利点です。

終了は `q` キーを押します。

### 期待される結果（参考値）

コントローラー経由での結果（KVcached 有効時）：

| モデル | TTFT (mean) | Throughput |
|--------|-------------|------------|
| Qwen2.5-0.5B | ~54ms | ~560 tok/s |
| Llama-3.2-1B | ~80ms | ~550 tok/s |

### クリーンアップ

```bash
# tmux セッションを終了
tmux kill-session -t kvcached-llama
tmux kill-session -t kvcached-qwen

# フロントエンドを Ctrl+C で停止
```

---

## 既存ベンチマークツール

リポジトリには以下のベンチマークツールが含まれています：

| ディレクトリ | 用途 |
|--------------|------|
| `benchmarks/simple_bench/` | シンプルなサーバー/クライアントベンチマーク |
| `benchmarks/bench_kvcached_overhead/` | KVcached オーバーヘッド測定 |
| `benchmarks/bench_latency_benefit/` | 複数モデル同時負荷ベンチマーク |

### simple_bench の使用例

```bash
cd benchmarks/simple_bench

# サーバー起動
./start_server.sh vllm --model meta-llama/Llama-3.2-1B

# クライアント実行（別ターミナル）
./start_client.sh vllm --model meta-llama/Llama-3.2-1B
```

---

## ベンチマーク4: 統一ベンチマークスクリプト

`unified_benchmark.py` は、KVcached の有効/無効を自動的に比較し、包括的なレポートを生成するスクリプトです。

### 機能

- vLLM サーバーの自動起動・停止
- KVcached 有効/無効の両方でベンチマークを実行
- vllm bench serve によるクライアント負荷生成
- vLLM Prometheus メトリクス（TTFT, ITL, E2E, スループット）の収集
- GPU メモリ・KVCache 使用量のモニタリング
- 実行ごとの CSV 出力と JSON サマリーの生成
- 結果の比較分析

### 使用方法

**重要**: KVCache 情報を正しく収集するため、venv の Python で実行してください。

```bash
cd benchmarks

# venv Python で実行（推奨）
/path/to/venv/bin/python unified_benchmark.py \
  --models meta-llama/Llama-3.2-1B \
  --venv-path /path/to/venv \
  --cuda-home /usr/local/cuda

# 複数モデルの場合
/path/to/venv/bin/python unified_benchmark.py \
  --models meta-llama/Llama-3.2-1B,Qwen/Qwen2.5-0.5B \
  --num-prompts 30 \
  --request-rate 5 \
  --venv-path /path/to/venv \
  --cuda-home /usr/local/cuda \
  --output-dir ./results
```

### オプション

| オプション | 説明 | デフォルト |
|------------|------|------------|
| `--models` | ベンチマーク対象のモデル（カンマ区切りで複数指定可） | 必須 |
| `--venv-path` | Python 仮想環境のパス | なし |
| `--cuda-home` | CUDA インストールディレクトリ | `/usr/local/cuda` |
| `--num-prompts` | 送信するリクエスト数（モデルごと） | `30` |
| `--request-rate` | 1秒あたりのリクエスト数 | `5.0` |
| `--gpu-memory-utilization` | GPU メモリ使用率（モデルごと） | `0.35` |
| `--max-model-len` | 最大シーケンス長 | `4096` |
| `--output-dir` | 結果出力ディレクトリ | `./benchmark_results` |
| `--skip-disabled` | KVcached 無効時のテストをスキップ | `False` |
| `--include-ondemand` | オンデマンドロードベンチマークを含める（非常に遅い） | `False` |
| `--ondemand-requests` | オンデマンドベンチマークのリクエスト数 | `6` |

### オンデマンドロードベンチマーク（オプション）

`--include-ondemand` フラグを指定すると、GPU メモリが制限された環境でのオンデマンドモデルロードをシミュレートします。

**注意**: このベンチマークは非常に時間がかかります（各リクエストでモデルの起動・停止を行うため、1リクエストあたり20-60秒）。デフォルトでは無効化されています。

```bash
# オンデマンドベンチマークを含める（時間がかかる）
/path/to/venv/bin/python unified_benchmark.py \
  --models meta-llama/Llama-3.2-1B,Qwen/Qwen2.5-0.5B \
  --include-ondemand \
  --ondemand-requests 4
```

このベンチマークは、GPU メモリ不足時にモデルを動的にロード・アンロードするシナリオを示します。結果として TTFT が 5,000〜30,000ms になることが期待されます（通常の 50ms と比較）

### 出力ファイル

```
benchmark_results/
├── run_20260125_143000/
│   ├── kvcached_enabled_metrics.csv   # KVcached有効時のメトリクス
│   ├── kvcached_disabled_metrics.csv  # KVcached無効時のメトリクス
│   └── summary.json                   # 比較サマリー
└── run_20260125_150000/
    └── ...
```

### CSV 出力カラム

| カラム | 説明 |
|--------|------|
| `timestamp` | タイムスタンプ |
| `elapsed_sec` | 開始からの経過時間 |
| `requests_running` | 実行中リクエスト数 |
| `requests_waiting` | 待機中リクエスト数 |
| `requests_completed` | 完了リクエスト数（セッション累計） |
| `prompt_tokens` | プロンプトトークン数（セッション累計） |
| `generation_tokens` | 生成トークン数（セッション累計） |
| `prompt_tokens_rate` | プロンプトトークンレート (tokens/s) |
| `generation_tokens_rate` | 生成トークンレート (tokens/s) |
| `request_rate` | リクエストレート (req/s) |
| `ttft_ms` | TTFT 直近平均 (ms) |
| `itl_ms` | ITL 直近平均 (ms) |
| `e2e_ms` | E2E レイテンシ直近平均 (ms) |
| `kv_cache_usage_perc` | vLLM KVCache 使用率 (%) |
| `kvcache_used_bytes` | KVcached 使用量 (bytes) |
| `kvcache_total_bytes` | KVcached 総容量 (bytes) |
| `gpu_used_bytes` | GPU メモリ使用量 (bytes) |
| `gpu_total_bytes` | GPU メモリ総容量 (bytes) |

### JSON サマリー

`summary.json` には以下の比較データが含まれます：

```json
{
  "config": {
    "model": "meta-llama/Llama-3.2-1B",
    "num_prompts": 50,
    "request_rate": 5
  },
  "kvcached_enabled": {
    "mean_ttft_ms": 45.2,
    "mean_itl_ms": 12.3,
    "mean_e2e_ms": 234.5,
    "total_tokens": 5000,
    "throughput_tokens_per_sec": 166.7,
    "max_gpu_usage_bytes": 4294967296,
    "max_kvcache_usage_bytes": 1073741824
  },
  "kvcached_disabled": {
    "mean_ttft_ms": 57.8,
    "mean_itl_ms": 14.1,
    "mean_e2e_ms": 289.3,
    "total_tokens": 5000,
    "throughput_tokens_per_sec": 142.9,
    "max_gpu_usage_bytes": 5368709120
  },
  "comparison": {
    "ttft_improvement_pct": 21.8,
    "itl_improvement_pct": 12.8,
    "e2e_improvement_pct": 18.9,
    "throughput_improvement_pct": 16.7,
    "gpu_memory_saved_pct": 20.0
  }
}
```

### ワークフロー

1. vLLM サーバーを KVcached 有効で起動
2. ウォームアップ期間でサーバー安定化
3. vllm bench serve でベンチマーク実行
4. メトリクス収集（1秒間隔）
5. サーバー停止
6. vLLM サーバーを KVcached 無効で起動
7. 手順 2-5 を繰り返し
8. 結果を比較し JSON サマリー生成
