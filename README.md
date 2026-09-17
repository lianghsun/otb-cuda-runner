# otb-cuda-runner

幫 [OpenTWBench](https://opentwbench.ai) 補跑 7 顆模型的腳本。任何一台 Linux + NVIDIA GPU 的機器都能跑，
跑完把結果送回，就會用跟榜上其他模型完全相同的方式計分上榜。

謝謝你幫忙 🙏

## 要跑哪些模型

清單在 [`models.txt`](models.txt)，由小到大：

- LiquidAI/LFM2-700M、LiquidAI/LFM2-1.2B
- ibm-granite/granite-4.0-micro、granite-4.0-h-tiny、granite-4.0-h-small
- taide/Llama-3.1-TAIDE-LX-8B-Chat
- taide/Gemma-3-TAIDE-12b-Chat-2602

前 6 顆其實已經在榜上，但當時用的題目還沒剔除壞題，需要用清理後的題目重算；Gemma-3-TAIDE 是新模型。

## 需要什麼樣的機器

依顯卡大概能跑到哪裡：

- 1 張 24GB（RTX 3090 / 4090）：前 5 顆
- 2 張 24GB，或 1 張 48GB（A6000 / L40S）：再加 Gemma-3-TAIDE，共 6 顆
- 1 張 80GB（A100 / H100）：7 顆全部；granite-4.0-h-small 偏緊，建議 `GPU_MEM_UTIL=0.95 ./run.sh`
- 4 張 24GB、2 張 48GB 以上：7 顆全部

放不下的模型會自動略過，不會白下載。其他條件：

- NVIDIA 驅動 ≥ 550（≥ 580 會裝跟 GB200 那輪相同的 vLLM 0.28；較舊的驅動改裝 vLLM 0.13）
- 磁碟 ≥ 100 GB 空間放模型權重（每顆跑完會自動刪掉）
- 記憶體建議 ≥ 64 GB
- 能連 Hugging Face

時間粗估：每顆模型要答約 24,700 題。大卡上小模型十幾分鐘；消費級顯卡上 8B 以上的模型可能要一兩個小時。
多卡機器會同時跑好幾顆（每顆只佔放得下的最少張卡），總時間會短很多。

## 步驟

### 1. 下載並設定 token

```bash
git clone https://github.com/lianghsun/otb-cuda-runner.git
cd otb-cuda-runner
echo 'HF_TOKEN=hf_你拿到的token' > .env
chmod 600 .env
```

評測題目放在未公開的資料集，所以一定要有 Hugging Face token（向評測負責人拿，或用你自己已獲授權的帳號）。
`.env` 不會被 git 追蹤。

如果用的是**你自己帳號**的 token，請先到這兩頁按同意授權（送出後自動核准）：

- https://huggingface.co/taide/Llama-3.1-TAIDE-LX-8B-Chat
- https://huggingface.co/taide/Gemma-3-TAIDE-12b-Chat-2602

### 2. 體檢

```bash
python3 preflight.py
```

不用先安裝任何東西。它會檢查顯卡、驅動、token 權限、磁碟空間，並列出這台機器每顆模型用幾張卡、放不放得下。
有 ✗ 的項目先處理，⏭ 代表顯存不夠、會自動略過，不影響其他模型。

### 3. 安裝

```bash
./setup.sh
```

建立 `.venv` 並安裝 vLLM（第一次約 5–10 分鐘），裝完會實際在每張卡上跑一次運算確認可用。

### 4. 跑評測

```bash
tmux new -s otb      # 建議在 tmux / screen 裡跑，斷線也不會中斷
./run.sh LiquidAI/LFM2-700M     # 第一次先跑最小的一顆，確認環境沒問題（幾分鐘）
./run.sh                        # 再跑全部；已完成的會自動跳過
```

- 中斷了（斷電、Ctrl+C、當機）直接再跑一次 `./run.sh`，已經算過的題目不會重算
- 只想跑某幾顆：`./run.sh LiquidAI/LFM2-700M taide/Llama-3.1-TAIDE-LX-8B-Chat`
- 輸出都存在 `logs/`；多卡並行時每顆模型各有一份 log

### 5. 送回結果

```bash
./send_results.sh
```

它會先驗證每顆模型的結果完整、題目對得上目前的題庫，然後：

- 推到這個 repo 的 `results/<主機名稱>-<時間>` 分支（需要 repo 的寫入權限；不會動到你的工作目錄）
- 同時打包一份 `otb-results-*.tar.gz`

沒有寫入權限的話，用 `NO_PUSH=1 ./send_results.sh` 只打包，再把 tar.gz 傳給評測負責人即可。
沒跑完的模型不會被送出，可以先送已完成的，之後再送一次。

## 會送出什麼

每一題只有「題目雜湊、答對與否、有沒有抓到答案格式」這幾個值，外加一份溯源紀錄
（`results/_provenance/`：GPU 型號、驅動、vLLM / torch 版本、批次參數）。
**不含**題目、選項或模型的回答。

評測題目是未公開資料，請不要另外保存或轉傳。跑完後可以刪掉快取：

```bash
rm -rf ~/.cache/huggingface/hub/datasets--OpenTWBench--otb-mac-data
```

## 疑難排解

- **`setup.sh` 最後說 torch 看不到 GPU**：驅動太舊，撐不起這版 CUDA。試 `VLLM_VERSION=0.13.0 ./setup.sh`，或升級驅動。
- **讀不到 config.json（HTTP 401 / 403）**：token 沒有該模型的權限。TAIDE 模型要先到頁面同意授權（見步驟 1）。
- **載入時 `CUDA out of memory`**：GPU 上有其他程式在用。`GPU_MEM_UTIL=0.8 ./run.sh`（預設 0.90），或 `OTB_MAX_NUM_SEQS=128 ./run.sh`。
- **只想用其中幾張卡**：`CUDA_VISIBLE_DEVICES=2,3 ./run.sh`。
- **編譯 kernel / CUDA graph 相關錯誤**：`ENFORCE_EAGER=1 ./run.sh`（會慢一些，但最穩）。
- **某顆一直失敗**：其他模型照跑不受影響。把 `logs/` 裡那顆的 log 傳給評測負責人。
- **想關掉多卡並行**，改成一次一顆、用滿所有卡：`OTB_PARALLEL=0 ./run.sh`。

## 評測協定的一致性

[`eval_vllm.py`](eval_vllm.py) 原封不動取自 `lianghsun/otb-b200-runner` commit `6e0ef175`（GB200 那輪用的同一份），
SHA-256 `38ed7748…2e87`。`run.py` 啟動時會核對雜湊，被改過就拒絕執行。

題目選項洗牌（以題目雜湊為種子）、system prompt、greedy decoding、答案萃取規則、結果格式都在這份檔案裡。
`run.py` 只調整「怎麼把模型跑起來」：TP 大小、批次參數、多卡排程、載入前的相容性檢查，不碰計分邏輯。

## 給評測負責人

### 開權限

GitHub（讓朋友能推結果分支；只傳 tar.gz 就不需要）：

```bash
gh api -X PUT repos/lianghsun/otb-cuda-runner/collaborators/<朋友的GitHub帳號> -f permission=push
```

Hugging Face，擇一：

- **把朋友加進 OpenTWBench 組織（read 角色）**，朋友用自己的 read token，並自己到 TAIDE 頁面同意授權。最乾淨，事後移除成員即可。
- **開一把 fine-grained token 給朋友**：Repositories permissions 只給 `OpenTWBench/otb-mac-data` 讀取；
  另勾「Read access to contents of all public gated repos you can access」（TAIDE 需要）。跑完就 revoke。

不要把自己平常用的 token 給出去。

### 收結果

```bash
git fetch origin 'refs/heads/results/*:refs/remotes/origin/results/*'
git branch -r | grep results/
git archive origin/results/<分支名> results | tar -x -C <目的地>
```

或直接解朋友傳來的 tar.gz。`results/<模型>/` 的格式與 GB200 那輪完全相同（`formosa.jsonl`、`exam.jsonl`、`.done`），
放到 home-srv 收 GB200 結果的地方，照原本流程跑 `fold_mac.py`。
`_provenance/` 是這個 runner 多出來的溯源紀錄，fold 用不到可先拿掉。

收之前可再驗一次：

```bash
.venv/bin/python verify_results.py
```

## 環境變數

- `HF_TOKEN`：必填，可寫在 `.env`
- `GPU_MEM_UTIL`：vLLM 可用的顯存比例，預設 0.90
- `CUDA_VISIBLE_DEVICES`：限定使用哪幾張卡
- `OTB_PARALLEL=0`：關閉多卡並行
- `TP`：強制指定 tensor parallel 大小（同時會關閉並行；不整除 head 數時自動往下調）
- `OTB_MAX_BATCHED_TOKENS`、`OTB_MAX_NUM_SEQS`：批次參數；預設單卡 ≥ 70GB 時 65536 / 1024，否則 16384 / 256
- `ENFORCE_EAGER=1`：關閉 CUDA graph
- `VLLM_VERSION`：`setup.sh` 要裝的 vLLM 版本
- `NO_PUSH=1`：`send_results.sh` 只打包不推送
