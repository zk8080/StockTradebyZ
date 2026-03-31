# AgentTrader

一个面向 A 股的半自动选股项目：

- 使用 **TDX API** 作为当前主数据源拉取股票日线数据
- 用量化规则做初选（当前正式主策略为启动版 `B1`，旧确认版保留为 `b1_legacy`）
- 导出候选股票 K 线图
- 调用 OpenAI 兼容视觉模型对图表进行 AI 复评打分
- 支持单股 AI 点评与后续飞书多维表格同步

---

## 当前状态（重要）

- 当前主数据源已经固定为 **TDX API**，不再以 Tushare 作为默认主抓取源
- 当前主流程是：`TDX 日线刷新 → 正式 B1 初选 → 候选图导出 → AI 复评 → 生成 Top10 产物`
- 当前正式 `B1` 是原先的启动/早期 `B1-E` 逻辑；旧确认版 `B1` 已下沉为 `b1_legacy`，默认仅供对比研究
- 当前正式 `B1` 的执行口径可先记成一句话：**低位 J + 涨放跌缩 + 知行线不坏 + 允许一次缩量轻破**
- `run_all.py` 当前**只负责生成本地产物与打印推荐结果**，**不负责自动写入飞书多维表格**
- 若需要形成完整闭环，应在主流程后继续执行：
  - Top10 payload 生成
  - `选股明细` 写入
  - `每日汇总` 写入
  - 写表结果回查
- 已支持单股 AI 点评闭环所需产物生成（JSON + Bitable payload）

---

## 1. 项目流程

完整流程对应 [run_all.py](run_all.py)：

1. 下载 K 线数据（`pipeline/fetch_kline_tdx_api.py`，当前主入口）
2. 量化初选（`pipeline.cli preselect`）
3. 导出候选图表（`dashboard/export_kline_charts.py`）
4. OpenAI 兼容复评（`agent/model_review.py`）
5. 打印推荐结果（读取 `suggestion.json`）

输出主链路：

- `data/raw`：原始日线 CSV
- `data/candidates`：初选候选列表
- `data/kline/日期`：候选图表
- `data/review/日期`：AI 单股评分与汇总建议
- `data/meta/bitable_top10_<date>.json`：Top10 写表 payload（需单独生成）
- `data/review_single/<date>`：单股 AI 点评结果

---

## 2. 目录说明

- [pipeline](pipeline)：数据抓取与量化初选
- [dashboard](dashboard)：看盘界面与图表导出
- [agent](agent)：LLM 评审逻辑（OpenAI 兼容）
- [config](config)：抓取、初选、模型复评配置
- [data](data)：运行数据与结果
- [run_all.py](run_all.py)：全流程一键入口

---

## 3. 快速开始（一键跑通）

### 3.1 Clone 项目

~~~bash
git clone https://github.com/SebastienZh/StockTradebyZ
cd StockTradebyZ
~~~

### 3.2 安装依赖

~~~bash
pip install -r requirements.txt
~~~

### 3.3 设置环境变量

Windows PowerShell（永久写入）：

~~~powershell
[Environment]::SetEnvironmentVariable("TUSHARE_TOKEN", "你的TushareToken", "User")
[Environment]::SetEnvironmentVariable("REVIEW_API_KEY", "你的OpenAI兼容ApiKey", "User")
[Environment]::SetEnvironmentVariable("LC_OPENAI_API_KEY", "你的OpenAI兼容ApiKey(可选)", "User")
[Environment]::SetEnvironmentVariable("REVIEW_BASE_URL", "https://api.openai.com/v1", "User")  # 可选
[Environment]::SetEnvironmentVariable("REVIEW_MODEL", "gpt-4o-mini", "User")  # 可选
~~~

写入后请重开终端，环境变量才会在新会话中生效。

### 3.4 运行一键脚本

在项目根目录执行：

~~~bash
python run_all.py
~~~

常用参数：

~~~bash
python run_all.py --skip-fetch
python run_all.py --start-from 3
~~~

参数说明：

- `--skip-fetch`：跳过数据下载，直接进入初选；**仅在你已确认 `data/raw/*.csv` 已更新到最新交易日时使用**
- `--start-from N`：从第 N 步开始执行（1 到 4）

> 默认推荐直接运行 `python run_all.py`，先刷新最新 K 线，再做 preselect，避免 `pick_date` 卡在旧交易日。

### 3.5 重要说明：`run_all.py` 不负责写飞书表格
`run_all.py` 跑完后，只表示以下产物已经生成：
- `data/review/<date>/suggestion.json`
- 候选图
- 复评 JSON

如果你要形成“盘后完整闭环”，还需要继续执行：
1. 生成 Top10 payload：

~~~bash
python3 scripts/sync_top10_to_bitable.py --pick-date 2026-03-24 --top-n 10
~~~

2. 将 Top10 写入飞书 `选股明细`
3. 将汇总写入飞书 `每日汇总`
4. 对两张表进行回查确认

也就是说：
> `run_all.py` 完成 ≠ 飞书写表闭环完成

---

## 4. 分步运行攻略

### 步骤 1：拉取 K 线（当前主入口：TDX）

~~~bash
python pipeline/fetch_kline_tdx_api.py --all-a --limit 200 --concurrency 20
~~~

说明：
- 当前项目默认主抓取入口是 `pipeline/fetch_kline_tdx_api.py`
- 输出到 `data/raw/<code>.csv`
- 当前 raw 契约核心字段为：`date, open, high, low, close, volume`

> 旧的 `pipeline.fetch_kline` / Tushare 抓取逻辑不再作为当前主流程默认入口。

### 步骤 2：量化初选

~~~bash
python -m pipeline.cli preselect
python -m pipeline.cli b1-eval --code 300683 --date 2026-03-23
~~~

可选参数示例：

~~~bash
python -m pipeline.cli preselect --date 2026-03-13
python -m pipeline.cli preselect --config config/rules_preselect.yaml --data data/raw
~~~

规则配置见 [config/rules_preselect.yaml](config/rules_preselect.yaml)。

### 步骤 3：导出候选图表

~~~bash
python dashboard/export_kline_charts.py
~~~

输出到 data/kline/选股日期，图像命名为 代码_day.jpg。

### 步骤 4：OpenAI 兼容图表复评

~~~bash
python agent/model_review.py
~~~

可选参数示例：

~~~bash
python agent/model_review.py --config config/model_review.yaml
~~~

配置见 [config/model_review.yaml](config/model_review.yaml)。

读取候选与图表后，输出：

- data/review/日期/代码.json
- data/review/日期/suggestion.json

### B1 专用 AI 复评（仅复评 B1分 >= 80）

如果你要运行 B1 专用复评，而不是通用图表复评，可使用：

~~~bash
python scripts/run_b1_review_flow.py
~~~

可选：

~~~bash
python scripts/run_b1_review_flow.py --reviewer gemini
python scripts/run_b1_review_flow.py --skip-export
~~~

对应配置：
- `config/model_review_b1.yaml`
- `config/gemini_review_b1.yaml`

对应提示词：
- `agent/prompt_b1_review_v1.md`

当前 B1 专用复评口径：
- 只复评 `matched_strategies` 含 `b1` 且 `b1_score >= 80` 的候选
- 使用用户定义的“理想 B1”样本做 few-shot 锚定
- 定位为“解释器为主、轻筛选为辅”
- 输出到 `data/review_b1/日期/`

### 盘后总流程最小入口（项目侧）

如果你只想在项目侧串起“抓盘后数据 → preselect → B1 AI复评”，可使用：

~~~bash
python scripts/run_daily_postclose_selection.py
~~~

可选：

~~~bash
python scripts/run_daily_postclose_selection.py --skip-fetch
python scripts/run_daily_postclose_selection.py --skip-b1-review
python scripts/run_daily_postclose_selection.py --reviewer gemini
python scripts/run_daily_postclose_selection.py --fetch-limit 200 --fetch-concurrency 20
~~~

这个入口**只负责项目侧结果生成**，不会直连 Feishu 写表。

运行完成后会：
- 刷新 `data/candidates/candidates_latest.json`
- 刷新 `data/candidates/candidates_YYYY-MM-DD.json`
- 按需生成 `data/review_b1/YYYY-MM-DD/`
- 输出机器可读摘要 JSON
- 将摘要保存到：
  - `data/postclose/YYYY-MM-DD/summary.json`
  - `data/postclose/latest_summary.json`

摘要中至少包含：
- `pick_date`
- `raw_max_date`
- `preselect_done`
- `candidate_count`
- `b1_count`
- `b1_ai_review_done`
- `b1_ai_review_count`
- `final_ready_for_bitable_sync`

---

## 5. 关键配置建议

### 6.1 抓取层

- 首次全量抓取建议 workers 设小一些（如 4 到 8）
- 若遇到频率限制，降低并发并重试

### 6.2 初选层

- top_m 决定流动性股票池大小
- `b1.enabled` 控制当前正式 B1（原 B1-E）
- `b1_legacy.enabled` 控制 legacy / 确认版旧 B1，默认关闭
- brick.enabled 控制砖型图策略开关
- 可先只开一个策略做回放验证

#### 当前正式 B1 快速口径

- 低位 `J`：`J < 15` 或 `J <= 历史 10% 分位`
- 知行线方向：`zxdq > zxdkx`
- 最近 `5` 天最多允许 `1` 天 `close < zxdkx`，且该跌破必须是**缩量轻破**
- 排除最近 `20` 天明显放量派发与最近 `5` 天放量跌破 `zxdkx`

#### 当前正式 B1 量价定义

- 放量上涨日：`close > prev_close and close > open and volume > vol_ma5 * 1.3`
- 缩量下跌日：`close < prev_close and volume < vol_ma5 * 0.95 and ret > -0.03`

#### 当前正式 B1 评分重点

- `zx_retest`: 30
- `down_shrink`: 25
- `up_expand`: 25
- `volume_alignment`: 20
- `center_uplift`: 0

其中 `volume_alignment` 会额外观察：
- `up_expand_avg_vol / down_shrink_avg_vol`
- 该直接对比明显强于 `1.0` 时，上调一档；若弱于 `1.0`，下调一档

> 当前正式 B1 重心是“知行线不被有效破坏 + 量价转顺”，而不是价格重心已经抬高。

### 6.3 复评层

在 [config/model_review.yaml](config/model_review.yaml) 中可调整：

- model：模型名称
- base_url：OpenAI 兼容 API 地址（可选）
- api_style：responses 或 chat_completions
- request_delay：调用间隔（防限流）
- skip_existing：是否断点续跑
- suggest_min_score：推荐分数门槛

---

## 6. 输出结果解读

### 候选文件

[data/candidates/candidates_latest.json](data/candidates/candidates_latest.json)

- pick_date：选股日期
- candidates：候选列表（含 code、strategy、close 等）

### 复评汇总

data/review/日期/suggestion.json

- recommendations：最终推荐（按分数排序）
- excluded：未达门槛代码
- min_score_threshold：推荐门槛

### 正式 B1 后验复盘建议（简版）

当前正式 B1 建议按交易日窗口做后验观察：
- `T+1`：看是否继续有承接、是否立刻被放量压回
- `T+3`：看是否形成“涨放跌缩”与知行线未坏的基本节奏
- `T+5`：作为第一阶段正式收口点，重点记录涨跌幅、区间最高涨幅、区间最大回撤、是否有效跌破知行线、是否出现放量派发
- `T+10`：做补充观察，识别慢启动 / 延迟兑现样本

对正式 B1，复盘重点不是只看最终涨幅，而是先看：
- 知行线有没有被有效破坏
- 量价节奏有没有继续顺
- 回撤是否仍属于洗盘而不是结构失败

---

## 7. Docker / compose 运行说明

### 7.1 当前推荐运行方式
该项目当前通常与外层 `deploy/docker-compose.yml` 配合使用，典型容器命令为：

~~~bash
cd ../deploy

docker-compose exec -T stocktradebyz python pipeline/fetch_kline_tdx_api.py --all-a --limit 200 --concurrency 20

docker-compose exec -T stocktradebyz python -m pipeline.cli preselect

docker-compose exec -T stocktradebyz python dashboard/export_kline_charts.py

docker-compose exec -T stocktradebyz python run_all.py --start-from 4
~~~

### 7.2 运行单独步骤
示例：只跑复评

~~~bash
cd ../deploy
docker-compose exec -T stocktradebyz python agent/model_review.py
~~~

### 7.3 单股点评闭环产物生成
示例：

~~~bash
cd ../deploy
docker-compose exec -T stocktradebyz python /app/scripts/run_single_review_flow.py --code 600644 --pick-date 2026-03-24
~~~

### 7.4 环境变量
容器通过 `.env` 或环境变量读取配置，当前重点通常是：

- `REVIEW_API_KEY` 或 `LC_OPENAI_API_KEY`
- `REVIEW_BASE_URL`（可选）
- `REVIEW_MODEL`（可选）
- `REVIEW_API_STYLE`（responses | chat_completions，可选）
- `TDX_API_BASE_URL`（由外层 compose 注入）

---

## 8. 冒烟测试

用一张本地图像验证视觉调用与 JSON 解析：

~~~bash
python agent/smoke_test_vision.py --image path/to/your_image.jpg
~~~

---

## 9. 常见问题

### Q1：fetch_kline 报 token 错误

- 检查 TUSHARE_TOKEN 是否已设置
- 确认 token 有效且账号权限正常

### Q2：导出图表时报 write_image 错误

- 确认已安装 kaleido
- 重新安装：pip install -U kaleido

### Q3：模型复评运行失败

- 检查 REVIEW_API_KEY 或 LC_OPENAI_API_KEY 是否设置
- 如使用私有兼容接口，确认 REVIEW_BASE_URL 是否正确
- 观察是否命中限流，可提高 request_delay

### Q4：没有候选股票

- 检查 data/raw 是否有最新数据
- 放宽初选阈值（如 B1 或 Brick 参数）
- 检查 pick_date 是否在有效交易日

---

## License

本项目采用 [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) 协议发布。

- 允许：学习、研究、非商业用途的使用与分发
- 禁止：任何形式的商业使用、出售或以盈利为目的的部署
- 要求：转载或引用须注明原作者与来源

Copyright © 2026 SebastienZh. All rights reserved.
