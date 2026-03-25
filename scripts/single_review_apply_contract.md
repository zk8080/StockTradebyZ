# Single Review Apply Contract

`scripts/apply_single_review_upsert.py` 输出的是“真实飞书写表前的一跳”。

约定：

- `operation=create` 且 `executable=true`
  - 映射到 `feishu_bitable_app_table_record.create`
  - 使用 `tool_name` 与 `tool_arguments`
- `operation=update` 且 `executable=true`
  - 映射到 `feishu_bitable_app_table_record.update`
  - 使用 `tool_name` 与 `tool_arguments`
- `operation=conflict` 或 `executable=false`
  - 不调用任何写表工具
  - 由主流程根据 `reason` / `conflicts` 进入人工处理

`tool_arguments` 结构：

- create: `app_token`, `table_id`, `fields`
- update: `app_token`, `table_id`, `record_id`, `fields`

如果当前阶段还没有真实飞书目标，可只保留 `table_target`，后续由主流程把它解析成 `app_token` + `table_id`。
