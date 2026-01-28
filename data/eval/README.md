# 评测数据集

本目录包含用于评测客服系统性能的数据集。

## 数据集说明

### rag_eval_dataset.json
RAG 召回率评测数据集，包含：
- `query`: 用户查询
- `expected_sources`: 预期召回的文档来源
- `expected_keywords`: 预期包含的关键词

### routing_eval_dataset.json
Agent 路由准确率评测数据集，包含：
- `query`: 用户查询
- `expected_agent`: 预期路由到的 Agent

## 使用方法

```bash
# 运行 RAG 召回率评测
python scripts/eval_rag.py

# 运行路由准确率评测
python scripts/eval_routing.py

# 运行性能压测
python scripts/benchmark.py
```
