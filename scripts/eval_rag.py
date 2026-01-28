#!/usr/bin/env python3
"""
RAG 召回率评测脚本

评测指标：
- 来源召回率 (Source Recall): 预期文档是否出现在召回结果中
- 关键词命中率 (Keyword Hit Rate): 召回内容是否包含预期关键词
- 平均召回数量: 每次查询召回的文档块数
"""

import json
import sys
from pathlib import Path
from datetime import datetime

# 添加项目根目录
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.rag import KnowledgeRetriever


def load_eval_dataset(dataset_path: Path) -> dict:
    """加载评测数据集"""
    with open(dataset_path, "r", encoding="utf-8") as f:
        return json.load(f)


def evaluate_rag(retriever: KnowledgeRetriever, dataset: dict, k: int = 3) -> dict:
    """
    执行 RAG 召回率评测
    
    Args:
        retriever: 知识库检索器
        dataset: 评测数据集
        k: 召回文档数
        
    Returns:
        评测结果
    """
    results = {
        "total_cases": 0,
        "source_recall_hits": 0,
        "keyword_total": 0,
        "keyword_hits": 0,
        "avg_docs_retrieved": 0,
        "details": [],
    }
    
    total_docs = 0
    
    for case in dataset["test_cases"]:
        query = case["query"]
        expected_sources = case["expected_sources"]
        expected_keywords = case["expected_keywords"]
        
        # 执行检索
        docs = retriever.search(query, k=k)
        
        # 获取召回的来源
        retrieved_sources = [doc.metadata.get("filename", "") for doc in docs]
        retrieved_content = " ".join([doc.page_content for doc in docs])
        
        # 计算来源召回
        source_hit = any(src in retrieved_sources for src in expected_sources)
        
        # 计算关键词命中
        keyword_hits = sum(1 for kw in expected_keywords if kw in retrieved_content)
        
        results["total_cases"] += 1
        if source_hit:
            results["source_recall_hits"] += 1
        
        results["keyword_total"] += len(expected_keywords)
        results["keyword_hits"] += keyword_hits
        
        total_docs += len(docs)
        
        # 详细结果
        results["details"].append({
            "id": case["id"],
            "query": query,
            "category": case.get("category", ""),
            "source_recall": source_hit,
            "keyword_hit_rate": keyword_hits / len(expected_keywords) if expected_keywords else 0,
            "retrieved_sources": retrieved_sources,
        })
    
    # 计算汇总指标
    results["source_recall_rate"] = results["source_recall_hits"] / results["total_cases"]
    results["keyword_hit_rate"] = results["keyword_hits"] / results["keyword_total"] if results["keyword_total"] > 0 else 0
    results["avg_docs_retrieved"] = total_docs / results["total_cases"]
    
    return results


def print_results(results: dict):
    """打印评测结果"""
    print("\n" + "=" * 60)
    print("📊 RAG 召回率评测结果")
    print("=" * 60)
    
    print(f"\n📈 汇总指标:")
    print(f"   测试用例数: {results['total_cases']}")
    print(f"   来源召回率: {results['source_recall_rate']:.1%} ({results['source_recall_hits']}/{results['total_cases']})")
    print(f"   关键词命中率: {results['keyword_hit_rate']:.1%} ({results['keyword_hits']}/{results['keyword_total']})")
    print(f"   平均召回文档数: {results['avg_docs_retrieved']:.1f}")
    
    print(f"\n📋 详细结果:")
    print("-" * 60)
    
    for detail in results["details"]:
        status = "✅" if detail["source_recall"] else "❌"
        kw_rate = f"{detail['keyword_hit_rate']:.0%}"
        print(f"   {status} [{detail['id']}] {detail['query'][:30]}...")
        print(f"      分类: {detail['category']} | 关键词命中: {kw_rate}")
        print(f"      召回来源: {', '.join(detail['retrieved_sources'])}")
    
    print("\n" + "=" * 60)
    
    # 评分建议
    recall_rate = results["source_recall_rate"]
    if recall_rate >= 0.9:
        print("🎉 优秀！召回率超过 90%")
    elif recall_rate >= 0.7:
        print("👍 良好！建议优化文档切分策略")
    else:
        print("⚠️ 需要优化！建议检查索引构建和查询策略")


def save_results(results: dict, output_path: Path):
    """保存评测结果"""
    results["timestamp"] = datetime.now().isoformat()
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n💾 结果已保存至: {output_path}")


def main():
    print("🔍 开始 RAG 召回率评测...")
    
    # 加载数据集
    dataset_path = project_root / "data" / "eval" / "rag_eval_dataset.json"
    dataset = load_eval_dataset(dataset_path)
    print(f"📚 加载评测数据集: {len(dataset['test_cases'])} 个测试用例")
    
    # 初始化检索器
    retriever = KnowledgeRetriever(
        persist_directory=project_root / "chroma_data",
        collection_name="customer_service_kb",
    )
    
    # 执行评测
    results = evaluate_rag(retriever, dataset, k=3)
    
    # 打印结果
    print_results(results)
    
    # 保存结果
    output_path = project_root / "data" / "eval" / "rag_eval_results.json"
    save_results(results, output_path)


if __name__ == "__main__":
    main()
