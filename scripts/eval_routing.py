#!/usr/bin/env python3
"""
Agent 路由准确率评测脚本

评测指标：
- 路由准确率 (Routing Accuracy): Agent 分配是否正确
- 各 Agent 准确率: 按 Agent 维度的准确率
- 各类别准确率: 按意图类别的准确率
"""

import json
import sys
from pathlib import Path
from datetime import datetime
from collections import defaultdict

# 添加项目根目录
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.graphs.customer_service_graph import SupervisorAgent


def load_eval_dataset(dataset_path: Path) -> dict:
    """加载评测数据集"""
    with open(dataset_path, "r", encoding="utf-8") as f:
        return json.load(f)


def evaluate_routing(supervisor: SupervisorAgent, dataset: dict) -> dict:
    """
    执行 Agent 路由准确率评测
    
    Args:
        supervisor: Supervisor Agent
        dataset: 评测数据集
        
    Returns:
        评测结果
    """
    results = {
        "total_cases": 0,
        "correct": 0,
        "by_agent": defaultdict(lambda: {"total": 0, "correct": 0}),
        "by_category": defaultdict(lambda: {"total": 0, "correct": 0}),
        "details": [],
        "confusion_matrix": defaultdict(lambda: defaultdict(int)),
    }
    
    for case in dataset["test_cases"]:
        query = case["query"]
        expected_agent = case["expected_agent"]
        category = case.get("category", "未分类")
        
        # 执行路由
        try:
            predicted_agent = supervisor.route(query)
        except Exception as e:
            predicted_agent = f"ERROR: {e}"
        
        # 判断是否正确
        is_correct = predicted_agent == expected_agent
        
        results["total_cases"] += 1
        if is_correct:
            results["correct"] += 1
        
        # 按 Agent 统计
        results["by_agent"][expected_agent]["total"] += 1
        if is_correct:
            results["by_agent"][expected_agent]["correct"] += 1
        
        # 按类别统计
        results["by_category"][category]["total"] += 1
        if is_correct:
            results["by_category"][category]["correct"] += 1
        
        # 混淆矩阵
        results["confusion_matrix"][expected_agent][predicted_agent] += 1
        
        # 详细结果
        results["details"].append({
            "id": case["id"],
            "query": query,
            "category": category,
            "expected": expected_agent,
            "predicted": predicted_agent,
            "correct": is_correct,
        })
    
    # 计算汇总指标
    results["accuracy"] = results["correct"] / results["total_cases"]
    
    # 转换 defaultdict 为普通 dict
    results["by_agent"] = dict(results["by_agent"])
    results["by_category"] = dict(results["by_category"])
    results["confusion_matrix"] = {k: dict(v) for k, v in results["confusion_matrix"].items()}
    
    # 计算各 Agent 准确率
    for agent, stats in results["by_agent"].items():
        stats["accuracy"] = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
    
    # 计算各类别准确率
    for category, stats in results["by_category"].items():
        stats["accuracy"] = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
    
    return results


def print_results(results: dict):
    """打印评测结果"""
    print("\n" + "=" * 60)
    print("📊 Agent 路由准确率评测结果")
    print("=" * 60)
    
    print(f"\n📈 汇总指标:")
    print(f"   测试用例数: {results['total_cases']}")
    print(f"   路由准确率: {results['accuracy']:.1%} ({results['correct']}/{results['total_cases']})")
    
    print(f"\n📋 按 Agent 准确率:")
    print("-" * 40)
    for agent, stats in sorted(results["by_agent"].items()):
        print(f"   {agent}: {stats['accuracy']:.1%} ({stats['correct']}/{stats['total']})")
    
    print(f"\n📋 按类别准确率:")
    print("-" * 40)
    for category, stats in sorted(results["by_category"].items()):
        status = "✅" if stats["accuracy"] >= 0.8 else "⚠️" if stats["accuracy"] >= 0.5 else "❌"
        print(f"   {status} {category}: {stats['accuracy']:.1%} ({stats['correct']}/{stats['total']})")
    
    print(f"\n📋 错误详情:")
    print("-" * 60)
    errors = [d for d in results["details"] if not d["correct"]]
    if errors:
        for detail in errors:
            print(f"   ❌ [{detail['id']}] \"{detail['query']}\"")
            print(f"      预期: {detail['expected']} | 实际: {detail['predicted']}")
    else:
        print("   🎉 没有错误！")
    
    print("\n" + "=" * 60)
    
    # 评分建议
    accuracy = results["accuracy"]
    if accuracy >= 0.9:
        print("🎉 优秀！路由准确率超过 90%")
    elif accuracy >= 0.7:
        print("👍 良好！建议优化 Supervisor Prompt")
    else:
        print("⚠️ 需要优化！建议调整意图分类策略")


def save_results(results: dict, output_path: Path):
    """保存评测结果"""
    results["timestamp"] = datetime.now().isoformat()
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n💾 结果已保存至: {output_path}")


def main():
    print("🔍 开始 Agent 路由准确率评测...")
    
    # 加载数据集
    dataset_path = project_root / "data" / "eval" / "routing_eval_dataset.json"
    dataset = load_eval_dataset(dataset_path)
    print(f"📚 加载评测数据集: {len(dataset['test_cases'])} 个测试用例")
    
    # 初始化 Supervisor
    supervisor = SupervisorAgent()
    
    # 执行评测
    print("⏳ 正在评测（需要调用 LLM，请耐心等待）...")
    results = evaluate_routing(supervisor, dataset)
    
    # 打印结果
    print_results(results)
    
    # 保存结果
    output_path = project_root / "data" / "eval" / "routing_eval_results.json"
    save_results(results, output_path)


if __name__ == "__main__":
    main()
