#!/usr/bin/env python3
"""
性能压测脚本

评测指标：
- QPS (Queries Per Second): 每秒处理请求数
- Latency P50/P95/P99: 响应时间分位数
- 成功率: 请求成功比例
- 吞吐量: 单位时间处理的总请求数
"""

import json
import time
import asyncio
import statistics
from pathlib import Path
from datetime import datetime
from typing import NamedTuple
from concurrent.futures import ThreadPoolExecutor

import aiohttp


class BenchmarkConfig(NamedTuple):
    """压测配置"""
    base_url: str = "http://localhost:8000"
    total_requests: int = 100
    concurrency: int = 10
    timeout: float = 30.0


class RequestResult(NamedTuple):
    """请求结果"""
    success: bool
    latency: float  # 秒
    status_code: int
    error: str | None = None


# 测试查询列表
TEST_QUERIES = [
    "你好",
    "iPhone 15 Pro 多少钱？",
    "查一下我的订单",
    "我想退货",
    "华为 Mate 60 有什么颜色？",
    "快递到哪了？",
    "会员折扣是多少？",
    "怎么申请换货？",
    "MacBook Pro 配置怎么样？",
    "有什么优惠活动？",
]


async def send_request(session: aiohttp.ClientSession, config: BenchmarkConfig, query: str) -> RequestResult:
    """发送单个请求"""
    start_time = time.perf_counter()
    
    try:
        async with session.post(
            f"{config.base_url}/chat",
            json={"message": query, "use_tools": False},
            timeout=aiohttp.ClientTimeout(total=config.timeout),
        ) as response:
            await response.json()
            latency = time.perf_counter() - start_time
            
            return RequestResult(
                success=response.status == 200,
                latency=latency,
                status_code=response.status,
            )
    except asyncio.TimeoutError:
        latency = time.perf_counter() - start_time
        return RequestResult(
            success=False,
            latency=latency,
            status_code=0,
            error="Timeout",
        )
    except Exception as e:
        latency = time.perf_counter() - start_time
        return RequestResult(
            success=False,
            latency=latency,
            status_code=0,
            error=str(e),
        )


async def run_benchmark(config: BenchmarkConfig) -> dict:
    """执行压测"""
    print(f"\n🚀 开始压测...")
    print(f"   目标: {config.base_url}")
    print(f"   总请求数: {config.total_requests}")
    print(f"   并发数: {config.concurrency}")
    
    results: list[RequestResult] = []
    
    connector = aiohttp.TCPConnector(limit=config.concurrency)
    async with aiohttp.ClientSession(connector=connector) as session:
        # 创建任务
        tasks = []
        for i in range(config.total_requests):
            query = TEST_QUERIES[i % len(TEST_QUERIES)]
            tasks.append(send_request(session, config, query))
        
        # 执行并收集结果
        start_time = time.perf_counter()
        results = await asyncio.gather(*tasks)
        total_time = time.perf_counter() - start_time
    
    # 计算指标
    latencies = [r.latency for r in results]
    success_count = sum(1 for r in results if r.success)
    
    metrics = {
        "config": {
            "base_url": config.base_url,
            "total_requests": config.total_requests,
            "concurrency": config.concurrency,
        },
        "summary": {
            "total_time": round(total_time, 2),
            "success_count": success_count,
            "failure_count": config.total_requests - success_count,
            "success_rate": round(success_count / config.total_requests, 4),
            "qps": round(config.total_requests / total_time, 2),
            "throughput": round(success_count / total_time, 2),
        },
        "latency": {
            "min": round(min(latencies), 3),
            "max": round(max(latencies), 3),
            "avg": round(statistics.mean(latencies), 3),
            "p50": round(statistics.median(latencies), 3),
            "p95": round(sorted(latencies)[int(len(latencies) * 0.95)], 3),
            "p99": round(sorted(latencies)[int(len(latencies) * 0.99)], 3),
        },
    }
    
    # 错误统计
    errors = [r.error for r in results if r.error]
    if errors:
        from collections import Counter
        metrics["errors"] = dict(Counter(errors))
    
    return metrics


def print_results(metrics: dict):
    """打印压测结果"""
    print("\n" + "=" * 60)
    print("📊 性能压测结果")
    print("=" * 60)
    
    summary = metrics["summary"]
    latency = metrics["latency"]
    
    print(f"\n📈 汇总指标:")
    print(f"   总耗时: {summary['total_time']}s")
    print(f"   成功/失败: {summary['success_count']}/{summary['failure_count']}")
    print(f"   成功率: {summary['success_rate']:.1%}")
    print(f"   QPS: {summary['qps']}")
    print(f"   吞吐量: {summary['throughput']} req/s")
    
    print(f"\n⏱️ 延迟分布 (秒):")
    print(f"   最小: {latency['min']}")
    print(f"   最大: {latency['max']}")
    print(f"   平均: {latency['avg']}")
    print(f"   P50:  {latency['p50']}")
    print(f"   P95:  {latency['p95']}")
    print(f"   P99:  {latency['p99']}")
    
    if "errors" in metrics:
        print(f"\n❌ 错误统计:")
        for error, count in metrics["errors"].items():
            print(f"   {error}: {count}")
    
    print("\n" + "=" * 60)
    
    # 性能评估
    qps = summary["qps"]
    p95 = latency["p95"]
    
    if qps >= 10 and p95 < 3:
        print("🎉 优秀！QPS >= 10, P95 < 3s")
    elif qps >= 5 and p95 < 5:
        print("👍 良好！建议优化响应时间")
    else:
        print("⚠️ 需要优化！并发性能较低")


def save_results(metrics: dict, output_path: Path):
    """保存压测结果"""
    metrics["timestamp"] = datetime.now().isoformat()
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    
    print(f"\n💾 结果已保存至: {output_path}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="性能压测脚本")
    parser.add_argument("--url", default="http://localhost:8000", help="目标 URL")
    parser.add_argument("--requests", "-n", type=int, default=50, help="总请求数")
    parser.add_argument("--concurrency", "-c", type=int, default=5, help="并发数")
    parser.add_argument("--timeout", "-t", type=float, default=30.0, help="超时时间")
    args = parser.parse_args()
    
    config = BenchmarkConfig(
        base_url=args.url,
        total_requests=args.requests,
        concurrency=args.concurrency,
        timeout=args.timeout,
    )
    
    # 执行压测
    metrics = asyncio.run(run_benchmark(config))
    
    # 打印结果
    print_results(metrics)
    
    # 保存结果
    project_root = Path(__file__).parent.parent
    output_path = project_root / "data" / "eval" / "benchmark_results.json"
    save_results(metrics, output_path)


if __name__ == "__main__":
    main()
