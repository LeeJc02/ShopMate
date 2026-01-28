"""
A/B 测试框架

支持对不同模型、Prompt、参数进行 A/B 测试：
1. 实验分流：基于 session_id 稳定分配
2. 指标收集：响应时间、用户反馈
3. 结果分析：各组对比

使用方式：
```python
from src.core.ab_testing import ABTestManager, Experiment

# 定义实验
manager = ABTestManager()
manager.create_experiment(
    name="prompt_v2",
    variants={"control": 0.5, "treatment": 0.5},
)

# 获取分组
variant = manager.get_variant("prompt_v2", session_id)
if variant == "treatment":
    # 使用新版 prompt
    ...
else:
    # 使用原版 prompt
    ...

# 记录结果
manager.record_result("prompt_v2", session_id, {"latency": 1.5, "rating": 5})
```
"""

import hashlib
import time
from typing import Any
from dataclasses import dataclass, field
from collections import defaultdict
import logging
import json
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class Experiment:
    """实验定义"""
    name: str
    variants: dict[str, float]  # variant_name -> traffic_weight (0-1)
    enabled: bool = True
    created_at: float = field(default_factory=time.time)
    description: str = ""


@dataclass
class ExperimentResult:
    """实验结果"""
    experiment_name: str
    variant: str
    session_id: str
    metrics: dict[str, Any]
    timestamp: float = field(default_factory=time.time)


class ABTestManager:
    """
    A/B 测试管理器
    
    管理多个实验，分配流量，收集结果
    """
    
    def __init__(self, results_dir: Path | None = None):
        """
        初始化
        
        Args:
            results_dir: 结果保存目录
        """
        self.experiments: dict[str, Experiment] = {}
        self.results: list[ExperimentResult] = []
        self.results_dir = results_dir
        
        # 预定义的实验
        self._init_default_experiments()
    
    def _init_default_experiments(self):
        """初始化默认实验"""
        # 模型对比实验
        self.create_experiment(
            name="llm_provider",
            variants={"dashscope": 1.0, "openai": 0.0},  # 默认 100% dashscope
            description="主备模型对比",
        )
        
        # Prompt 版本实验
        self.create_experiment(
            name="prompt_version",
            variants={"v1": 1.0, "v2": 0.0},
            description="Prompt 版本对比",
        )
        
        # 缓存开关实验
        self.create_experiment(
            name="cache_enabled",
            variants={"enabled": 1.0, "disabled": 0.0},
            description="响应缓存开关",
        )
    
    def create_experiment(
        self,
        name: str,
        variants: dict[str, float],
        description: str = "",
    ) -> Experiment:
        """
        创建实验
        
        Args:
            name: 实验名称
            variants: 变体及其流量权重 {"control": 0.5, "treatment": 0.5}
            description: 实验描述
            
        Returns:
            实验对象
        """
        # 验证权重和为 1
        total_weight = sum(variants.values())
        if abs(total_weight - 1.0) > 0.01:
            raise ValueError(f"变体权重之和必须为 1，当前为 {total_weight}")
        
        experiment = Experiment(
            name=name,
            variants=variants,
            description=description,
        )
        self.experiments[name] = experiment
        logger.info(f"创建实验: {name} -> {variants}")
        return experiment
    
    def get_variant(self, experiment_name: str, session_id: str) -> str:
        """
        获取 session 的实验分组
        
        使用 consistent hashing 确保同一 session 总是分配到同一组
        
        Args:
            experiment_name: 实验名称
            session_id: 会话 ID
            
        Returns:
            变体名称
        """
        experiment = self.experiments.get(experiment_name)
        if not experiment or not experiment.enabled:
            # 实验不存在或未启用，返回第一个变体
            if experiment:
                return list(experiment.variants.keys())[0]
            return "default"
        
        # 基于 session_id 和实验名称计算 hash
        hash_input = f"{experiment_name}:{session_id}"
        hash_value = int(hashlib.md5(hash_input.encode()).hexdigest(), 16)
        bucket = (hash_value % 10000) / 10000  # 0-1 之间的值
        
        # 根据权重分配变体
        cumulative = 0.0
        for variant, weight in experiment.variants.items():
            cumulative += weight
            if bucket < cumulative:
                return variant
        
        # 兜底返回最后一个变体
        return list(experiment.variants.keys())[-1]
    
    def record_result(
        self,
        experiment_name: str,
        session_id: str,
        metrics: dict[str, Any],
    ):
        """
        记录实验结果
        
        Args:
            experiment_name: 实验名称
            session_id: 会话 ID
            metrics: 指标数据 {"latency": 1.5, "rating": 5}
        """
        variant = self.get_variant(experiment_name, session_id)
        
        result = ExperimentResult(
            experiment_name=experiment_name,
            variant=variant,
            session_id=session_id,
            metrics=metrics,
        )
        self.results.append(result)
        
        logger.debug(f"记录结果: {experiment_name}/{variant} -> {metrics}")
    
    def get_experiment_stats(self, experiment_name: str) -> dict:
        """
        获取实验统计
        
        Args:
            experiment_name: 实验名称
            
        Returns:
            各变体的统计数据
        """
        experiment = self.experiments.get(experiment_name)
        if not experiment:
            return {"error": "实验不存在"}
        
        # 按变体分组统计
        variant_results: dict[str, list] = defaultdict(list)
        for result in self.results:
            if result.experiment_name == experiment_name:
                variant_results[result.variant].append(result.metrics)
        
        stats = {
            "experiment": experiment_name,
            "description": experiment.description,
            "enabled": experiment.enabled,
            "variants": {},
        }
        
        for variant, results in variant_results.items():
            if not results:
                continue
            
            # 计算常见指标的统计值
            variant_stats = {
                "count": len(results),
            }
            
            # 对数值型指标计算均值
            for key in results[0].keys():
                values = [r[key] for r in results if isinstance(r.get(key), (int, float))]
                if values:
                    variant_stats[f"{key}_avg"] = round(sum(values) / len(values), 3)
                    variant_stats[f"{key}_min"] = min(values)
                    variant_stats[f"{key}_max"] = max(values)
            
            stats["variants"][variant] = variant_stats
        
        return stats
    
    def update_traffic(self, experiment_name: str, variants: dict[str, float]):
        """
        更新实验流量分配
        
        Args:
            experiment_name: 实验名称
            variants: 新的变体权重
        """
        if experiment_name not in self.experiments:
            raise ValueError(f"实验 '{experiment_name}' 不存在")
        
        total_weight = sum(variants.values())
        if abs(total_weight - 1.0) > 0.01:
            raise ValueError(f"变体权重之和必须为 1，当前为 {total_weight}")
        
        self.experiments[experiment_name].variants = variants
        logger.info(f"更新实验流量: {experiment_name} -> {variants}")
    
    def enable_experiment(self, experiment_name: str, enabled: bool = True):
        """启用/禁用实验"""
        if experiment_name in self.experiments:
            self.experiments[experiment_name].enabled = enabled
            logger.info(f"实验 {experiment_name} {'启用' if enabled else '禁用'}")
    
    def export_results(self, output_path: Path):
        """导出实验结果到文件"""
        data = {
            "experiments": {
                name: {
                    "variants": exp.variants,
                    "enabled": exp.enabled,
                    "description": exp.description,
                }
                for name, exp in self.experiments.items()
            },
            "results": [
                {
                    "experiment": r.experiment_name,
                    "variant": r.variant,
                    "session_id": r.session_id,
                    "metrics": r.metrics,
                    "timestamp": r.timestamp,
                }
                for r in self.results
            ],
        }
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"实验结果已导出: {output_path}")
    
    def get_all_experiments(self) -> dict:
        """获取所有实验配置"""
        return {
            name: {
                "variants": exp.variants,
                "enabled": exp.enabled,
                "description": exp.description,
            }
            for name, exp in self.experiments.items()
        }


# ===== 全局实例 =====

_ab_manager: ABTestManager | None = None


def get_ab_manager() -> ABTestManager:
    """获取全局 A/B 测试管理器"""
    global _ab_manager
    if _ab_manager is None:
        _ab_manager = ABTestManager()
    return _ab_manager
