#!/usr/bin/env python3
"""知识库初始化脚本 - 加载 Markdown 文件并构建向量索引"""

import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.rag import KnowledgeRetriever


def main():
    """初始化知识库"""
    print("=" * 50)
    print("📚 知识库初始化脚本")
    print("=" * 50)
    
    # 知识库目录
    knowledge_dir = project_root / "data" / "knowledge"
    
    if not knowledge_dir.exists():
        print(f"❌ 错误: 知识库目录不存在: {knowledge_dir}")
        sys.exit(1)
    
    # 列出所有 Markdown 文件
    md_files = list(knowledge_dir.glob("**/*.md"))
    print(f"\n📁 发现 {len(md_files)} 个 Markdown 文件:")
    for md_file in md_files:
        print(f"   - {md_file.name}")
    
    # 创建检索器
    persist_dir = project_root / "chroma_data"
    retriever = KnowledgeRetriever(
        persist_directory=persist_dir,
        collection_name="customer_service_kb",
    )
    
    print(f"\n🔄 正在构建向量索引...")
    print(f"   持久化目录: {persist_dir}")
    
    # 构建索引
    doc_count = retriever.build_index(
        knowledge_dir=knowledge_dir,
        chunk_size=500,
        chunk_overlap=100,
    )
    
    if doc_count > 0:
        print(f"\n✅ 索引构建完成!")
        print(f"   文档块数量: {doc_count}")
        
        # 测试检索
        print("\n🔍 测试检索...")
        test_queries = [
            "iPhone 15 Pro 多少钱？",
            "退货政策是什么？",
            "快递几天能到？",
        ]
        
        for query in test_queries:
            print(f"\n   查询: {query}")
            results = retriever.search(query, k=1)
            if results:
                source = results[0].metadata.get("filename", "未知")
                content_preview = results[0].page_content[:100].replace("\n", " ")
                print(f"   来源: {source}")
                print(f"   内容: {content_preview}...")
            else:
                print("   未找到相关内容")
        
        print("\n" + "=" * 50)
        print("🎉 知识库初始化完成!")
        print("=" * 50)
    else:
        print("\n❌ 错误: 没有成功索引任何文档")
        sys.exit(1)


if __name__ == "__main__":
    main()
