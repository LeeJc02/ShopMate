"""文档加载器 - 使用 LangChain Document Loader 加载 Markdown 文件"""

from pathlib import Path

from langchain_community.document_loaders import DirectoryLoader, UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


def load_markdown_files(directory: str | Path) -> list[Document]:
    """
    加载目录下所有 Markdown 文件
    
    Args:
        directory: Markdown 文件所在目录
        
    Returns:
        Document 列表
    """
    directory = Path(directory)
    
    if not directory.exists():
        raise FileNotFoundError(f"目录不存在: {directory}")
    
    # 使用 DirectoryLoader 加载所有 .md 文件
    loader = DirectoryLoader(
        str(directory),
        glob="**/*.md",
        loader_cls=UnstructuredMarkdownLoader,
        show_progress=True,
        use_multithreading=True,
    )
    
    documents = loader.load()
    
    # 为每个文档添加元数据
    for doc in documents:
        # 提取文件名作为来源
        source_path = Path(doc.metadata.get("source", ""))
        doc.metadata["filename"] = source_path.name
        doc.metadata["category"] = _infer_category(source_path.name)
    
    return documents


def _infer_category(filename: str) -> str:
    """根据文件名推断文档类别"""
    category_map = {
        "product": "商品信息",
        "catalog": "商品信息",
        "after_sales": "售后政策",
        "policy": "售后政策",
        "delivery": "物流配送",
        "shipping": "物流配送",
        "promotion": "促销活动",
        "discount": "促销活动",
    }
    
    filename_lower = filename.lower()
    for key, category in category_map.items():
        if key in filename_lower:
            return category
    
    return "其他"


def split_documents(
    documents: list[Document],
    chunk_size: int = 500,
    chunk_overlap: int = 100,
) -> list[Document]:
    """
    将文档切分为小块
    
    Args:
        documents: 原始文档列表
        chunk_size: 每块的最大字符数
        chunk_overlap: 块之间的重叠字符数
        
    Returns:
        切分后的文档列表
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=[
            "\n## ",      # 二级标题
            "\n### ",     # 三级标题
            "\n---",      # 分隔线
            "\n\n",       # 空行
            "\n",         # 换行
            "。",         # 中文句号
            "！",         # 中文感叹号
            "？",         # 中文问号
            "；",         # 中文分号
            " ",          # 空格
        ],
    )
    
    split_docs = text_splitter.split_documents(documents)
    
    return split_docs


def load_knowledge_base(
    directory: str | Path,
    chunk_size: int = 500,
    chunk_overlap: int = 100,
) -> list[Document]:
    """
    加载并处理知识库
    
    Args:
        directory: 知识库目录
        chunk_size: 文档块大小
        chunk_overlap: 块重叠大小
        
    Returns:
        处理后的文档列表
    """
    # 加载 Markdown 文件
    documents = load_markdown_files(directory)
    
    if not documents:
        print(f"警告: 目录 {directory} 中没有找到 Markdown 文件")
        return []
    
    print(f"✅ 加载了 {len(documents)} 个 Markdown 文件")
    
    # 切分文档
    split_docs = split_documents(
        documents,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    
    print(f"✅ 切分为 {len(split_docs)} 个文档块")
    
    return split_docs
