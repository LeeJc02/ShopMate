"""向量检索器 - 使用 Chroma 进行向量检索"""

from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from .document_loader import load_knowledge_base
from .embeddings import get_embeddings


class KnowledgeRetriever:
    """知识库检索器"""
    
    def __init__(
        self,
        persist_directory: str | Path = "chroma_data",
        collection_name: str = "customer_service_kb",
    ):
        """
        初始化知识库检索器
        
        Args:
            persist_directory: Chroma 持久化目录
            collection_name: 集合名称
        """
        self.persist_directory = Path(persist_directory)
        self.collection_name = collection_name
        self.embeddings = get_embeddings()
        self._vectorstore: Chroma | None = None
    
    @property
    def vectorstore(self) -> Chroma:
        """获取向量存储实例"""
        if self._vectorstore is None:
            self._vectorstore = Chroma(
                collection_name=self.collection_name,
                embedding_function=self.embeddings,
                persist_directory=str(self.persist_directory),
            )
        return self._vectorstore
    
    def build_index(
        self,
        knowledge_dir: str | Path,
        chunk_size: int = 500,
        chunk_overlap: int = 100,
    ) -> int:
        """
        构建知识库索引
        
        Args:
            knowledge_dir: 知识库目录
            chunk_size: 文档块大小
            chunk_overlap: 块重叠大小
            
        Returns:
            索引的文档数量
        """
        # 加载并切分文档
        documents = load_knowledge_base(
            knowledge_dir,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        
        if not documents:
            return 0
        
        # 创建向量存储
        self._vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            collection_name=self.collection_name,
            persist_directory=str(self.persist_directory),
        )
        
        print(f"✅ 成功构建索引，共 {len(documents)} 个文档块")
        
        return len(documents)
    
    def search(
        self,
        query: str,
        k: int = 3,
        filter_dict: dict | None = None,
    ) -> list[Document]:
        """
        检索相关文档
        
        Args:
            query: 查询文本
            k: 返回的文档数量
            filter_dict: 过滤条件
            
        Returns:
            相关文档列表
        """
        if filter_dict:
            docs = self.vectorstore.similarity_search(
                query,
                k=k,
                filter=filter_dict,
            )
        else:
            docs = self.vectorstore.similarity_search(query, k=k)
        
        return docs
    
    def search_with_score(
        self,
        query: str,
        k: int = 3,
    ) -> list[tuple[Document, float]]:
        """
        检索相关文档并返回相似度分数
        
        Args:
            query: 查询文本
            k: 返回的文档数量
            
        Returns:
            (文档, 分数) 元组列表
        """
        return self.vectorstore.similarity_search_with_score(query, k=k)
    
    def get_retriever(self, k: int = 3):
        """
        获取 LangChain Retriever 接口
        
        Args:
            k: 返回的文档数量
            
        Returns:
            VectorStoreRetriever
        """
        return self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k},
        )
