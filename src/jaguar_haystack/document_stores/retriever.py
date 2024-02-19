import json, logging
from typing import Any, List, Dict, Optional
from haystack import Document, component, default_from_dict, default_to_dict
from jaguar_haystack.jaguar import JaguarDocumentStore

logger = logging.getLogger(__name__)

@component
class JaguarEmbeddingRetriever:
    """Jaguar dense embedding retriever.

    See http://www.jaguardb.com
    See https://github.com/fserv/jaguar-sdk

    Example:
       .. code-block:: python

           docstore = JaguarEmbeddingRetriever(
               document_store = jaguar_document_store,
               top_k = 3
           )
    """
    def __init__(
        self,
        document_store: JaguarDocumentStore,
        filters: Optional[Dict[str, Any]] = None,
        top_k: int = 5,
    ):
        """
        Create the JaguarEmbeddingRetriever component.

        :param document_store: An instance of JaguarDocumentStore.
        :param filters: Filters applied to the retrieved Documents. Defaults to None.
        :param top_k: Maximum number of Documents to return, defaults to 5.

        :raises ValueError: If `document_store` is not an instance of JaguarDocumentStore.
        """

        if not isinstance(document_store, JaguarDocumentStore):
            msg = "document_store must be an instance of JaguarDocumentStore "
            raise ValueError(msg)

        self._document_store = document_store
        self._filters = filters
        self._top_k = top_k

    @classmethod
    def class_name(cls) -> str:
        return "JaguarEmbeddingRetriever"

    def to_dict(self) -> Dict[str, Any]:
        return default_to_dict(
            self,
            filters=self._filters,
            top_k=self._top_k,
            document_store=self._document_store.to_dict(),
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "JaguarEmbeddingRetriever":
        return default_from_dict(cls, data)

    @component.output_types(documents=List[Document])
    def run(self, query_embedding: List[float], **kwargs ):
        """
        Retrieve documents from the JaguarDocumentStore, based on their dense embeddings.

        :param query_embedding: Embedding of the query.
        :param kwargs:  may contain 'where', 'metadata_fields', 'args', 'fetch_k'
        :return: List of Document similar to `query_embedding`.
        """
        if self._filters is not None:
            where = self._document_store._convert_to_where(self._filters)
            kwargs['where'] = where

        docs = self._document_store.search_similar_documents(embedding=query_embedding, k=self._top_k, **kwargs)
        return {"documents": docs}

    def count_all_documents(self) -> int:
        """Count documents of a store in jaguardb.

        Args: no args
        Returns: (int) number of records in pod store
        """
        return self._document_store.count_documents()

    def is_anomalous(
        self,
        doc: Document,
        **kwargs: Any,
    ) -> bool:
        """Detect if given document is anomalous from the dataset.

        Args:
            query: Text to detect if it is anomaly
        Returns:
            True or False
        """
        return self._document_store.is_anomalous( doc, **kwargs )
