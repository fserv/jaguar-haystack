import json, logging
from typing import Any, List, Dict, Optional
from haystack import default_to_dict, default_from_dict
from haystack.dataclasses import Document

logger = logging.getLogger(__name__)

class JaguarEmbeddingRetriever:
    """Jaguar dense embedding retriever.

    See http://www.jaguardb.com
    See https://github.com/fserv/jaguar-sdk

    Example:
       .. code-block:: python

           docstore = JaguarEmbeddingRetriever(
               pod = 'vdb',
               store = 'mystore',
               vector_index = 'v',
               vector_type = 'cosine_fraction_float',
               vector_dimension = 1536,
               url='http://192.168.8.88:8080/fwww/',
           )
    """
    def __init__(
        self,
        pod: str,
        store: str,
        vector_index: str,
        vector_type: str,
        vector_dimension: int,
        url: str,
        **kwargs: Any,
    ):
        """Constructor of JaguarEmbeddingRetriever.

        Args:
            pod: str:  name of the pod (database)
            store: str:  name of vector store in the pod
            vector_index: str:  name of vector index of the store
            vector_type: str:  type of the vector index
            vector_dimension: int:  dimension of the vector index
            url: str:  URL end point of jaguar http server
        """
        self._pod = pod
        self._store = store
        self._vector_index = vector_index
        self._vector_type = vector_type
        self._vector_dimension = vector_dimension
        self._url = url
        self._kwargs = kwargs

        try:
            from jaguardb_http_client.JaguarHttpClient import JaguarHttpClient
        except ImportError:
            logger.error("E0021 error import JaguarHttpClient")
            raise ValueError(
                "Could not import jaguardb-http-client python package. "
                "Please install it with `pip install -U jaguardb-http-client`"
            )

        self._jag = JaguarHttpClient(url)
        self._token = ""

    def __del__(self) -> None:
        pass

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "JaguarEmbeddingRetriever":
        return default_from_dict(cls, data)

    @classmethod
    def class_name(cls) -> str:
        return "JaguarEmbeddingRetriever"

    def to_dict(self) -> Dict[str, Any]:
        return default_to_dict(
            self,
            pod = self._pod,
            store = self._store,
            vector_index = self._vector_index,
            vector_type = self._vector_type,
            vector_dimension = self._vector_dimension,
            url = self._url,
            **self._kwargs,
        )

    def count_documents(self) -> int:
        """Count documents of a store in jaguardb.

        Args: no args
        Returns: (int) number of records in pod store
        """
        podstore = self._pod + "." + self._store
        q = "select count() from " + podstore
        js = self.runQuery(q)
        if isinstance(js, list) and len(js) == 0:
            return 0
        jd = json.loads(js[0])
        return int(jd["data"])

    def run(
        self,
        embedding: List[float],
        top_k: Optional[int] = 3,
        **kwargs: Any
    ) -> Dist[str, List[Document]]:
        """Query index to load top k most similar documents.

        Args:
            embedding: a list of floats
            top_k: topK number
            kwargs:  may contain 'where', 'metadata_fields', 'args', 'fetch_k'
        """
        docs = self.similarity_search_with_score(embedding, k=top_k, **kwargs)
        return {"documents": docs}

    def similarity_search_with_score(
        self,
        embedding: Optional[List[float]],
        k: int = 3,
        **kwargs: Any,
    ) -> List[Document]:
        """Return nodes most similar to query embedding, along with ids and scores.

        Args:
            embedding: embedding of text to look up.
            k: Number of nodes to return. Defaults to 3.
            kwargs: may have where, metadata_fields, args, fetch_k
        Returns:
            list of documents
        """
        where = kwargs.get("where", None)
        metadata_fields = kwargs.get("metadata_fields", None)

        args = kwargs.get("args", None)
        fetch_k = kwargs.get("fetch_k", -1)

        vcol = self._vector_index
        vtype = self._vector_type
        if embedding is None:
            return []
        str_embeddings = [str(f) for f in embedding]
        qv_comma = ",".join(str_embeddings)
        podstore = self._pod + "." + self._store
        q = (
            "select similarity("
            + vcol
            + ",'"
            + qv_comma
            + "','topk="
            + str(k)
            + ",fetch_k="
            + str(fetch_k)
            + ",type="
            + vtype
        )
        q += ",with_score=yes,with_text=yes"
        if args is not None:
            q += "," + args

        if metadata_fields is not None:
            x = "&".join(metadata_fields)
            q += ",metadata=" + x

        q += "') from " + podstore

        if where is not None:
            q += " where " + where

        jarr = self.run(q)

        if jarr is None:
            return []

        nodes = []
        ids = []
        simscores = []
        docs = []
        for js in jarr:
            score = js["score"]
            text = js["text"]
            zid = js["zid"]

            md = {}
            md["zid"] = zid
            if metadata_fields is not None:
                for m in metadata_fields:
                    mv = js[m]
                    md[m] = mv

            doc = Document(
                id=zid,
                content=text,
                meta=md,
                score = score,
            )
            docs.append(doc)

        return docs

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
        vcol = self._vector_index
        vtype = self._vector_type
        str_embeddings = [str(f) for f in doc.embedding]
        qv_comma = ",".join(str_embeddings)
        podstore = self._pod + "." + self._store
        q = "select anomalous(" + vcol + ", '" + qv_comma + "', 'type=" + vtype + "')"
        q += " from " + podstore

        js = self.run(q)
        if isinstance(js, list) and len(js) == 0:
            return False
        jd = json.loads(js[0])
        if jd["anomalous"] == "YES":
            return True
        return False

    def runQuery(self, query: str, withFile: bool = False) -> dict:
        """Run any query statement in jaguardb.

        Args:
            query (str): query statement to jaguardb
        Returns:
            None for invalid token, or
            json result string
        """
        if self._token == "":
            logger.error(f"E0005 error runQuery({query})")
            return {}

        resp = self._jag.post(query, self._token, withFile)
        txt = resp.text
        try:
            return json.loads(txt)
        except Exception:
            return {}

    def login(
        self,
        jaguar_api_key: Optional[str] = "",
    ) -> bool:
        """Login to jaguar server with a jaguar_api_key or let self._jag find a key.

        Args:
            optional jaguar_api_key (str): API key of user to jaguardb server
        Returns:
            True if successful; False if not successful
        """
        if jaguar_api_key == "":
            jaguar_api_key = self._jag.getApiKey()
        self._jaguar_api_key = jaguar_api_key
        self._token = self._jag.login(jaguar_api_key)
        if self._token == "":
            logger.error("E0001 error init(): invalid jaguar_api_key")
            return False
        return True

    def logout(self) -> None:
        """Logout to cleanup resources.

        Args: no args
        Returns: None
        """
        self._jag.logout(self._token)
