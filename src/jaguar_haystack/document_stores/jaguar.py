import json, logging
from typing import Any, List, Dict, Optional, Tuple
from haystack import default_to_dict, default_from_dict
from haystack.dataclasses import Document
from haystack.document_stores.protocol import DuplicatePolicy

logger = logging.getLogger(__name__)

class JaguarDocumentStore:
    """Jaguar document store.

    See http://www.jaguardb.com
    See https://github.com/fserv/jaguar-sdk

    Example:
       .. code-block:: python

           docstore = JaguarDocumentStore(
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
        """Constructor of JaguarDocumentStore.

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
            logger.error("E0001 error import JaguarHttpClient")
            raise ValueError(
                "Could not import jaguardb-http-client python package. "
                "Please install it with `pip install -U jaguardb-http-client`"
            )

        self._jag = JaguarHttpClient(url)
        self._token = ""

    def __del__(self) -> None:
        pass

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "JaguarDocumentStore":
        return default_from_dict(cls, data)

    @classmethod
    def class_name(cls) -> str:
        return "JaguarDocumentStore"

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
        js = self.run(q)
        if isinstance(js, list) and len(js) == 0:
            return 0
        jd = json.loads(js[0])
        return int(jd["data"])

    def write_documents(self, documents: List[Document], 
                                policy: DuplicatePolicy = DuplicatePolicy.NONE,
                                **kwargs: Any) -> int:
        """Add documents to store.

        Args:
            nodes: List[Document]: list of nodes with embeddings
            policy: DuplicatePolicy: how to handle duplicates
        """
        cnt = 0
        for doc in documents:
            text = doc.content
            embedding = doc.embedding
            metadata = doc.meta
            if doc.dataframe is not None:
                for idx, row in doc.dataframe.iterraows():
                    for c in doc.dataframe.columns:
                        metadata[c] = row[c]
                    break
                
            zid = self.add_text(text, embedding, metadata, **kwargs)
            if zid != "":
                cnt += 1

        return cnt

    def delete_documents(self, document_ids: List[str]) -> None:
        """
        Delete documents with a list of id strings

        Args:
            document_ids:  List[str]
        """
        podstore = self._pod + "." + self._store
        for zid in document_ids:
            q = "delete from " + podstore + " where zid='" + zid + "'"
            self.run(q)

    def search_similar_documents(
        self, embedding: List[float], k: int, **kwargs: Any
    ) -> List[Document]:
        """Query index to load top k most similar documents.

        Args:
            embedding: a list of floats
            k: topK number
            kwargs:  may contain 'where', 'metadata_fields', 'args', 'fetch_k'
        """
        return self.similarity_search_with_score(embedding, k=k, **kwargs)

    def filter_documents(self, 
        filters: Optional[Dict[str, Any]] = None,
        k:  int = 1000,
        **kwargs: Any
    ) -> List[Document]:
        """Return nodes most similar to query embedding, along with ids and scores.

        Args:
            filters: Dict[str, Any]
                as described in haystack.DocumentStore: filter_documents()
            k: int  limit of records returned, default is 1000
            kwargs: may have any options for extensibility
        Returns:
            List of Documents (read meta in each Document for field values)
        """
        where = ""
        try:
            where = self._convert_to_where(filters)
        except ValueError as e:
            logger.error(f"E0008 _convert_to_where error {e}")
            return []

        if where == "":
            return []

        podstore = self._pod + "." + self._store
        q = "select * from " + podstore
        q += " where " + where
        q += " limit " + str(k)

        jarr = self.run(q)

        if jarr is None:
            return []

        docs = []
        for js in jarr:
            pdict = json.loads(js)
            doc = Document(
                id=pdict["zid"],
                meta=pdict
            )
            docs.append(doc)

        return docs


    def create(
        self,
        metadata_fields: str,
        text_size: int,
    ) -> None:
        """
        create the document store on the backend database.

        Args:
            metadata_fields (str):  exrta metadata columns and types
        Returns:
            True if successful; False if not successful
        """
        podstore = self._pod + "." + self._store

        """
        v:text column is required.
        """
        q = "create store "
        q += podstore
        q += f" ({self._vector_index} vector({self._vector_dimension},"
        q += f" '{self._vector_type}'),"
        q += f"  v:text char({text_size}),"
        q += metadata_fields + ")"
        self.run(q)

    def add_text(
        self,
        text: str,
        embedding: List[float],
        metadata: Optional[dict] = None,
        **kwargs: Any,
    ) -> str:
        """
        Add  texts through the embeddings and add to the vectorstore.

        Args:
          texts: text string to add to the jaguar vector store.
          embedding: embedding vector of the text, list of floats
          metadata: dict of metadata
          kwargs: vector_index=name_of_vector_index
                  file_column=name_of_file_column
                  metadata={...}

        Returns:
            id from adding the text into the vectorstore
        """
        text = text.replace("'", "\\'")
        vcol = self._vector_index
        filecol = kwargs.get("file_column", "")

        podstorevcol = self._pod + "." + self._store + "." + vcol
        q = "textcol " + podstorevcol
        js = self.run(q)
        if js == "":
            return ""
        textcol = js["data"]

        zid = ""
        if metadata is None:
            ### no metadata and no files to upload
            str_vec = [str(x) for x in embedding]
            values_comma = ",".join(str_vec)
            podstore = self._pod + "." + self._store
            q = "insert into " + podstore + " ("
            q += vcol + "," + textcol + ") values ('" + values_comma
            q += "','" + text + "')"
            js = self.run(q, False)
            zid = js["zid"]
        else:
            str_vec = [str(x) for x in embedding]
            nvec, vvec, filepath = self._parseMeta(metadata, filecol)
            if filecol != "":
                rc = self._jag.postFile(self._token, filepath, 1)
                if not rc:
                    return ""
            names_comma = ",".join(nvec)
            names_comma += "," + vcol
            ## col1,col2,col3,vecl

            if vvec is not None and len(vvec) > 0:
                values_comma = "'" + "','".join(vvec) + "'"
            else:
                values_comma = "'" + "','".join(vvec) + "'"

            ### 'va1','val2','val3'
            values_comma += ",'" + ",".join(str_vec) + "'"
            ### 'v1,v2,v3'
            podstore = self._pod + "." + self._store
            q = "insert into " + podstore + " ("
            q += names_comma + "," + textcol + ") values (" + values_comma
            q += ",'" + text + "')"
            if filecol != "":
                js = self.run(q, True)
            else:
                js = self.run(q, False)
            zid = js["zid"]

        return zid

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
            Tuple(list of nodes, list of ids, list of similaity scores)
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

    def run(self, query: str, withFile: bool = False) -> dict:
        """Run any query statement in jaguardb.

        Args:
            query (str): query statement to jaguardb
        Returns:
            None for invalid token, or
            json result string
        """
        if self._token == "":
            logger.error(f"E0005 error run({query})")
            return {}

        resp = self._jag.post(query, self._token, withFile)
        txt = resp.text
        try:
            return json.loads(txt)
        except Exception:
            return {}

    def clear(self) -> None:
        """Delete all records in jaguardb.

        Args: No args
        Returns: None
        """
        podstore = self._pod + "." + self._store
        q = "truncate store " + podstore
        self.run(q)

    def drop(self) -> None:
        """Drop or remove a store in jaguardb.

        Args: no args
        Returns: None
        """
        podstore = self._pod + "." + self._store
        q = "drop store " + podstore
        self.run(q)

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

    def _parseMeta(self, nvmap: dict, filecol: str) -> Tuple[List[str], List[str], str]:
        filepath = ""
        if filecol == "":
            nvec = list(nvmap.keys())
            vvec = list(nvmap.values())
        else:
            nvec = []
            vvec = []
            if filecol in nvmap:
                nvec.append(filecol)
                vvec.append(nvmap[filecol])
                filepath = nvmap[filecol]

            for k, v in nvmap.items():
                if k != filecol:
                    nvec.append(k)
                    vvec.append(v)

        return nvec, vvec, filepath

    def _convert_to_where(self, filter_object, is_nested=False):
        def format_value(value):
            if isinstance(value, list):
                # Apply formatting to each element in the list
                return ", ".join([f"'{v}'" for v in value])
            else:
                return f"'{value}'"
    
        if 'field' in filter_object:
            # Comparison filter
            field = filter_object['field']
            operator = filter_object['operator']
            value = format_value(filter_object['value'])
    
            # For SQL, 'in' and 'not in' operators need special handling
            if operator in ['in', 'not in']:
                return f"{field} {operator} ({value})"
            else:
                # Adjusting the operator for SQL syntax
                sql_operator = '=' if operator == '==' else operator
                return f"{field} {sql_operator} {value}"
    
        elif 'operator' in filter_object:
            # Logic filter
            operator = filter_object['operator']
            conditions = filter_object['conditions']
    
            # Recursively convert each condition and join with the operator
            condition_strings = [self._convert_to_where(cond, is_nested=True) for cond in conditions]
            joined_conditions = f" {operator} ".join(condition_strings)
    
            # Add parentheses only if it's a nested condition
            if is_nested and len(conditions) > 1:
                return f"({joined_conditions})"
            else:
                return joined_conditions
    
        else:
            raise ValueError("Invalid filter object")

