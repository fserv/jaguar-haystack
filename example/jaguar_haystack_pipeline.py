#from haystack.utils import fetch_archive_from_http
from haystack import Pipeline
from haystack.nodes import TextConverter, PreProcessor
import json, os, time
from haystack.dataclasses import Document
from jaguar_haystack.jaguar import JaguarDocumentStore
import logging
from haystack.nodes import BM25Retriever
from haystack.nodes import FARMReader
from pprint import pprint
from haystack.utils import print_answers
from haystack.telemetry import tutorial_running
from haystack.telemetry import send_event

logging.basicConfig(filename='/tmp/haystack.log', level=logging.DEBUG, format='%(asctime)s %(levelname)s:%(message)s')
logger = logging.getLogger(__name__)


tutorial_running(3)

### get text
doc_dir = "data"

### make writer pipeline
indexing_pipeline = Pipeline()
text_converter = TextConverter()
preprocessor = PreProcessor(
    clean_whitespace=True,
    clean_header_footer=True,
    clean_empty_lines=True,
    split_by="word",
    split_length=200,
    split_overlap=20,
    split_respect_sentence_boundary=True,
)

url = "http://127.0.0.1:8080/fwww/"
pod = "vdb"
store = "haystack_test_store"
vector_index = "v"
vector_type = "cosine_fraction_float"
vector_dimension = 1536
document_store = JaguarDocumentStore(
    pod,
    store,
    vector_index,
    vector_type,
    vector_dimension,
    url,
)

# any meta data fields to be created and included
metadata_fields = "author char(64), category char(16)"
# size of each document text to be saved
text_size = 1024

document_store.create(metadata_fields, text_size)
document_store.login( "demouser" )


indexing_pipeline.add_node(component=text_converter, name="TextConverter", inputs=["File"])
indexing_pipeline.add_node(component=preprocessor, name="PreProcessor", inputs=["TextConverter"])
indexing_pipeline.add_node(component=document_store, name="DocumentStore", inputs=["PreProcessor"])
files_to_index = [doc_dir + "/" + f for f in os.listdir(doc_dir)]
indexing_pipeline.run_batch(file_paths=files_to_index)


####### ask questions 
retriever = BM25Retriever(document_store=document_store)
reader = FARMReader(model_name_or_path="deepset/roberta-base-squad2", use_gpu=False)
querying_pipeline = Pipeline()
querying_pipeline.add_node(component=retriever, name="Retriever", inputs=["Query"])
querying_pipeline.add_node(component=reader, name="Reader", inputs=["Retriever"])

question = "Is jaguardb scalable?"
print(f"Question: {question}")

prediction = querying_pipeline.run(
    query=question, params={"Retriever": {"top_k": 10}, "Reader": {"top_k": 5}}
)

print("Answer:");
pprint(prediction)

print("Minimum Answer:");
print_answers(prediction, details="minimum")  ## Choose from `minimum`, `medium` and `all`

document_store.logout()
