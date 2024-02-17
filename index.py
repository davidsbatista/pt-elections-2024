from datasets import load_dataset
from haystack import Document, Pipeline
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack.components.readers import ExtractiveReader
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
from haystack.components.writers import DocumentWriter
from haystack.document_stores.in_memory import InMemoryDocumentStore

model = "sentence-transformers/multi-qa-mpnet-base-dot-v1"


def index():
    dataset = load_dataset("bilgeyucel/seven-wonders", split="train")
    documents = [Document(content=doc["content"], meta=doc["meta"]) for doc in dataset]
    document_store = InMemoryDocumentStore()
    indexing_pipeline = Pipeline()
    indexing_pipeline.add_component(instance=SentenceTransformersDocumentEmbedder(model=model), name="embedder")
    indexing_pipeline.add_component(instance=DocumentWriter(document_store=document_store), name="writer")
    indexing_pipeline.connect("embedder.documents", "writer.documents")

    print("Indexing documents...")
    indexing_pipeline.run({"documents": documents})
    print(document_store.count_documents())
    print(document_store.to_dict())

    return document_store


def extractive_retriever(document_store):
    retriever = InMemoryEmbeddingRetriever(document_store=document_store)
    reader = ExtractiveReader()
    reader.warm_up()
    extractive_qa_pipeline = Pipeline()
    extractive_qa_pipeline.add_component(instance=SentenceTransformersTextEmbedder(model=model), name="embedder")
    extractive_qa_pipeline.add_component(instance=retriever, name="retriever")
    extractive_qa_pipeline.add_component(instance=reader, name="reader")
    extractive_qa_pipeline.connect("embedder.embedding", "retriever.query_embedding")
    extractive_qa_pipeline.connect("retriever.documents", "reader.documents")
    return extractive_qa_pipeline


def main():
    document_store = index()
    extractive_qa_pipeline = extractive_retriever(document_store)
    query = "Who was Pliny the Elder?"
    extractive_qa_pipeline.run(data={"embedder": {"text": query}, "retriever": {"top_k": 3}, "reader": {"query": query, "top_k": 2}})


if __name__ == '__main__':
    main()