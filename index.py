from pathlib import Path

from datasets import load_dataset
from haystack import Document, Pipeline
from haystack.components.converters import PyPDFToDocument
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack.components.preprocessors import DocumentCleaner, DocumentSplitter
from haystack.components.readers import ExtractiveReader
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
from haystack.components.writers import DocumentWriter
from haystack.document_stores.in_memory import InMemoryDocumentStore

model = "sentence-transformers/multi-qa-mpnet-base-dot-v1"


def index_programs():

    # create an in-memory document store
    document_store = InMemoryDocumentStore()

    # create a pipeline with the components
    pipeline = Pipeline()
    pipeline.add_component("converter", PyPDFToDocument())
    pipeline.add_component("cleaner", DocumentCleaner())
    pipeline.add_component("splitter", DocumentSplitter(split_by="sentence", split_length=10))
    pipeline.add_component("embedder", SentenceTransformersDocumentEmbedder(model=model))
    pipeline.add_component("writer", DocumentWriter(document_store=document_store, policy="skip"))

    # connect the components
    pipeline.connect("converter", "cleaner")
    pipeline.connect("cleaner", "splitter")
    pipeline.connect("splitter", "embedder")
    pipeline.connect("embedder", "writer")

    # create a list of paths to the PDFs, by expanding the glob pattern
    files = list(Path("programas/2024").glob("*.pdf"))
    pipeline.run({"converter": {"sources": files}})

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