"""
Demo-RAG-Pipeline (Prüfobjekt).

Aufbau: LangChain + ChromaDB + HuggingFace Embeddings + Claude als Generator.
Diese Pipeline dient als zu testender RAG-Agent im Framework.
"""

from pathlib import Path
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_anthropic import ChatAnthropic
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate

import config


SYSTEM_PROMPT = """Du bist ein hilfreicher Assistent der Syntax Systems GmbH.
Beantworte Fragen ausschließlich auf Basis der bereitgestellten Dokumente.
Wenn die Information nicht in den Dokumenten enthalten ist, sage das klar.
Erfinde keine Informationen und weiche nicht von den Dokumenten ab.

Kontext:
{context}

Frage: {question}

Antwort:"""


def build_rag_pipeline(
    docs_dir: Path | None = None,
    system_prompt: str = SYSTEM_PROMPT,
) -> RetrievalQA:
    """
    Erstellt eine RAG-Pipeline aus einem Dokumentenverzeichnis.

    Args:
        docs_dir: Pfad zum Dokumentenverzeichnis. Standard: sample_docs/
        system_prompt: Prompt-Template. Muss {context} und {question} enthalten.

    Returns:
        LangChain RetrievalQA-Chain
    """
    if docs_dir is None:
        docs_dir = config.SAMPLE_DOCS_DIR

    # 1. Dokumente laden
    loader = DirectoryLoader(
        str(docs_dir),
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
    )
    raw_docs = loader.load()

    # 2. Chunking
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
    )
    docs = splitter.split_documents(raw_docs)

    # 3. Embeddings (lokal, kein API-Key nötig)
    embeddings = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL)

    # 4. Vektordatenbank
    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=str(config.CHROMA_DIR),
    )

    # 5. LLM (Claude als Generator)
    llm = ChatAnthropic(
        model=config.GENERATOR_MODEL,
        api_key=config.ANTHROPIC_API_KEY,
        max_tokens=1024,
    )

    # 6. Chain zusammenbauen
    prompt = PromptTemplate(
        template=system_prompt,
        input_variables=["context", "question"],
    )
    retriever = vectorstore.as_retriever(
        search_kwargs={"k": config.RETRIEVER_K}
    )
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
    )

    return chain


def load_existing_vectorstore() -> Chroma:
    """Lädt eine bereits erstellte ChromaDB (ohne Dokumente neu zu laden)."""
    embeddings = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL)
    return Chroma(
        persist_directory=str(config.CHROMA_DIR),
        embedding_function=embeddings,
    )

