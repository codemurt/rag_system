import os
import torch
from pathlib import Path
from typing import List, Dict, Any, Union, Optional, Tuple, Generator
import json
import logging
import time
from dotenv import load_dotenv

load_dotenv()

import gradio as gr
from langchain_docling import DoclingLoader
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.schema import Document
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, T5ForConditionalGeneration, T5Tokenizer
from langchain_community.llms import HuggingFacePipeline
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from docling.datamodel import vlm_model_specs
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import VlmPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.pipeline.vlm_pipeline import VlmPipeline

log_level = os.getenv('LOG_LEVEL', 'INFO')
log_file = os.getenv('LOG_FILE', '/app/logs/rag_system.log')

os.makedirs(os.path.dirname(log_file), exist_ok=True)

logging.basicConfig(
    level=getattr(logging, log_level),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_file, encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

os.environ["TOKENIZERS_PARALLELISM"] = os.getenv("TOKENIZERS_PARALLELISM", "false")

def clean_metadata(metadata: Dict[str, Any]) -> Dict[str, Union[str, int, float, bool, None]]:
    """
    Очищает метаданные от сложных структур для совместимости с Chroma.

    Args:
        metadata: Исходные метаданные

    Returns:
        Очищенные метаданные
    """
    cleaned = {}

    for key, value in metadata.items():
        if isinstance(value, (str, int, float, bool, type(None))):
            cleaned[key] = value
        elif isinstance(value, dict):
            if key == 'dl_meta' and isinstance(value, dict):
                if 'doc_items' in value and isinstance(value['doc_items'], list):
                    for item in value['doc_items']:
                        if isinstance(item, dict) and 'prov' in item:
                            prov_list = item.get('prov', [])
                            if prov_list and isinstance(prov_list[0], dict):
                                page_no = prov_list[0].get('page_no')
                                if page_no:
                                    cleaned['page_number'] = page_no
                                    break

                if 'headings' in value and isinstance(value['headings'], list):
                    if value['headings']:
                        cleaned['section'] = str(value['headings'][0])

                if 'origin' in value and isinstance(value['origin'], dict):
                    filename = value['origin'].get('filename')
                    if filename:
                        cleaned['source'] = filename
            else:
                cleaned[key] = str(value)
        elif isinstance(value, list):
            if value and all(isinstance(item, (str, int, float)) for item in value):
                cleaned[key] = ', '.join(str(item) for item in value)
            else:
                cleaned[key] = str(value)
        else:
            cleaned[key] = str(value)

    return cleaned

class RAG:
    """
    Универсальная RAG система для работы с технической документацией.
    Поддерживает русский язык, OCR и различные форматы документов.
    """
    def __init__(
        self,
        model_id: str = None,
        embed_model_id: str = None,
        chunk_size: int = None,
        chunk_overlap: int = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        llm: Optional[Any] = None,
        embeddings: Optional[Any] = None
    ):
        self.logger = logging.getLogger(self.__class__.__name__)
        
        
        self.model_id = model_id or os.getenv('MODEL_ID', "unsloth/Llama-3.2-1B-Instruct")
        self.embed_model_id = embed_model_id or os.getenv('EMBED_MODEL_ID', "sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
        self.chunk_size = chunk_size or int(os.getenv('CHUNK_SIZE', 512))
        self.chunk_overlap = chunk_overlap or int(os.getenv('CHUNK_OVERLAP', 50))
        self.device = device

        if llm:
            self.llm = llm
            self.logger.info("Используется предзагруженная LLM")
        else:
            self._init_llm()

        if embeddings:
            self.embeddings = embeddings
            self.logger.info("Используются предзагруженные эмбеддинги")
        else:
            self._init_embeddings()

        self.vectorstore = None
        self.retriever = None

    def _init_llm(self):
        """Инициализация языковой модели."""
        self.logger.info(f"Загрузка модели {self.model_id}...")

        tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None,
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )

        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=512,
            temperature=0.1,
            top_p=0.95,
            repetition_penalty=1.15,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            truncation=True,
            max_length=2048
        )

        self.llm = HuggingFacePipeline(pipeline=pipe)
        self.logger.info("Модель загружена успешно")

    def _init_embeddings(self):
        """Инициализация модели эмбеддингов."""
        self.logger.info(f"Загрузка модели эмбеддингов {self.embed_model_id}...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.embed_model_id,
            model_kwargs={'device': self.device}
        )
        self.logger.info("Модель эмбеддингов загружена успешно!")

    def load_document(self, file_path: str | List[str]) -> List[Any]:
        """
        Загрузка и обработка документов с использованием Docling.

        Args:
            file_path: Путь к файлу или список путей

        Returns:
            Список документов LangChain
        """
        if isinstance(file_path, str):
            file_path = [file_path]

        self.logger.info(f"Загрузка документов с VLM: {file_path}")

        try:
            converter = DocumentConverter(
                format_options={
                    InputFormat.PDF: PdfFormatOption(
                        pipeline_cls=VlmPipeline,
                    ),
                }
            )

            all_docs = []

            for path in file_path:
                self.logger.info(f"Обработка файла: {path}")

                result = converter.convert(source=path)
                doc = result.document

                markdown_content = doc.export_to_markdown()

                metadata = {
                    'source': path,
                    'filename': Path(path).name,
                    'format': 'pdf',
                    'converter': 'VLM Docling'
                }

                if hasattr(doc, 'metadata') and doc.metadata:
                    for key, value in doc.metadata.items():
                        if isinstance(value, (str, int, float, bool)):
                            metadata[key] = value

                sections = markdown_content.split('\n\n')

                for i, section in enumerate(sections):
                    if section.strip():
                        section_metadata = metadata.copy()
                        section_metadata['section_index'] = i

                        lines = section.strip().split('\n')
                        if lines and lines[0].startswith('#'):
                            section_metadata['section'] = lines[0].strip('#').strip()

                        langchain_doc = Document(
                            page_content=section.strip(),
                            metadata=section_metadata
                        )
                        all_docs.append(langchain_doc)

            self.logger.info(f"Загружено {len(all_docs)} секций из VLM Docling")

            cleaned_docs = []
            for doc in all_docs:
                cleaned_metadata = clean_metadata(doc.metadata)
                cleaned_doc = Document(
                    page_content=doc.page_content,
                    metadata=cleaned_metadata
                )
                cleaned_docs.append(cleaned_doc)

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                length_function=len,
                separators=["\n\n", "\n", ".", " ", ""]
            )

            chunked_docs = []
            for doc in cleaned_docs:
                if len(doc.page_content) < 50:
                    continue

                chunks = text_splitter.split_text(doc.page_content)
                for i, chunk in enumerate(chunks):
                    chunk_metadata = doc.metadata.copy()
                    chunk_metadata['chunk_index'] = i
                    chunk_metadata['total_chunks'] = len(chunks)

                    chunked_docs.append(Document(
                        page_content=chunk,
                        metadata=chunk_metadata
                    ))

            self.logger.info(f"Создано {len(chunked_docs)} чанков документов")
            return chunked_docs

        except Exception as e:
            self.logger.error(f"Ошибка при загрузке через VLM Docling: {e}")
            self.logger.warning("Используем альтернативный загрузчик...")

            all_docs = []
            for path in file_path:
                try:
                    loader = PyPDFLoader(path)
                    docs = loader.load()

                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=self.chunk_size,
                        chunk_overlap=self.chunk_overlap
                    )

                    split_docs = text_splitter.split_documents(docs)
                    all_docs.extend(split_docs)

                except Exception as e2:
                    self.logger.error(f"Ошибка при загрузке {path}: {e2}")

            self.logger.info(f"Загружено {len(all_docs)} чанков через PyPDF")
            return all_docs

    def create_index(self, documents: List[Any], collection_name: str = "universal_rag"):
        """
        Создание векторного индекса из документов.

        Args:
            documents: Список документов LangChain
            collection_name: Название коллекции в векторной БД
        """
        self.logger.info("Создание векторного индекса...")

        filtered_docs = filter_complex_metadata(documents)

        persist_directory = os.getenv('CHROMA_PERSIST_DIRECTORY', './chroma_db')
        
        self.vectorstore = Chroma.from_documents(
            documents=filtered_docs,
            embedding=self.embeddings,
            collection_name=collection_name,
            persist_directory=persist_directory
        )

        self.retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}
        )

        self.logger.info("Индекс создан успешно!")

    def _create_prompt_template(self) -> PromptTemplate:
        """Создание шаблона промпта для RAG."""
        template = """Ты - помощник, отвечающий на вопросы строго на основе предоставленного контекста.

        Контекст из документации:
        {context}

        Вопрос пользователя: {question}

        Инструкции:
        1. Отвечай ТОЛЬКО на основе информации из контекста
        2. Если информация отсутствует в контексте, честно скажи "Информация не найдена в документе"
        3. Если вопрос не относится к теме документа, скажи "Вопрос не относится к содержанию документа"
        4. Указывай источники информации (номера страниц, если доступны)
        5. Отвечай на том же языке, на котором задан вопрос

        Ответ:"""

        return PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )

    def answer_question(self, question: str, return_sources: bool = True) -> Dict[str, Any]:
        """
        Ответ на вопрос с использованием RAG.

        Args:
            question: Вопрос пользователя
            return_sources: Возвращать ли источники

        Returns:
            Словарь с ответом и метаданными
        """
        if not self.retriever:
            raise ValueError("Индекс не создан. Сначала загрузите документы и создайте индекс.")

        relevant_docs = self.retriever.get_relevant_documents(question)

        if not relevant_docs:
            return {
                "answer": "Информация не найдена в документе",
                "sources": [],
                "relevant_chunks": []
            }

        prompt = self._create_prompt_template()

        context_parts = []
        sources = []

        for i, doc in enumerate(relevant_docs):
            context_parts.append(f"[Фрагмент {i+1}]:\n{doc.page_content}")

            metadata = doc.metadata
            source_info = {
                "chunk_id": i + 1,
                "content": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
            }

            for page_key in ['page_number', 'page', 'page_no']:
                if page_key in metadata:
                    source_info['page_no'] = metadata[page_key]
                    break

            if 'section' in metadata:
                source_info['section'] = metadata['section']

            sources.append(source_info)

        context = "\n\n".join(context_parts)

        try:
            formatted_prompt = prompt.format(context=context, question=question)
            answer = self.llm.invoke(formatted_prompt)

            if isinstance(answer, str):
                answer = answer.strip()
                if "Ответ:" in answer:
                    answer = answer.split("Ответ:")[-1].strip()

        except Exception as e:
            self.logger.error(f"Ошибка при генерации ответа: {e}")
            answer = "Произошла ошибка при генерации ответа"

        result = {
            "answer": answer,
            "question": question,
            "sources": sources if return_sources else [],
            "relevant_chunks": [doc.page_content for doc in relevant_docs] if return_sources else []
        }

        return result

    def process_questions(self, questions: List[str]) -> List[Dict[str, Any]]:
        """
        Обработка списка вопросов.

        Args:
            questions: Список вопросов

        Returns:
            Список ответов с метаданными
        """
        results = []

        for i, question in enumerate(questions, 1):
            self.logger.info(f"Обработка вопроса {i}/{len(questions)}: {question}")
            result = self.answer_question(question)
            results.append(result)

            answer_preview = result['answer'][:200] + "..." if len(result['answer']) > 200 else result['answer']
            self.logger.info(f"Ответ: {answer_preview}")

        return results


logger.info("Предзагрузка моделей")
global_llm = None
global_embeddings = None

def preload_models():
    """Предзагружает модели для ускорения работы"""
    global global_llm, global_embeddings

    logger.info("Предзагрузка LLM...")
    temp_rag = RAG()
    global_llm = temp_rag.llm
    global_embeddings = temp_rag.embeddings
    logger.info("Модели предзагружены!")

preload_models()

def process_uploaded_file_and_question(file_path: str, question: str) -> str:
    """Обрабатывает загруженный файл и вопрос"""
    try:
        rag_system = RAG(
            llm=global_llm,
            embeddings=global_embeddings
        )

        logger.info(f"Загрузка документа: {file_path}")
        documents = rag_system.load_document(file_path)

        logger.info("Создание векторного индекса...")
        rag_system.create_index(documents)

        logger.info(f"Обработка вопроса: {question}")
        result = rag_system.answer_question(question)

        response = result['answer']
        if result['sources']:
            response += "\n\n Источники:\n"
            for source in result['sources']:
                page_info = f" (страница {source['page_no']})" if 'page_no' in source else ""
                response += f"- {source['content']}{page_info}\n"

        return response

    except Exception as e:
        logger.exception(f"Ошибка обработки: {e}")
        return f"Критическая ошибка: {str(e)}"

with gr.Blocks(title="RAG для технической документации", theme=gr.themes.Soft()) as demo:
    gr.Markdown("## RAG система для технической документации")
    gr.Markdown("Загрузите PDF-документ и задайте вопрос по его содержанию")

    with gr.Row():
        with gr.Column(scale=3):
            file_input = gr.File(
                label="Загрузите PDF-документ",
                type="filepath",
                file_types=[".pdf"]
            )
        with gr.Column(scale=7):
            question_input = gr.Textbox(
                label="Ваш вопрос",
                placeholder="Задайте вопрос о документе...",
                lines=3
            )

    submit_btn = gr.Button("Получить ответ", variant="primary")

    answer_output = gr.Textbox(
        label="Ответ системы",
        interactive=False,
        lines=10
    )


    submit_btn.click(
        fn=process_uploaded_file_and_question,
        inputs=[file_input, question_input],
        outputs=answer_output
    )

if __name__ == "__main__":
    logger.info("Запуск Gradio интерфейса")
    demo.launch(
        server_name=os.getenv("GRADIO_SERVER_NAME", "0.0.0.0"),
        server_port=int(os.getenv("GRADIO_SERVER_PORT", 7860)),
        share=os.getenv("GRADIO_SHARE", "false").lower() == "true"
    )
