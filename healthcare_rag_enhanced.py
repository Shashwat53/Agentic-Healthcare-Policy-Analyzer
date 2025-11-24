"""
Enhanced Healthcare Document Extraction System
Production-grade implementation with:
- Professional prompt engineering
- Structured table outputs
- Enhanced LangGraph with visualization
- Query classification and routing
- Confidence scoring
- Clean interface
"""

import os
import re
from pathlib import Path
from typing import List, Dict, Optional, Any, TypedDict, Literal
from dataclasses import dataclass
from datetime import datetime
import json

import numpy as np
import pandas as pd
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
import faiss
from rank_bm25 import BM25Okapi

from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langgraph.graph import StateGraph, START, END

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch


# ============================================
# CONFIGURATION
# ============================================

@dataclass
class SystemConfig:
    """System configuration parameters"""
    # Model settings
    llm_model: str = "Qwen/Qwen2.5-3B-Instruct"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Retrieval settings
    top_k_retrieve: int = 10
    top_k_final: int = 5
    rerank_enabled: bool = True
    
    # Chunking settings
    chunk_size: int = 800
    chunk_overlap: int = 200
    
    # Generation settings
    max_new_tokens: int = 800
    temperature: float = 0.1
    
    # System behavior
    enable_confidence_scoring: bool = True
    enable_query_classification: bool = True
    min_confidence_threshold: float = 0.6


CONFIG = SystemConfig()


# ============================================
# PROFESSIONAL PROMPTS (NO AI PATTERNS)
# ============================================

class PromptLibrary:
    """Professional medical document prompts without AI-generated patterns"""
    
    QUERY_CLASSIFIER = """Analyze the following medical query and classify it into ONE category.

Query: {query}

Categories:
- CLINICAL: Patient care, treatment protocols, clinical guidelines
- ADMINISTRATIVE: Billing, insurance, documentation requirements
- REGULATORY: Compliance, legal requirements, policy changes
- PHARMACEUTICAL: Medication information, drug interactions, formularies
- RESEARCH: Clinical studies, medical research, evidence-based data

Response format (JSON):
{{
    "category": "<category>",
    "confidence": <0.0-1.0>,
    "reasoning": "<brief explanation>"
}}

Classification:"""

    RAG_ANSWER_GENERATION = """You are a medical documentation specialist. Answer the query using ONLY the provided source documents.

Query: {query}

Source Documents:
{context}

Instructions:
1. Answer directly based on source documents
2. Cite specific sources using [Source N] format
3. If information is insufficient, state what is missing
4. Use precise medical terminology
5. Organize complex answers with clear structure
6. Never speculate beyond source material

Answer:"""

    ANSWER_VERIFICATION = """Verify if the following answer appropriately addresses the query using the source documents.

Query: {query}

Answer: {answer}

Source Documents:
{context}

Evaluation criteria:
1. Factual accuracy against sources
2. Completeness of answer
3. Proper source citation
4. Clinical appropriateness

Response format (JSON):
{{
    "is_adequate": <true/false>,
    "confidence_score": <0.0-1.0>,
    "missing_elements": ["<list any missing information>"],
    "suggested_improvements": "<brief suggestion if inadequate>"
}}

Verification:"""

    EXTRACTION_WITH_EXAMPLES = """Extract specific information from the medical document according to the query.

Query: {query}

Document Content:
{content}

Example extractions:

Query: "What is the coverage period?"
Answer: "Coverage period is January 1, 2024 to December 31, 2024 per Section 2.1"

Query: "What are the prior authorization requirements?"
Answer: "Prior authorization required for: 1) Specialty medications 2) Procedures over $5000 3) Out-of-network services per Policy Section 4.2"

Query: "List covered preventive services"
Answer: "Covered preventive services include: Annual physical exam, Immunizations per CDC schedule, Cancer screenings per USPSTF guidelines, Well-child visits. Reference: Benefits Schedule Section 3"

Your extraction:"""

    CONFIDENCE_ASSESSMENT = """Assess confidence in the following answer based on source quality and completeness.

Query: {query}
Answer: {answer}

Number of sources: {num_sources}
Source types: {source_types}

Confidence factors:
- Multiple corroborating sources (+)
- Official/authoritative documents (+)
- Specific citations with page numbers (+)
- Vague or conflicting information (-)
- Limited source coverage (-)

Response format (JSON):
{{
    "confidence_score": <0.0-1.0>,
    "reliability": "<high/medium/low>",
    "reasoning": "<brief explanation>",
    "limitations": ["<list any limitations>"]
}}

Assessment:"""


# ============================================
# HEALTHCARE-AWARE PDF PROCESSOR
# ============================================

class HealthcarePDFProcessor:
    """Extract and chunk healthcare documents with medical terminology awareness"""
    
    # Medical section headers that should be preserved
    MEDICAL_HEADERS = [
        r'^BENEFITS?\s+SUMMARY',
        r'^COVERAGE\s+DETAILS?',
        r'^LIMITATIONS?\s+(?:AND\s+)?EXCLUSIONS?',
        r'^PRIOR\s+AUTHORIZATION',
        r'^FORMULARY',
        r'^PROVIDER\s+(?:NETWORK|DIRECTORY)',
        r'^CLAIMS?\s+PROCEDURES?',
        r'^APPEALS?\s+PROCESS',
        r'^SECTION\s+\d+',
        r'^SCHEDULE\s+OF\s+BENEFITS?',
    ]
    
    # Medical entities to preserve during chunking
    MEDICAL_ENTITIES = [
        r'\b(?:ICD|CPT|HCPCS)-\d+(?:\.\d+)?',  # Medical codes
        r'\b\d{3}-\d{2}-\d{4}\b',  # SSN/Member ID
        r'\$\d+(?:,\d{3})*(?:\.\d{2})?',  # Currency
        r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4}',  # Dates
    ]
    
    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def extract_from_pdf(self, pdf_path: str) -> List[Document]:
        """Extract text from PDF with metadata preservation"""
        documents = []
        
        try:
            pdf_document = fitz.open(pdf_path)
            filename = Path(pdf_path).name
            
            for page_num in range(len(pdf_document)):
                page = pdf_document[page_num]
                
                # Extract text with layout preservation
                text = page.get_text("text")
                
                # Clean and normalize
                text = self._clean_text(text)
                
                if text.strip():
                    # Identify section headers
                    sections = self._identify_sections(text)
                    
                    doc = Document(
                        page_content=text,
                        metadata={
                            "filename": filename,
                            "page_num": page_num + 1,
                            "sections": sections,
                            "char_count": len(text),
                            "source_type": "pdf"
                        }
                    )
                    documents.append(doc)
            
            pdf_document.close()
            
        except Exception as e:
            print(f"Error processing {pdf_path}: {str(e)}")
        
        return documents
    
    def _clean_text(self, text: str) -> str:
        """Clean extracted text while preserving medical formatting"""
        # Remove excessive whitespace but preserve paragraph breaks
        text = re.sub(r' +', ' ', text)
        text = re.sub(r'\n\n+', '\n\n', text)
        
        # Remove page numbers and headers/footers
        text = re.sub(r'^(?:Page|\d+)\s+of\s+\d+$', '', text, flags=re.MULTILINE)
        
        # Normalize bullet points
        text = re.sub(r'^\s*[•●○◦▪▫-]\s+', '- ', text, flags=re.MULTILINE)
        
        return text.strip()
    
    def _identify_sections(self, text: str) -> List[str]:
        """Identify medical document sections"""
        sections = []
        
        for pattern in self.MEDICAL_HEADERS:
            matches = re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE)
            for match in matches:
                sections.append(match.group().strip())
        
        return sections
    
    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """Chunk documents with medical context awareness"""
        chunked_docs = []
        
        for doc in documents:
            text = doc.page_content
            
            # Split on medical section boundaries first
            sections = self._split_on_sections(text)
            
            for section_text, section_header in sections:
                # Further split large sections
                if len(section_text) > self.chunk_size:
                    sub_chunks = self._sliding_window_chunk(section_text)
                else:
                    sub_chunks = [section_text]
                
                for chunk_text in sub_chunks:
                    # Create chunk with enhanced metadata
                    chunk_doc = Document(
                        page_content=chunk_text,
                        metadata={
                            **doc.metadata,
                            "section_header": section_header,
                            "chunk_size": len(chunk_text)
                        }
                    )
                    chunked_docs.append(chunk_doc)
        
        return chunked_docs
    
    def _split_on_sections(self, text: str) -> List[tuple]:
        """Split text on medical section headers"""
        sections = []
        current_header = "General"
        current_text = []
        
        lines = text.split('\n')
        
        for line in lines:
            # Check if line is a section header
            is_header = False
            for pattern in self.MEDICAL_HEADERS:
                if re.match(pattern, line.strip(), re.IGNORECASE):
                    # Save previous section
                    if current_text:
                        sections.append(('\n'.join(current_text), current_header))
                    
                    # Start new section
                    current_header = line.strip()
                    current_text = []
                    is_header = True
                    break
            
            if not is_header:
                current_text.append(line)
        
        # Add final section
        if current_text:
            sections.append(('\n'.join(current_text), current_header))
        
        return sections if sections else [(text, "General")]
    
    def _sliding_window_chunk(self, text: str) -> List[str]:
        """Split long text using sliding window with overlap"""
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            
            # Try to break at sentence boundary
            if end < len(text):
                # Look for period + space + capital letter
                sentence_end = text.rfind('. ', start, end)
                if sentence_end > start:
                    end = sentence_end + 1
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            start = end - self.chunk_overlap if end < len(text) else end
        
        return chunks


# ============================================
# HYBRID RETRIEVAL SYSTEM
# ============================================

class HybridRetriever:
    """Hybrid retrieval combining dense (FAISS) and sparse (BM25) search"""
    
    def __init__(self, embedding_model: str, rerank: bool = True):
        print(f"Initializing retrieval system with {embedding_model}")
        
        self.embedding_model = SentenceTransformer(embedding_model)
        self.rerank_enabled = rerank
        
        self.documents = []
        self.faiss_index = None
        self.bm25_index = None
        
    def index_documents(self, documents: List[Document]):
        """Create vector and BM25 indices"""
        print(f"Indexing {len(documents)} document chunks...")
        
        self.documents = documents
        texts = [doc.page_content for doc in documents]
        
        # Create dense embeddings for FAISS
        print("Creating vector embeddings...")
        embeddings = self.embedding_model.encode(
            texts,
            show_progress_bar=True,
            batch_size=32
        )
        
        # Build FAISS index
        dimension = embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatL2(dimension)
        self.faiss_index.add(embeddings.astype('float32'))
        
        # Build BM25 index
        print("Building BM25 index...")
        tokenized_corpus = [doc.lower().split() for doc in texts]
        self.bm25_index = BM25Okapi(tokenized_corpus)
        
        print(f"Indexing complete. Vector dim: {dimension}")
    
    def retrieve(
        self,
        query: str,
        k: int = 10,
        alpha: float = 0.5
    ) -> List[Document]:
        """Hybrid retrieval with reciprocal rank fusion"""
        
        # Dense retrieval (FAISS)
        query_embedding = self.embedding_model.encode([query])[0]
        distances, indices = self.faiss_index.search(
            query_embedding.reshape(1, -1).astype('float32'),
            k=k
        )
        dense_docs = [(idx, 1.0 / (60 + rank)) 
                      for rank, idx in enumerate(indices[0])]
        
        # Sparse retrieval (BM25)
        tokenized_query = query.lower().split()
        bm25_scores = self.bm25_index.get_scores(tokenized_query)
        top_bm25_indices = np.argsort(bm25_scores)[-k:][::-1]
        sparse_docs = [(idx, 1.0 / (60 + rank)) 
                       for rank, idx in enumerate(top_bm25_indices)]
        
        # Reciprocal Rank Fusion
        doc_scores = {}
        for idx, score in dense_docs:
            doc_scores[idx] = doc_scores.get(idx, 0) + alpha * score
        for idx, score in sparse_docs:
            doc_scores[idx] = doc_scores.get(idx, 0) + (1 - alpha) * score
        
        # Sort and return top-k
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        retrieved = [self.documents[idx] for idx, _ in sorted_docs[:k]]
        
        # Optional reranking
        if self.rerank_enabled and len(retrieved) > 3:
            retrieved = self._rerank(query, retrieved)
        
        return retrieved
    
    def _rerank(self, query: str, documents: List[Document]) -> List[Document]:
        """Simple reranking based on query term overlap"""
        query_terms = set(query.lower().split())
        
        scores = []
        for doc in documents:
            doc_terms = set(doc.page_content.lower().split())
            overlap = len(query_terms & doc_terms)
            scores.append(overlap)
        
        # Sort by overlap score
        ranked_indices = np.argsort(scores)[::-1]
        return [documents[i] for i in ranked_indices]


# ============================================
# ENHANCED LLM WITH STRUCTURED OUTPUT
# ============================================

class HealthcareLLM:
    """Healthcare-specialized LLM with structured output parsing"""
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-3B-Instruct",
        device: str = "cuda",
        max_new_tokens: int = 800
    ):
        print(f"Loading LLM: {model_name} on {device}")
        
        self.model_name = model_name
        self.device = device
        self.max_new_tokens = max_new_tokens
        
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None,
            trust_remote_code=True
        )
        
        if device == "cpu":
            self.model = self.model.to(device)
        
        self.model.eval()
        
        print("LLM loaded successfully\n")
    
    def generate(
        self,
        prompt: str,
        temperature: float = 0.1,
        extract_json: bool = False
    ) -> str:
        """Generate response with optional JSON extraction"""
        
        # Format as chat message
        messages = [{"role": "user", "content": prompt}]
        
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=4096
        ).to(self.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode
        response = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )
        
        # Extract JSON if requested
        if extract_json:
            response = self._extract_json(response)
        
        return response.strip()
    
    def _extract_json(self, text: str) -> str:
        """Extract JSON from response text"""
        # Try to find JSON block
        json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text, re.DOTALL)
        if json_match:
            return json_match.group()
        return text


# ============================================
# QUERY CLASSIFIER
# ============================================

class QueryClassifier:
    """Classify medical queries for routing"""
    
    def __init__(self, llm: HealthcareLLM):
        self.llm = llm
        self.prompt_template = PromptLibrary.QUERY_CLASSIFIER
    
    def classify(self, query: str) -> Dict[str, Any]:
        """Classify query into medical category"""
        
        prompt = self.prompt_template.format(query=query)
        
        try:
            response = self.llm.generate(prompt, temperature=0.0, extract_json=True)
            classification = json.loads(response)
            return classification
        except:
            # Fallback classification
            return {
                "category": "CLINICAL",
                "confidence": 0.5,
                "reasoning": "Default classification due to parsing error"
            }


# ============================================
# LANGGRAPH RAG SYSTEM
# ============================================

class RAGState(TypedDict):
    """State for RAG workflow"""
    query: str
    query_classification: Dict[str, Any]
    retrieved_docs: List[Document]
    context: str
    answer: str
    verification: Dict[str, Any]
    confidence_score: float
    iteration: int
    max_iterations: int
    needs_refinement: bool
    final_output: Dict[str, Any]


class EnhancedRAGSystem:
    """Production RAG system with LangGraph orchestration"""
    
    def __init__(
        self,
        llm: HealthcareLLM,
        retriever: HybridRetriever,
        config: SystemConfig
    ):
        self.llm = llm
        self.retriever = retriever
        self.config = config
        
        # Initialize query classifier
        self.classifier = QueryClassifier(llm) if config.enable_query_classification else None
        
        # Build graph
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build LangGraph workflow"""
        
        workflow = StateGraph(RAGState)
        
        # Add nodes
        workflow.add_node("classify_query", self._classify_query_node)
        workflow.add_node("retrieve_documents", self._retrieve_documents_node)
        workflow.add_node("format_context", self._format_context_node)
        workflow.add_node("generate_answer", self._generate_answer_node)
        workflow.add_node("verify_answer", self._verify_answer_node)
        workflow.add_node("assess_confidence", self._assess_confidence_node)
        workflow.add_node("prepare_output", self._prepare_output_node)
        
        # Define edges
        workflow.add_edge(START, "classify_query")
        workflow.add_edge("classify_query", "retrieve_documents")
        workflow.add_edge("retrieve_documents", "format_context")
        workflow.add_edge("format_context", "generate_answer")
        workflow.add_edge("generate_answer", "verify_answer")
        workflow.add_edge("verify_answer", "assess_confidence")
        
        # Conditional edge for refinement
        workflow.add_conditional_edges(
            "assess_confidence",
            self._should_refine,
            {
                "refine": "retrieve_documents",
                "finalize": "prepare_output"
            }
        )
        
        workflow.add_edge("prepare_output", END)
        
        return workflow.compile()
    
    def _classify_query_node(self, state: RAGState) -> RAGState:
        """Node: Classify the query"""
        if self.classifier:
            classification = self.classifier.classify(state["query"])
            state["query_classification"] = classification
        else:
            state["query_classification"] = {"category": "GENERAL", "confidence": 1.0}
        
        return state
    
    def _retrieve_documents_node(self, state: RAGState) -> RAGState:
        """Node: Retrieve relevant documents"""
        docs = self.retriever.retrieve(
            state["query"],
            k=self.config.top_k_retrieve
        )
        state["retrieved_docs"] = docs
        return state
    
    def _format_context_node(self, state: RAGState) -> RAGState:
        """Node: Format retrieved docs as context"""
        docs = state["retrieved_docs"][:self.config.top_k_final]
        
        context_parts = []
        for idx, doc in enumerate(docs, 1):
            source_info = f"{doc.metadata['filename']} (Page {doc.metadata['page_num']})"
            if 'section_header' in doc.metadata:
                source_info += f" - {doc.metadata['section_header']}"
            
            context_parts.append(
                f"[Source {idx}: {source_info}]\n{doc.page_content}\n"
            )
        
        state["context"] = "\n".join(context_parts)
        return state
    
    def _generate_answer_node(self, state: RAGState) -> RAGState:
        """Node: Generate answer using LLM"""
        prompt = PromptLibrary.RAG_ANSWER_GENERATION.format(
            query=state["query"],
            context=state["context"]
        )
        
        answer = self.llm.generate(prompt, temperature=self.config.temperature)
        state["answer"] = answer
        return state
    
    def _verify_answer_node(self, state: RAGState) -> RAGState:
        """Node: Verify answer quality"""
        prompt = PromptLibrary.ANSWER_VERIFICATION.format(
            query=state["query"],
            answer=state["answer"],
            context=state["context"]
        )
        
        try:
            verification_response = self.llm.generate(
                prompt,
                temperature=0.0,
                extract_json=True
            )
            verification = json.loads(verification_response)
        except:
            verification = {
                "is_adequate": True,
                "confidence_score": 0.7,
                "missing_elements": [],
                "suggested_improvements": ""
            }
        
        state["verification"] = verification
        return state
    
    def _assess_confidence_node(self, state: RAGState) -> RAGState:
        """Node: Assess overall confidence"""
        
        # Combine multiple confidence signals
        verification_conf = state["verification"].get("confidence_score", 0.7)
        classification_conf = state["query_classification"].get("confidence", 1.0)
        
        # Simple weighted average
        overall_confidence = (verification_conf * 0.7 + classification_conf * 0.3)
        
        state["confidence_score"] = overall_confidence
        state["iteration"] = state.get("iteration", 0) + 1
        
        return state
    
    def _should_refine(self, state: RAGState) -> Literal["refine", "finalize"]:
        """Decision: Should we refine the answer?"""
        
        # Check if we should refine based on:
        # 1. Low confidence
        # 2. Not exceeded max iterations
        # 3. Answer marked as inadequate
        
        max_iter = state.get("max_iterations", 1)
        current_iter = state.get("iteration", 0)
        
        is_inadequate = not state["verification"].get("is_adequate", True)
        low_confidence = state["confidence_score"] < self.config.min_confidence_threshold
        
        if (is_inadequate or low_confidence) and current_iter < max_iter:
            return "refine"
        
        return "finalize"
    
    def _prepare_output_node(self, state: RAGState) -> RAGState:
        """Node: Prepare final structured output"""
        
        # Extract source citations
        sources = []
        for idx, doc in enumerate(state["retrieved_docs"][:self.config.top_k_final], 1):
            sources.append({
                "source_id": idx,
                "filename": doc.metadata["filename"],
                "page": doc.metadata["page_num"],
                "section": doc.metadata.get("section_header", "N/A"),
                "relevance": "High" if idx <= 3 else "Medium"
            })
        
        # Prepare final output
        state["final_output"] = {
            "query": state["query"],
            "category": state["query_classification"].get("category", "GENERAL"),
            "answer": state["answer"],
            "confidence_score": round(state["confidence_score"], 3),
            "confidence_level": self._confidence_label(state["confidence_score"]),
            "sources": sources,
            "verification_status": "Verified" if state["verification"].get("is_adequate", True) else "Needs Review",
            "iterations": state["iteration"],
            "limitations": state["verification"].get("missing_elements", [])
        }
        
        return state
    
    def _confidence_label(self, score: float) -> str:
        """Convert confidence score to label"""
        if score >= 0.8:
            return "High"
        elif score >= 0.6:
            return "Medium"
        else:
            return "Low"
    
    def query(
        self,
        question: str,
        max_iterations: int = 1
    ) -> Dict[str, Any]:
        """Execute RAG query through the graph"""
        
        initial_state = {
            "query": question,
            "query_classification": {},
            "retrieved_docs": [],
            "context": "",
            "answer": "",
            "verification": {},
            "confidence_score": 0.0,
            "iteration": 0,
            "max_iterations": max_iterations,
            "needs_refinement": False,
            "final_output": {}
        }
        
        # Execute graph
        final_state = self.graph.invoke(initial_state)
        
        return final_state["final_output"]
    
    def get_graph_visualization(self) -> str:
        """Generate ASCII visualization of the graph"""
        return self.graph.get_graph().draw_ascii()


# ============================================
# OUTPUT FORMATTING
# ============================================

class OutputFormatter:
    """Format RAG outputs as structured tables and reports"""
    
    @staticmethod
    def format_result_as_dataframe(result: Dict[str, Any]) -> pd.DataFrame:
        """Convert result to pandas DataFrame"""
        
        # Main result info
        main_df = pd.DataFrame([{
            "Query": result["query"],
            "Category": result["category"],
            "Confidence": f"{result['confidence_score']:.1%}",
            "Level": result["confidence_level"],
            "Status": result["verification_status"],
            "Iterations": result["iterations"]
        }])
        
        return main_df
    
    @staticmethod
    def format_sources_as_dataframe(result: Dict[str, Any]) -> pd.DataFrame:
        """Convert sources to DataFrame"""
        sources_data = []
        
        for source in result["sources"]:
            sources_data.append({
                "ID": source["source_id"],
                "Document": source["filename"],
                "Page": source["page"],
                "Section": source["section"],
                "Relevance": source["relevance"]
            })
        
        return pd.DataFrame(sources_data)
    
    @staticmethod
    def print_formatted_result(result: Dict[str, Any]):
        """Print formatted result with tables"""
        
        print("\n" + "="*80)
        print("QUERY ANALYSIS")
        print("="*80 + "\n")
        
        # Main result table
        main_df = OutputFormatter.format_result_as_dataframe(result)
        print(main_df.to_string(index=False))
        
        print("\n" + "-"*80)
        print("ANSWER")
        print("-"*80 + "\n")
        print(result["answer"])
        
        print("\n" + "-"*80)
        print("SOURCE DOCUMENTS")
        print("-"*80 + "\n")
        
        # Sources table
        sources_df = OutputFormatter.format_sources_as_dataframe(result)
        print(sources_df.to_string(index=False))
        
        # Limitations if any
        if result.get("limitations"):
            print("\n" + "-"*80)
            print("LIMITATIONS")
            print("-"*80 + "\n")
            for limitation in result["limitations"]:
                print(f"- {limitation}")
        
        print("\n" + "="*80 + "\n")


# ============================================
# MAIN EXECUTION
# ============================================

def main():
    """Main execution function"""
    
    print("="*80)
    print("ENHANCED HEALTHCARE DOCUMENT EXTRACTION SYSTEM")
    print("="*80 + "\n")
    
    # Initialize components
    print("Initializing system components...\n")
    
    # 1. PDF Processor
    pdf_processor = HealthcarePDFProcessor(
        chunk_size=CONFIG.chunk_size,
        chunk_overlap=CONFIG.chunk_overlap
    )
    
    # 2. Retriever
    retriever = HybridRetriever(
        embedding_model=CONFIG.embedding_model,
        rerank=CONFIG.rerank_enabled
    )
    
    # 3. LLM
    llm = HealthcareLLM(
        model_name=CONFIG.llm_model,
        device=CONFIG.device,
        max_new_tokens=CONFIG.max_new_tokens
    )
    
    # 4. RAG System
    rag_system = EnhancedRAGSystem(
        llm=llm,
        retriever=retriever,
        config=CONFIG
    )
    
    print("\nSystem initialized successfully!")
    print("\nLangGraph Visualization:")
    print("-"*80)
    print(rag_system.get_graph_visualization())
    print("-"*80 + "\n")
    
    return rag_system, pdf_processor, retriever, llm


if __name__ == "__main__":
    rag_system, pdf_processor, retriever, llm = main()
