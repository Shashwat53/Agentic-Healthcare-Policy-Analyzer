"""
Gradio Interface for Healthcare Document Extraction System
Clean, professional UI for medical document queries
"""

import gradio as gr
import pandas as pd
from typing import Dict, Any, Tuple
import os
from pathlib import Path


class HealthcareRAGInterface:
    """Gradio interface for RAG system"""
    
    def __init__(self, rag_system, pdf_processor, retriever):
        self.rag_system = rag_system
        self.pdf_processor = pdf_processor
        self.retriever = retriever
        self.documents_loaded = False
    
    def process_documents(self, pdf_files) -> str:
        """Process uploaded PDF files"""
        if not pdf_files:
            return "Error: No files uploaded"
        
        try:
            all_documents = []
            
            for pdf_file in pdf_files:
                # Extract text from PDF
                documents = self.pdf_processor.extract_from_pdf(pdf_file.name)
                all_documents.extend(documents)
            
            # Chunk documents
            chunked_docs = self.pdf_processor.chunk_documents(all_documents)
            
            # Index documents
            self.retriever.index_documents(chunked_docs)
            
            self.documents_loaded = True
            
            return f"""Document Processing Complete
            
Files processed: {len(pdf_files)}
Total pages extracted: {len(all_documents)}
Chunks created: {len(chunked_docs)}

System ready for queries."""
            
        except Exception as e:
            return f"Error processing documents: {str(e)}"
    
    def query_documents(
        self,
        query: str,
        max_iterations: int,
        enable_refinement: bool
    ) -> Tuple[str, str, str]:
        """Query the document system"""
        
        if not self.documents_loaded:
            return (
                "Error: Please upload and process documents first.",
                "",
                ""
            )
        
        if not query.strip():
            return (
                "Error: Please enter a query.",
                "",
                ""
            )
        
        try:
            # Execute query
            iterations = max_iterations if enable_refinement else 1
            result = self.rag_system.query(query, max_iterations=iterations)
            
            # Format outputs
            answer_text = self._format_answer(result)
            metadata_html = self._format_metadata_html(result)
            sources_html = self._format_sources_html(result)
            
            return answer_text, metadata_html, sources_html
            
        except Exception as e:
            return (
                f"Error processing query: {str(e)}",
                "",
                ""
            )
    
    def _format_answer(self, result: Dict[str, Any]) -> str:
        """Format answer text"""
        confidence_emoji = {
            "High": "🟢",
            "Medium": "🟡",
            "Low": "🔴"
        }
        
        confidence = result["confidence_level"]
        emoji = confidence_emoji.get(confidence, "⚪")
        
        output = f"""CONFIDENCE: {emoji} {confidence} ({result['confidence_score']:.1%})
CATEGORY: {result['category']}
STATUS: {result['verification_status']}

ANSWER:
{result['answer']}
"""
        
        if result.get("limitations"):
            output += "\n\nLIMITATIONS:\n"
            for limitation in result["limitations"]:
                output += f"• {limitation}\n"
        
        return output
    
    def _format_metadata_html(self, result: Dict[str, Any]) -> str:
        """Format metadata as HTML table"""
        
        metadata_items = [
            ("Query", result["query"]),
            ("Category", result["category"]),
            ("Confidence Score", f"{result['confidence_score']:.1%}"),
            ("Confidence Level", result["confidence_level"]),
            ("Verification", result["verification_status"]),
            ("Iterations", result["iterations"]),
        ]
        
        html = '<table style="width:100%; border-collapse: collapse;">'
        html += '<tr style="background-color: #f0f0f0;"><th style="padding: 8px; text-align: left; border: 1px solid #ddd;">Field</th><th style="padding: 8px; text-align: left; border: 1px solid #ddd;">Value</th></tr>'
        
        for field, value in metadata_items:
            html += f'<tr><td style="padding: 8px; border: 1px solid #ddd; font-weight: bold;">{field}</td><td style="padding: 8px; border: 1px solid #ddd;">{value}</td></tr>'
        
        html += '</table>'
        
        return html
    
    def _format_sources_html(self, result: Dict[str, Any]) -> str:
        """Format sources as HTML table"""
        
        html = '<table style="width:100%; border-collapse: collapse;">'
        html += '''<tr style="background-color: #f0f0f0;">
            <th style="padding: 8px; text-align: left; border: 1px solid #ddd;">ID</th>
            <th style="padding: 8px; text-align: left; border: 1px solid #ddd;">Document</th>
            <th style="padding: 8px; text-align: left; border: 1px solid #ddd;">Page</th>
            <th style="padding: 8px; text-align: left; border: 1px solid #ddd;">Section</th>
            <th style="padding: 8px; text-align: left; border: 1px solid #ddd;">Relevance</th>
        </tr>'''
        
        for source in result["sources"]:
            relevance_color = "#28a745" if source["relevance"] == "High" else "#ffc107"
            
            html += f'''<tr>
                <td style="padding: 8px; border: 1px solid #ddd;">{source["source_id"]}</td>
                <td style="padding: 8px; border: 1px solid #ddd;">{source["filename"]}</td>
                <td style="padding: 8px; border: 1px solid #ddd;">{source["page"]}</td>
                <td style="padding: 8px; border: 1px solid #ddd;">{source["section"]}</td>
                <td style="padding: 8px; border: 1px solid #ddd; background-color: {relevance_color}; color: white; font-weight: bold;">{source["relevance"]}</td>
            </tr>'''
        
        html += '</table>'
        
        return html
    
    def create_interface(self) -> gr.Blocks:
        """Create Gradio interface"""
        
        with gr.Blocks(
            title="Healthcare Document Extraction System",
            theme=gr.themes.Soft()
        ) as interface:
            
            gr.Markdown(
                """
                # Healthcare Document Extraction System
                
                Professional medical document analysis using advanced retrieval and AI.
                
                **Features:**
                - Multi-document PDF processing
                - Hybrid retrieval (semantic + keyword)
                - Query classification and routing
                - Confidence scoring and verification
                - Structured source attribution
                """
            )
            
            with gr.Tab("Document Upload"):
                gr.Markdown("### Upload Medical Documents")
                gr.Markdown("Upload one or more PDF documents for analysis.")
                
                with gr.Row():
                    pdf_input = gr.File(
                        label="Upload PDF Documents",
                        file_count="multiple",
                        file_types=[".pdf"]
                    )
                
                process_btn = gr.Button("Process Documents", variant="primary")
                process_output = gr.Textbox(
                    label="Processing Status",
                    lines=8,
                    interactive=False
                )
                
                process_btn.click(
                    fn=self.process_documents,
                    inputs=[pdf_input],
                    outputs=[process_output]
                )
            
            with gr.Tab("Query Documents"):
                gr.Markdown("### Query Your Documents")
                
                with gr.Row():
                    with gr.Column(scale=2):
                        query_input = gr.Textbox(
                            label="Enter your query",
                            placeholder="Example: What are the coverage details for preventive care services?",
                            lines=3
                        )
                    
                    with gr.Column(scale=1):
                        max_iter_input = gr.Slider(
                            minimum=1,
                            maximum=3,
                            value=1,
                            step=1,
                            label="Max Refinement Iterations",
                            info="Number of refinement attempts for low-confidence answers"
                        )
                        
                        refinement_checkbox = gr.Checkbox(
                            label="Enable Answer Refinement",
                            value=False,
                            info="Automatically refine answers with low confidence"
                        )
                
                query_btn = gr.Button("Query Documents", variant="primary")
                
                gr.Markdown("### Results")
                
                with gr.Row():
                    with gr.Column(scale=2):
                        answer_output = gr.Textbox(
                            label="Answer",
                            lines=15,
                            interactive=False
                        )
                    
                    with gr.Column(scale=1):
                        metadata_output = gr.HTML(
                            label="Query Metadata"
                        )
                
                gr.Markdown("### Source Documents")
                sources_output = gr.HTML()
                
                query_btn.click(
                    fn=self.query_documents,
                    inputs=[query_input, max_iter_input, refinement_checkbox],
                    outputs=[answer_output, metadata_output, sources_output]
                )
            
            with gr.Tab("Example Queries"):
                gr.Markdown(
                    """
                    ### Sample Queries
                    
                    Try these example queries to test the system:
                    
                    **Clinical Queries:**
                    - What are the coverage details for preventive care services?
                    - List all services that require prior authorization
                    - What is the process for filing an appeal?
                    
                    **Administrative Queries:**
                    - What are the copayment amounts for specialist visits?
                    - Explain the out-of-network coverage policy
                    - What documentation is required for claims submission?
                    
                    **Pharmaceutical Queries:**
                    - Which medications are on the formulary?
                    - What is the drug utilization review process?
                    - Explain the step therapy requirements
                    
                    **Regulatory Queries:**
                    - What are the HIPAA compliance requirements mentioned?
                    - List all member rights and responsibilities
                    - What are the quality assurance standards?
                    """
                )
            
            with gr.Tab("System Info"):
                gr.Markdown(
                    f"""
                    ### System Configuration
                    
                    **Models:**
                    - LLM: {self.rag_system.llm.model_name}
                    - Embeddings: {self.rag_system.config.embedding_model}
                    - Device: {self.rag_system.config.device}
                    
                    **Retrieval Settings:**
                    - Top-K Retrieve: {self.rag_system.config.top_k_retrieve}
                    - Top-K Final: {self.rag_system.config.top_k_final}
                    - Reranking: {'Enabled' if self.rag_system.config.rerank_enabled else 'Disabled'}
                    
                    **Generation Settings:**
                    - Max Tokens: {self.rag_system.config.max_new_tokens}
                    - Temperature: {self.rag_system.config.temperature}
                    
                    **Features:**
                    - Query Classification: {'Enabled' if self.rag_system.config.enable_query_classification else 'Disabled'}
                    - Confidence Scoring: {'Enabled' if self.rag_system.config.enable_confidence_scoring else 'Disabled'}
                    - Min Confidence Threshold: {self.rag_system.config.min_confidence_threshold}
                    
                    ### LangGraph Workflow
                    
                    ```
                    START
                      ↓
                    Classify Query
                      ↓
                    Retrieve Documents
                      ↓
                    Format Context
                      ↓
                    Generate Answer
                      ↓
                    Verify Answer
                      ↓
                    Assess Confidence
                      ↓
                    [Decision: Refine or Finalize?]
                      ↓
                    Prepare Output
                      ↓
                    END
                    ```
                    """
                )
        
        return interface


def create_gradio_interface(rag_system, pdf_processor, retriever):
    """Create and launch Gradio interface"""
    
    interface_handler = HealthcareRAGInterface(
        rag_system=rag_system,
        pdf_processor=pdf_processor,
        retriever=retriever
    )
    
    interface = interface_handler.create_interface()
    
    return interface


# Example usage in Colab:
# interface = create_gradio_interface(rag_system, pdf_processor, retriever)
# interface.launch(share=True, debug=True)
