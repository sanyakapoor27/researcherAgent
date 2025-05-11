import streamlit as st
import os
import json
import datetime
import uuid
import requests
from bs4 import BeautifulSoup
import re
import sys
try:
    import pysqlite3
    sys.modules["sqlite3"] = sys.modules["pysqlite3"]
except ImportError:
    raise RuntimeError("pysqlite3-binary must be installed in requirements.txt")
import chromadb
from chromadb import PersistentClient
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import google.generativeai as genai
from typing import List, Dict, Any, Optional
import pandas as pd
import time
import logging
from urllib.parse import urlparse
import random
import httpx
from ratelimit import limits, sleep_and_retry
from functools import lru_cache

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set page configuration
st.set_page_config(
    page_title="Research Agent with Citations",
    page_icon="📚",
    layout="wide",
)

# Initialize the Gemini client
def get_gemini_client():
    if "GEMINI_API_KEY" not in st.session_state:
        st.session_state.GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
    
    if st.session_state.GEMINI_API_KEY:
        genai.configure(api_key=st.session_state.GEMINI_API_KEY)
        return genai
    return None

# Initialize ChromaDB for storing documents and citations
@st.cache_resource
def get_chroma_client():
    try:
        # First try the updated method for newer versions of ChromaDB
        client = PersistentClient(path="/tmp/chroma")
    except Exception as e:
        logger.warning(f"Failed to initialize ChromaDB with persist_directory: {e}")
        # Fallback to in-memory if persist_directory doesn't work
        client = chromadb.Client(Settings(chroma_db_impl="duckdb+parquet"))
    
    # Create collections if they don't exist
    try:
        documents_collection = client.get_collection("research_documents")
    except:
        documents_collection = client.create_collection("research_documents")
    
    try:
        memory_collection = client.get_collection("agent_memory")
    except:
        memory_collection = client.create_collection("agent_memory")
    
    return client

# Rate limit settings
RATE_LIMIT_CALLS = 3     # Number of calls
RATE_LIMIT_PERIOD = 60   # In seconds

# Text processing utilities
def clean_text(text):
    """Clean the text by removing extra whitespace and newlines"""
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def extract_text_from_html(html_content):
    """Extract readable text from HTML"""
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Remove script and style elements
    for script in soup(["script", "style"]):
        script.extract()
    
    # Get text
    text = soup.get_text()
    
    # Clean text
    lines = (line.strip() for line in text.splitlines())
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    text = '\n'.join(chunk for chunk in chunks if chunk)
    
    return text

def split_text_into_chunks(text, chunk_size=1000, chunk_overlap=100):
    """Split text into chunks for processing"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    
    chunks = text_splitter.split_text(text)
    return chunks

# Semantic Scholar API functions
@sleep_and_retry
@limits(calls=RATE_LIMIT_CALLS, period=RATE_LIMIT_PERIOD)
def search_semantic_scholar(query, limit=10):
    """Search for papers using Semantic Scholar API with rate limiting"""
    logger.info(f"Searching Semantic Scholar for: {query}")
    
    base_url = "https://api.semanticscholar.org/graph/v1/paper/search"
    
    params = {
        'query': query,
        'limit': limit,
        'fields': 'title,authors,abstract,url,venue,year,citations,references'
    }
    
    headers = {
        'User-Agent': 'ResearchAgent/1.0 (Academic Research Project)'
    }
    
    try:
        response = requests.get(base_url, params=params, headers=headers)
        response.raise_for_status()
        
        results = response.json()
        papers = results.get('data', [])
        
        return papers
    except requests.exceptions.RequestException as e:
        logger.error(f"Error searching Semantic Scholar: {e}")
        # Add exponential backoff if needed
        time.sleep(5)
        return []

def format_semantic_scholar_results(papers):
    """Format the results from Semantic Scholar API"""
    formatted_results = []
    
    for paper in papers:
        # Extract author names
        authors = []
        if paper.get('authors'):
            authors = [author.get('name', '') for author in paper.get('authors', [])]
        
        # Format the paper data
        paper_data = {
            "title": paper.get('title', 'Untitled Paper'),
            "url": paper.get('url', '') or f"https://www.semanticscholar.org/paper/{paper.get('paperId', '')}",
            "abstract": paper.get('abstract', 'No abstract available'),
            "authors": authors,
            "venue": paper.get('venue', 'Unknown Venue'),
            "year": paper.get('year', 'Unknown Year'),
            "source": "Semantic Scholar"
        }
        
        formatted_results.append(paper_data)
    
    return formatted_results

# Web scraping with rate limiting
@sleep_and_retry
@limits(calls=RATE_LIMIT_CALLS, period=RATE_LIMIT_PERIOD)
def fetch_content_with_rate_limit(url):
    """Fetch content from a URL with rate limiting"""
    logger.info(f"Fetching content from: {url}")
    
    # Parse URL to get domain for polite scraping
    parsed_url = urlparse(url)
    domain = parsed_url.netloc
    
    headers = {
        'User-Agent': 'ResearchAgent/1.0 (Academic Research Project)',
        'Accept': 'text/html,application/xhtml+xml,application/xml',
        'Accept-Language': 'en-US,en;q=0.9'
    }
    
    try:
        # Add a small random delay to be more polite
        time.sleep(random.uniform(1, 3))
        
        with httpx.Client(timeout=30.0) as client:
            response = client.get(url, headers=headers, follow_redirects=True)
            response.raise_for_status()
            
            content_type = response.headers.get('Content-Type', '')
            
            # Handle different content types
            if 'text/html' in content_type or 'application/xhtml+xml' in content_type:
                # HTML content
                return extract_text_from_html(response.text)
            elif 'application/pdf' in content_type:
                # For PDFs, we would need a PDF parser
                # This is a placeholder - in a real implementation you'd use PyPDF2 or similar
                return "PDF content detected. PDF parsing would be implemented here."
            elif 'text/plain' in content_type:
                # Plain text
                return response.text
            else:
                return f"Unsupported content type: {content_type}"
    except Exception as e:
        logger.error(f"Error fetching content from {url}: {e}")
        return f"Error fetching content: {str(e)}"

# ChromaDB memory functions
def add_to_memory(memory_item):
    """Add item to agent memory in ChromaDB"""
    client = get_chroma_client()
    memory_collection = client.get_collection("agent_memory")
    
    item_id = str(uuid.uuid4())
    
    memory_collection.add(
        ids=[item_id],
        documents=[memory_item["content"]],
        metadatas=[{
            "type": memory_item["type"],
            "timestamp": str(datetime.datetime.now()),
            "query": memory_item.get("query", ""),
            "source": memory_item.get("source", "")
        }]
    )
    
    return item_id

def add_document_with_citation(document, metadata):
    """Add document chunks with citation metadata to the database"""
    client = get_chroma_client()
    docs_collection = client.get_collection("research_documents")
    
    chunks = split_text_into_chunks(document)
    ids = [str(uuid.uuid4()) for _ in chunks]
    
    metadatas = [{
        "title": metadata.get("title", "Unknown"),
        "url": metadata.get("url", ""),
        "source": metadata.get("source", "Unknown"),
        "authors": ', '.join(metadata.get("authors", [])) if isinstance(metadata.get("authors"), list) else metadata.get("authors", ""),
        "year": metadata.get("year", ""),
        "venue": metadata.get("venue", ""),
        "timestamp": str(datetime.datetime.now()),
        "chunk_id": i
    } for i in range(len(chunks))]
    
    docs_collection.add(
        ids=ids,
        documents=chunks,
        metadatas=metadatas
    )
    
    return ids

def retrieve_relevant_documents(query, limit=5):
    """Retrieve relevant documents based on the query"""
    client = get_chroma_client()
    docs_collection = client.get_collection("research_documents")
    
    results = docs_collection.query(
        query_texts=[query],
        n_results=limit
    )
    
    documents = []
    for i, doc in enumerate(results["documents"][0]):
        documents.append({
            "content": doc,
            "metadata": results["metadatas"][0][i]
        })
    
    return documents

def retrieve_memory(query=None, memory_type=None, limit=5):
    """Retrieve agent memory from ChromaDB, optionally filtered by query and type"""
    client = get_chroma_client()
    memory_collection = client.get_collection("agent_memory")
    
    if query:
        where_filter = None
        if memory_type:
            where_filter = {"type": memory_type}
        
        results = memory_collection.query(
            query_texts=[query],
            where=where_filter,
            n_results=limit
        )
        
        memories = []
        for i, doc in enumerate(results["documents"][0]):
            memories.append({
                "content": doc,
                "metadata": results["metadatas"][0][i]
            })
        
        return memories
    else:
        # Get all memories, optionally filtered by type
        if memory_type:
            all_items = memory_collection.get(
                where={"type": memory_type}
            )
        else:
            all_items = memory_collection.get()
        
        memories = []
        for i, doc in enumerate(all_items["documents"]):
            if i < len(all_items["metadatas"]):
                memories.append({
                    "content": doc,
                    "metadata": all_items["metadatas"][i]
                })
            else:
                # Handle case where metadatas may be missing
                memories.append({
                    "content": doc,
                    "metadata": {"type": "unknown"}
                })
        
        return memories

# AI processing functions
def generate_research_summary(query, documents):
    """Generate a summary of research findings with citations"""
    gemini_client = get_gemini_client()
    if not gemini_client:
        logger.error("Gemini client not available")
        return "Error: Gemini API key not configured."
    
    # Prepare documents for context
    doc_context = ""
    for i, doc in enumerate(documents):
        doc_context += f"\nDocument {i+1} from {doc['metadata']['source']}"
        if 'authors' in doc['metadata'] and doc['metadata']['authors']:
            doc_context += f" by {doc['metadata']['authors']}"
        if 'year' in doc['metadata'] and doc['metadata']['year']:
            doc_context += f" ({doc['metadata']['year']})"
        doc_context += f" (URL: {doc['metadata']['url']}):\n"
        doc_context += doc["content"]
        doc_context += "\n---\n"
    
    prompt = f"""
    You are a research assistant. Based on the following documents, create a comprehensive
    summary addressing this research query: "{query}"
    
    Include key findings, methodologies, and conclusions. Cite your sources using numbered citations in square brackets [1], [2], etc.
    
    At the end of your summary, include a "References" section listing all the sources in proper academic format.
    
    Here are the documents:
    {doc_context}
    """
    
    generation_config = {
        "temperature": 0.2,
        "max_output_tokens": 2000,
    }
    
    safety_settings = [
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    ]
    
    try:
        model = gemini_client.GenerativeModel(
            model_name="gemini-2.0-flash",
            generation_config=generation_config,
            safety_settings=safety_settings,
        )
        
        chat = model.start_chat()

        system_instruction = "You are a research planning assistant who creates practical research strategies.\n\n"
        response = chat.send_message(system_instruction + prompt)

        summary = response.text
        
        # Store this summary in memory
        add_to_memory({
            "type": "research_summary",
            "content": summary,
            "query": query,
            "source": "generated_summary"
        })
        
        return summary
    except Exception as e:
        logger.error(f"Error generating research summary: {e}")
        return f"Error generating research summary: {str(e)}"

def generate_follow_up_questions(query, summary):
    """Generate follow-up research questions based on the current findings"""
    gemini_client = get_gemini_client()
    if not gemini_client:
        logger.error("Gemini client not available")
        return "Error: Gemini API key not configured."
    
    prompt = f"""
    Based on the following research query and summary, generate 3-5 relevant follow-up questions
    that would help deepen the research. Identify gaps in knowledge or areas worth exploring further.
    
    Original query: "{query}"
    
    Summary of findings:
    {summary}
    
    Format your response as a numbered list of questions only.
    """
    
    generation_config = {
        "temperature": 0.7,
        "max_output_tokens": 500,
    }
    
    try:
        model = gemini_client.GenerativeModel(
            model_name="gemini-2.0-flash",
            generation_config=generation_config,
        )
        
        chat = model.start_chat()
        system_instruction = "You are a research assistant helping to identify valuable follow-up questions."
        
        response = chat.send_message(system_instruction + prompt)
        follow_up_questions = response.text
        
        return follow_up_questions
    except Exception as e:
        logger.error(f"Error generating follow-up questions: {e}")
        return f"Error generating follow-up questions: {str(e)}"

def get_research_plan(query):
    """Generate a research plan for a given query"""
    gemini_client = get_gemini_client()
    if not gemini_client:
        logger.error("Gemini client not available")
        return "Error: Gemini API key not configured."
    
    prompt = f"""
    Create a research plan for the following query: "{query}"
    
    Your plan should include:
    1. Key aspects of the topic to explore
    2. Types of sources to prioritize
    3. Specific search terms to use
    4. How to evaluate and synthesize the information
    
    Make this plan practical and actionable for academic research.
    """
    
    generation_config = {
        "temperature": 0.4,
        "max_output_tokens": 800,
    }
    
    try:
        model = gemini_client.GenerativeModel(
            model_name="gemini-2.0-flash",
            generation_config=generation_config,
        )
        
        chat = model.start_chat()
        system_instruction="You are a research planning assistant who creates practical research strategies."

        response = chat.send_message(system_instruction + prompt)
        plan = response.text
        
        # Store this plan in memory
        add_to_memory({
            "type": "research_plan",
            "content": plan,
            "query": query,
            "source": "generated_plan"
        })
        
        return plan
    except Exception as e:
        logger.error(f"Error generating research plan: {e}")
        return f"Error generating research plan: {str(e)}"

# Main application functions
def run_autonomous_research(query, depth=2):
    """Run the full research process autonomously"""
    # Step 1: Check if we already have research on this query
    existing_memories = retrieve_memory(query)
    relevant_summaries = [m for m in existing_memories if m["metadata"].get("type") == "research_summary"]
    
    # If we have existing research, we'll include it
    if relevant_summaries:
        st.info("📚 Found existing research on this topic in memory")
    
    # Step 2: Generate a research plan
    with st.spinner("🧠 Generating research plan..."):
        plan = get_research_plan(query)
        st.write("### Research Plan")
        st.write(plan)
    
    # Step 3: Search for relevant content using Semantic Scholar
    with st.spinner("🔍 Searching for relevant academic papers..."):
        papers = search_semantic_scholar(query, limit=depth * 3)
        search_results = format_semantic_scholar_results(papers)
        
        st.write(f"### Found {len(search_results)} Relevant Papers")
        for i, result in enumerate(search_results):
            authors = result.get('authors', [])
            authors_str = ", ".join(authors[:3])
            if len(authors) > 3:
                authors_str += " et al."
                
            st.write(f"**{i+1}. {result['title']}** ({result.get('year', 'n/a')})")
            st.write(f"_{authors_str}_ | {result.get('venue', 'Unknown venue')}")
            st.write(f"URL: {result['url']}")
            abstract = result.get('abstract')
            if abstract:
                st.write(abstract[:300] + ("..." if len(abstract) > 300 else ""))
            else:
                st.write("_No abstract available._")

    
    # Step 4: Fetch and process content
    documents = []
    with st.spinner("📄 Retrieving and processing documents..."):
        for result in search_results:
            try:
                # First try to fetch content from URL if available
                if result["url"]:
                    content = fetch_content_with_rate_limit(result["url"])
                    if not content or "Error fetching content" in content:
                        # If error or no content, use abstract as fallback
                        content = result.get("abstract", "No content available")
                else:
                    # If no URL, use abstract
                    content = result.get("abstract", "No content available")
                
                processed_text = clean_text(content)
                
                # Add to database with citation info
                doc_ids = add_document_with_citation(processed_text, result)
                
                # Get the document chunks we just added
                client = get_chroma_client()
                docs_collection = client.get_collection("research_documents")
                added_docs = docs_collection.get(ids=doc_ids)
                
                for i, doc_text in enumerate(added_docs["documents"]):
                    if i < len(added_docs["metadatas"]):
                        documents.append({
                            "content": doc_text,
                            "metadata": added_docs["metadatas"][i]
                        })
            except Exception as e:
                logger.error(f"Error processing document {result['title']}: {e}")
                st.error(f"Error processing document: {result['title']}")
    
    # Step 5: Generate summary with citations
    with st.spinner("✍️ Synthesizing information and generating summary..."):
        # Combine new documents with any existing relevant research
        if relevant_summaries:
            for summary in relevant_summaries[:2]:  # Include top 2 existing summaries
                documents.append({
                    "content": summary["content"],
                    "metadata": {
                        "title": "Previous Research Summary",
                        "url": "",
                        "source": "Agent Memory",
                        "authors": ""
                    }
                })
        
        summary = generate_research_summary(query, documents)
        st.write("### Research Summary")
        st.markdown(summary)
    
    # Step 6: Generate follow-up questions
    with st.spinner("🤔 Generating follow-up questions..."):
        follow_up = generate_follow_up_questions(query, summary)
        st.write("### Suggested Follow-up Questions")
        st.markdown(follow_up)
    
    # Return everything for the session state
    return {
        "plan": plan,
        "sources": search_results,
        "summary": summary,
        "follow_up": follow_up
    }

def display_memory_explorer():
    """Display and allow exploration of agent's memory"""
    st.write("## 🧠 Memory Explorer")
    
    memories = retrieve_memory()
    
    # Group memories by type
    memory_types = {}
    for memory in memories:
        mem_type = memory["metadata"].get("type", "unknown")
        if mem_type not in memory_types:
            memory_types[mem_type] = []
        memory_types[mem_type].append(memory)
    
    # Display counts
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Research Summaries", len(memory_types.get("research_summary", [])))
    with col2:
        st.metric("Research Plans", len(memory_types.get("research_plan", [])))
    with col3:
        st.metric("Total Memory Items", len(memories))
    
    # Create a dataframe for easy exploration
    memory_data = []
    for memory in memories:
        memory_data.append({
            "Type": memory["metadata"].get("type", "unknown"),
            "Timestamp": memory["metadata"].get("timestamp", "unknown"),
            "Query": memory["metadata"].get("query", ""),
            "Preview": memory["content"][:100] + "...",
            "Full Content": memory["content"]
        })
    
    if memory_data:
        df = pd.DataFrame(memory_data)
        
        # Allow filtering by type
        types = ["All"] + list(set([m["Type"] for m in memory_data]))
        selected_type = st.selectbox("Filter by type", types)
        
        # Add search box
        search_query = st.text_input("Search in memory:", "")
        
        if selected_type != "All":
            filtered_df = df[df["Type"] == selected_type]
        else:
            filtered_df = df
        
        # Apply search filter if provided
        if search_query:
            filtered_df = filtered_df[
                filtered_df["Full Content"].str.contains(search_query, case=False) | 
                filtered_df["Query"].str.contains(search_query, case=False)
            ]
        
        # Display as an interactive table
        if not filtered_df.empty:
            st.dataframe(
                filtered_df[["Type", "Timestamp", "Query", "Preview"]],
                use_container_width=True
            )
            
            # Allow viewing full content of selected memory
            selected_indices = st.multiselect(
                "Select memory items to view in full:",
                filtered_df.index
            )
            
            for idx in selected_indices:
                with st.expander(f"{filtered_df.iloc[idx]['Type']} - {filtered_df.iloc[idx]['Query']}"):
                    st.markdown(filtered_df.iloc[idx]["Full Content"])
        else:
            st.write("No memory items match the current filters.")
    else:
        st.write("No memories stored yet. Run some research queries to populate memory.")

def display_citation_explorer():
    """Display and allow exploration of stored citations and documents"""
    st.write("## 📚 Citation Database Explorer")
    
    client = get_chroma_client()
    try:
        docs_collection = client.get_collection("research_documents")
        all_docs = docs_collection.get()
        
        if not all_docs["ids"]:
            st.write("No documents in citation database yet. Run some research queries to populate.")
            return
        
        # Extract sources and years
        sources = set()
        years = set()
        for metadata in all_docs["metadatas"]:
            sources.add(metadata.get("source", "Unknown"))
            if metadata.get("year"):
                years.add(metadata.get("year"))
        
        # Display metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Document Chunks", len(all_docs["ids"]))
        with col2:
            st.metric("Unique Sources", len(sources))
        with col3:
            st.metric("Publication Years", len(years))
        
        # Create filters
        col1, col2 = st.columns(2)
        with col1:
            selected_source = st.selectbox("Filter by source", ["All"] + sorted(list(sources)))
        with col2:
            selected_year = st.selectbox("Filter by year", ["All"] + sorted(list(years), reverse=True))
        
        # Create a search box
        search_term = st.text_input("Search within documents:")
        
        # Prepare data for display
        doc_data = []
        for i, doc_id in enumerate(all_docs["ids"]):
            if i < len(all_docs["metadatas"]):
                metadata = all_docs["metadatas"][i]
                doc_data.append({
                    "ID": doc_id,
                    "Title": metadata.get("title", "Unknown"),
                    "Authors": metadata.get("authors", "Unknown"),
                    "Year": metadata.get("year", "Unknown"),
                    "Source": metadata.get("source", "Unknown"),
                    "Venue": metadata.get("venue", "Unknown"),
                    "URL": metadata.get("url", ""),
                    "Added": metadata.get("timestamp", "Unknown"),
                    "Preview": all_docs["documents"][i][:100] + "...",
                    "Full Content": all_docs["documents"][i]
                })
        
        # Convert to dataframe for easier filtering
        df = pd.DataFrame(doc_data)
        
        # Apply filters
        if selected_source != "All":
            df = df[df["Source"] == selected_source]
        
        if selected_year != "All":
            df = df[df["Year"] == selected_year]
        
        if search_term:
            df = df[df["Full Content"].str.contains(search_term, case=False) | 
                    df["Title"].str.contains(search_term, case=False) |
                    df["Authors"].str.contains(search_term, case=False)]
        
        if not df.empty:
            st.dataframe(
                df[["Title", "Authors", "Year", "Source", "Preview"]],
                use_container_width=True
            )
            
            # Allow viewing full document
            selected_indices = st.multiselect(
                "Select documents to view in full:",
                df.index
            )
            
            for idx in selected_indices:
                with st.expander(f"{df.iloc[idx]['Title']} ({df.iloc[idx]['Year']})"):
                    st.markdown(df.iloc[idx]["Full Content"])
                    st.markdown(f"**Source:** [{df.iloc[idx]['Source']}]({df.iloc[idx]['URL']})")
                    st.markdown(f"**Authors:** {df.iloc[idx]['Authors']}")
                    if df.iloc[idx]['Venue'] != "Unknown":
                        st.markdown(f"**Venue:** {df.iloc[idx]['Venue']}")
        else:
            st.write("No documents match the current filters.")
    except Exception as e:
        st.error(f"Error accessing citation database: {e}")

def export_citation_list():
    """Export citations in various formats"""
    st.write("## 📑 Export Citations")
    
    client = get_chroma_client()
    docs_collection = client.get_collection("research_documents")
    all_docs = docs_collection.get()
    
    if not all_docs["ids"]:
        st.write("No citations to export. Run some research queries first.")
        return
    
    # Get unique documents (by title)
    unique_docs = {}
    for i, doc_id in enumerate(all_docs["ids"]):
        if i < len(all_docs["metadatas"]):
            metadata = all_docs["metadatas"][i]
            title = metadata.get("title", "Unknown")
            
            # Only add each unique document once
            if title not in unique_docs and title != "Previous Research Summary":
                unique_docs[title] = {
                    "title": title,
                    "authors": metadata.get("authors", ""),
                    "year": metadata.get("year", ""),
                    "venue": metadata.get("venue", ""),
                    "url": metadata.get("url", ""),
                    "source": metadata.get("source", "")
                }
    
    # Format options
    citation_format = st.selectbox(
        "Select citation format:",
        ["APA", "MLA", "Chicago", "IEEE", "BibTeX"]
    )
    
    # Generate citations in the selected format
    citations = []
    for doc_title, doc_info in unique_docs.items():
        if citation_format == "APA":
            # Format: Author, A. A. (Year). Title. Venue. URL
            authors = doc_info["authors"]
            if authors:
                # Convert author string to APA format
                author_list = authors.split(", ")
                formatted_authors = ""
                for author in author_list:
                    name_parts = author.strip().split(" ")
                    if len(name_parts) > 1:
                        last_name = name_parts[-1]
                        initials = "".join([n[0] + "." for n in name_parts[:-1]])
                        formatted_authors += f"{last_name}, {initials}, "
                formatted_authors = formatted_authors.rstrip(", ")
            else:
                formatted_authors = "Unknown"
                
            year = f"({doc_info['year']}). " if doc_info['year'] else ""
            venue = f"{doc_info['venue']}. " if doc_info['venue'] else ""
            url = f"Retrieved from {doc_info['url']}" if doc_info['url'] else ""
            
            citation = f"{formatted_authors} {year}{doc_info['title']}. {venue}{url}"
            citations.append(citation)
            
        elif citation_format == "MLA":
            # Format: Author. "Title." Venue, Year. Web. URL
            authors = doc_info["authors"].replace(", ", " and ") if doc_info["authors"] else "Unknown"
            year = f", {doc_info['year']}" if doc_info['year'] else ""
            venue = f"{doc_info['venue']}" if doc_info['venue'] else "n.p."
            url = f" Web. {doc_info['url']}" if doc_info['url'] else ""
            
            citation = f"{authors}. \"{doc_info['title']}.\" {venue}{year}.{url}"
            citations.append(citation)
            
        elif citation_format == "Chicago":
            # Format: Author. Year. "Title." Venue. URL
            authors = doc_info["authors"] if doc_info["authors"] else "Unknown"
            year = f"{doc_info['year']}. " if doc_info['year'] else ""
            venue = f"{doc_info['venue']}. " if doc_info['venue'] else ""
            url = f"{doc_info['url']}" if doc_info['url'] else ""
            
            citation = f"{authors}. {year}\"{doc_info['title']}.\" {venue}{url}"
            citations.append(citation)
            
        elif citation_format == "IEEE":
            # Format: [#] Author, "Title," Venue, Year. URL
            authors = doc_info["authors"] if doc_info["authors"] else "Unknown"
            year = f", {doc_info['year']}" if doc_info['year'] else ""
            venue = f", {doc_info['venue']}" if doc_info['venue'] else ""
            url = f". [Online]. Available: {doc_info['url']}" if doc_info['url'] else ""
            
            citation = f"{authors}, \"{doc_info['title']}\"{venue}{year}{url}"
            citations.append(citation)
            
        elif citation_format == "BibTeX":
            # Format BibTeX entry
            key = "".join(doc_info['title'].split()[:2]).lower()
            if doc_info['year']:
                key += doc_info['year']
            else:
                key += "unknown"
                
            authors = doc_info["authors"].replace(", ", " and ") if doc_info["authors"] else "Unknown"
            year = f"{doc_info['year']}" if doc_info['year'] else ""
            venue = f"{doc_info['venue']}" if doc_info['venue'] else ""
            url = f"{doc_info['url']}" if doc_info['url'] else ""
            
            bibtex = f"""@article{{{key},
  author = {{{authors}}},
  title = {{{doc_info['title']}}},
  journal = {{{venue}}},
  year = {{{year}}},
  url = {{{url}}}
}}"""
            citations.append(bibtex)
    
    # Display citations
    if citations:
        st.write(f"### Citations ({len(citations)})")
        citation_text = ""
        
        if citation_format == "BibTeX":
            # For BibTeX, we just join with double newlines
            citation_text = "\n\n".join(citations)
        else:
            # For other formats, we number the citations
            for i, citation in enumerate(citations):
                citation_text += f"{i+1}. {citation}\n\n"
        
        st.text_area("Copy citations:", citation_text, height=300)
        
        # Allow downloading as a file
        file_extension = "bib" if citation_format == "BibTeX" else "txt"
        st.download_button(
            label=f"Download as {file_extension}",
            data=citation_text,
            file_name=f"citations_{citation_format.lower()}.{file_extension}",
            mime="text/plain"
        )
    else:
        st.write("No citations to display.")

def export_research_report(query=None):
    """Export research findings as a formatted report"""
    st.write("## 📊 Export Research Report")
    
    # Get recent research if query not specified
    if not query:
        # Get recent research summaries
        memories = retrieve_memory(memory_type="research_summary", limit=10)
        
        if not memories:
            st.write("No research summaries found. Run some research queries first.")
            return
        
        # Let user select which research to export
        summary_options = []
        for mem in memories:
            query_text = mem["metadata"].get("query", "Unknown query")
            timestamp = mem["metadata"].get("timestamp", "")
            summary_options.append(f"{query_text} ({timestamp})")
        
        selected_summary = st.selectbox("Select research to export:", summary_options)
        selected_index = summary_options.index(selected_summary)
        selected_memory = memories[selected_index]
        
        query = selected_memory["metadata"].get("query", "")
        summary_content = selected_memory["content"]
    else:
        # Get the most recent summary for this query
        memories = retrieve_memory(query=query, memory_type="research_summary", limit=1)
        if memories:
            summary_content = memories[0]["content"]
        else:
            st.write(f"No research summary found for query: {query}")
            return
    
    # Format options
    report_format = st.selectbox(
        "Select report format:",
        ["Markdown", "PDF"]
    )
    
    # Get related citations
    client = get_chroma_client()
    docs_collection = client.get_collection("research_documents")
    
    # Search for documents related to this query
    results = docs_collection.query(
        query_texts=[query],
        n_results=10
    )
    
    # Extract citation information
    citations = []
    seen_titles = set()
    for i, doc in enumerate(results["documents"][0]):
        metadata = results["metadatas"][0][i]
        title = metadata.get("title", "")
        
        # Skip duplicates
        if title in seen_titles or title == "Previous Research Summary":
            continue
        
        seen_titles.add(title)
        
        # Format citation in APA style for the report
        authors = metadata.get("authors", "Unknown")
        year = f"({metadata.get('year', 'n.d.')}). " if metadata.get('year') else "(n.d.). "
        venue = f"{metadata.get('venue', '')}. " if metadata.get('venue') else ""
        url = f"Retrieved from {metadata.get('url', '')}" if metadata.get('url') else ""
        
        citation = f"{authors} {year}{title}. {venue}{url}"
        citations.append(citation)
    
    # Generate report content
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    
    report_content = f"""# Research Report: {query}
    
Generated on: {now}

## Summary

{summary_content}

## References

"""
    
    for i, citation in enumerate(citations):
        report_content += f"{i+1}. {citation}\n\n"
    
    # Display preview
    st.write("### Report Preview")
    st.markdown(report_content)
    
    # Download options
    if report_format == "Markdown":
        st.download_button(
            label="Download Markdown Report",
            data=report_content,
            file_name=f"research_report_{query[:20].replace(' ', '_')}.md",
            mime="text/markdown"
        )
    elif report_format == "PDF":
        st.warning("PDF export would require additional libraries like ReportLab or pdfkit. Implementation placeholder.")
        # In a real implementation, you would convert markdown to PDF here

# Sidebar configuration
def configure_sidebar():
    """Configure the application sidebar"""
    with st.sidebar:
        st.title("🧠 Research Agent")
        st.write("An autonomous research assistant powered by AI")
        
        # API keys
        st.subheader("API Configuration")
        
        # Gemini API Key
        gemini_api_key = st.text_input(
            "Gemini API Key", 
            value=st.session_state.get("GEMINI_API_KEY", ""),
            type="password"
        )
        if gemini_api_key:
            st.session_state.GEMINI_API_KEY = gemini_api_key
        
        # Mem0 API Key
        mem0_api_key = st.text_input(
            "Mem0 API Key (Optional)",
            value=st.session_state.get("MEM0_API_KEY", ""),
            type="password"
        )
        if mem0_api_key:
            st.session_state.MEM0_API_KEY = mem0_api_key
        
        # Settings
        st.subheader("Settings")
        st.session_state.research_depth = st.slider(
            "Research Depth",
            min_value=1,
            max_value=5,
            value=st.session_state.get("research_depth", 3),
            help="Higher values mean more sources but slower research"
        )
        
        # Navigation
        st.subheader("Navigation")
        page = st.radio(
            "Select Page",
            ["Research", "Memory Explorer", "Citation Database", "Export"]
        )
        
        # Add info section
        st.info(
            """
            This application uses:
            - Gemini for summarization
            - Semantic Scholar API for research
            - ChromaDB for document storage
            - Mem0 for memory
            """
        )
        
        return page

# Main application
def main():
    """Main application entry point"""
    # Initialize session state
    if "research_history" not in st.session_state:
        st.session_state.research_history = []
    
    if "GEMINI_API_KEY" not in st.session_state:
        st.session_state.GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
    
    if "MEM0_API_KEY" not in st.session_state:
        st.session_state.MEM0_API_KEY = os.environ.get("MEM0_API_KEY", "")
    
    # Configure the sidebar and get selected page
    page = configure_sidebar()
    
    # Check if Gemini API key is provided
    if not st.session_state.GEMINI_API_KEY:
        st.error("⚠️ Please provide your Gemini API key in the sidebar.")
        return
    
    # Display different pages based on selection
    if page == "Research":
        st.title("🔍 Research Assistant")
        
        with st.container():
            research_query = st.text_area(
                "What would you like me to research?",
                height=100,
                placeholder="Enter a research question, topic, or area of interest..."
            )
            
            col1, col2 = st.columns([1, 3])
            with col1:
                submit_button = st.button("Start Research", type="primary")
            with col2:
                if st.session_state.research_history:
                    clear_history = st.button("Clear Research History")
                    if clear_history:
                        st.session_state.research_history = []
                        st.rerun()
        
        if submit_button and research_query:
            try:
                with st.spinner("🧠 Researching... this may take a few minutes"):
                    # Run the autonomous research process
                    research_results = run_autonomous_research(
                        research_query, 
                        depth=st.session_state.research_depth
                    )
                    
                    # Add to history
                    st.session_state.research_history.append({
                        "query": research_query,
                        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
                        "results": research_results
                    })
            except Exception as e:
                st.error(f"An error occurred during research: {str(e)}")
                logger.error(f"Research error: {e}", exc_info=True)
        
        # Display research history
        if st.session_state.research_history:
            st.write("## Previous Research")
            
            for i, research_item in enumerate(reversed(st.session_state.research_history)):
                with st.expander(f"Research: {research_item['query']} ({research_item['timestamp']})"):
                    st.write(f"### Summary")
                    st.markdown(research_item['results']['summary'])
                    
                    st.write(f"### Follow-up Questions")
                    st.markdown(research_item['results']['follow_up'])
                    
                    st.write(f"### Research Plan")
                    st.markdown(research_item['results']['plan'])
                    
                    if st.button(f"Export this research", key=f"export_{i}"):
                        export_research_report(research_item['query'])
    
    elif page == "Memory Explorer":
        display_memory_explorer()
    
    elif page == "Citation Database":
        display_citation_explorer()
    
    elif page == "Export":
        st.title("📊 Export Options")
        
        export_tab1, export_tab2 = st.tabs(["Research Reports", "Citations"])
        
        with export_tab1:
            export_research_report()
        
        with export_tab2:
            export_citation_list()

# Run the main application
if __name__ == "__main__":
    main()