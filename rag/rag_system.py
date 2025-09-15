"""
RAG (Retrieval-Augmented Generation) System
Integrates knowledge base with course materials and implements vector database for context retrieval.
"""

# Type ignore for FAISS library compatibility
# type: ignore

import os
import json
import pickle
from typing import Dict, List, Any, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from pathlib import Path
import logging

import config

class RAGSystem:
    """
    Retrieval-Augmented Generation system for enhancing lecture summaries
    with relevant course materials and academic knowledge.
    """
    
    def __init__(self):
        self.logger = logging.getLogger("RAGSystem")
        self.embedding_model = None
        self.vector_index = None
        self.document_store = {}
        self.knowledge_base_loaded = False
        
        self._initialize_system()
    
    def _initialize_system(self):
        """Initialize the RAG system components."""
        try:
            # Load embedding model
            self.logger.info(f"Loading embedding model: {config.EMBEDDING_MODEL}")
            self.embedding_model = SentenceTransformer(config.EMBEDDING_MODEL)
            
            # Create directories
            os.makedirs(config.VECTOR_DB_PATH, exist_ok=True)
            os.makedirs(config.KNOWLEDGE_BASE_PATH, exist_ok=True)
            
            # Load or create vector database
            self._load_or_create_vector_db()
            
            # Load knowledge base
            self._load_knowledge_base()
            
            self.logger.info("RAG system initialized successfully")
            
            # Create directories
            os.makedirs(config.VECTOR_DB_PATH, exist_ok=True)
            os.makedirs(config.KNOWLEDGE_BASE_PATH, exist_ok=True)
            
            # Load or create vector database
            self._load_or_create_vector_db()
            
            # Load knowledge base
            self._load_knowledge_base()
            
            self.logger.info("RAG system initialized successfully")
            
        except Exception as e:
            self.logger.error(f"RAG system initialization failed: {e}")
            raise
    
    def _load_or_create_vector_db(self):
        """Load existing vector database or create a new one."""
        index_path = os.path.join(config.VECTOR_DB_PATH, "faiss_index.index")
        store_path = os.path.join(config.VECTOR_DB_PATH, "document_store.pkl")
        
        if os.path.exists(index_path) and os.path.exists(store_path):
            # Load existing database
            self.logger.info("Loading existing vector database...")
            try:
                self.vector_index = faiss.read_index(index_path)
                
                with open(store_path, 'rb') as f:
                    self.document_store = pickle.load(f)
                
                self.logger.info(f"Vector database loaded: {self.vector_index.ntotal} documents")
            except Exception as e:
                self.logger.warning(f"Failed to load existing database: {e}. Creating new one.")
                self._create_new_vector_db()
        else:
            # Create new database
            self._create_new_vector_db()
            
    def _create_new_vector_db(self):
        """Create a new vector database with proper dimensions."""
        self.logger.info("Creating new vector database...")
        
        # Get embedding dimension from the model
        try:
            if self.embedding_model is None:
                raise ValueError("Embedding model not initialized")
            test_embedding = self.embedding_model.encode(["test"])
            dimension = test_embedding.shape[1]
            self.logger.info(f"Using embedding dimension: {dimension}")
        except Exception as e:
            self.logger.error(f"Failed to determine embedding dimension: {e}")
            # Fallback dimensions for common models
            if "mpnet" in config.EMBEDDING_MODEL.lower():
                dimension = 768
            elif "minilm" in config.EMBEDDING_MODEL.lower():
                dimension = 384
            else:
                dimension = 768  # Default fallback
            self.logger.info(f"Using fallback dimension: {dimension}")
        
        self.vector_index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        self.document_store = {}
        
        # Initialize with sample academic content
        self._initialize_sample_knowledge_base()
    
    def _initialize_sample_knowledge_base(self):
        """Initialize with sample academic content for demonstration."""
        sample_documents = [
            {
                "id": "math_001",
                "title": "Calculus - Derivatives",
                "content": "A derivative measures the rate of change of a function. For a function f(x), the derivative f'(x) represents how f changes with respect to x. Common rules include the power rule: d/dx(x^n) = nx^(n-1), and the chain rule for composite functions.",
                "subject": "mathematics",
                "type": "concept"
            },
            {
                "id": "math_002", 
                "title": "Linear Algebra - Matrices",
                "content": "A matrix is a rectangular array of numbers arranged in rows and columns. Matrix operations include addition, multiplication, and finding determinants. Eigenvalues and eigenvectors are fundamental concepts in linear algebra with applications in data science and physics.",
                "subject": "mathematics",
                "type": "definition"
            },
            {
                "id": "cs_001",
                "title": "Algorithms - Complexity Analysis",
                "content": "Time complexity describes how algorithm runtime scales with input size. Big O notation provides upper bounds: O(1) constant time, O(log n) logarithmic, O(n) linear, O(n²) quadratic. Space complexity measures memory usage patterns.",
                "subject": "computer_science",
                "type": "concept"
            },
            {
                "id": "cs_002",
                "title": "Data Structures - Binary Trees",
                "content": "A binary tree is a hierarchical data structure where each node has at most two children. Binary search trees maintain ordering properties for efficient search, insertion, and deletion operations. Tree traversal methods include in-order, pre-order, and post-order.",
                "subject": "computer_science", 
                "type": "definition"
            },
            {
                "id": "physics_001",
                "title": "Classical Mechanics - Newton's Laws",
                "content": "Newton's first law: objects at rest stay at rest unless acted upon by force. Second law: F = ma, force equals mass times acceleration. Third law: for every action there is an equal and opposite reaction. These laws form the foundation of classical mechanics.",
                "subject": "physics",
                "type": "law"
            },
            {
                "id": "physics_002",
                "title": "Thermodynamics - Energy Conservation",
                "content": "The first law of thermodynamics states that energy cannot be created or destroyed, only transformed. Internal energy change equals heat added minus work done: ΔU = Q - W. This principle applies to all thermodynamic processes.",
                "subject": "physics",
                "type": "principle"
            },
            {
                "id": "bio_001",
                "title": "Cell Biology - DNA Structure",
                "content": "DNA is a double helix composed of nucleotides containing four bases: adenine (A), thymine (T), guanine (G), and cytosine (C). Base pairing rules: A pairs with T, G pairs with C. DNA stores genetic information and serves as template for RNA synthesis.",
                "subject": "biology",
                "type": "structure"
            },
            {
                "id": "chem_001", 
                "title": "Chemistry - Chemical Bonding",
                "content": "Chemical bonds form when atoms share or transfer electrons. Ionic bonds occur between metals and nonmetals through electron transfer. Covalent bonds involve electron sharing between nonmetals. Bond strength affects molecular properties and reactivity.",
                "subject": "chemistry",
                "type": "concept"
            }
        ]
        
        # Add documents to the vector database
        for doc in sample_documents:
            self.add_document(doc)
        
        self.logger.info(f"Initialized knowledge base with {len(sample_documents)} sample documents")
    
    def add_document(self, document: Dict[str, Any]) -> bool:
        """Add a document to the knowledge base and vector index."""
        try:
            doc_id = document["id"]
            content = document["content"]
            
            # Skip empty content
            if not content or not content.strip():
                self.logger.warning(f"Skipping document {doc_id}: empty content")
                return False
            
            # Generate embedding
            if self.embedding_model is None:
                self.logger.error("Embedding model not initialized")
                return False
            embedding = self.embedding_model.encode([content])
            
            # Validate embedding
            if embedding is None or len(embedding) == 0:
                self.logger.warning(f"Skipping document {doc_id}: failed to generate embedding")
                return False
            
            # Normalize for cosine similarity
            embedding = embedding / np.linalg.norm(embedding, axis=1, keepdims=True)
            
            # Add to vector index
            if self.vector_index is None:
                self.logger.error("Vector index not initialized")
                return False
            self.vector_index.add(embedding.astype('float32'))  # type: ignore
            
            # Store document metadata
            self.document_store[self.vector_index.ntotal - 1] = document
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to add document {document.get('id', 'unknown')}: {str(e)}")
            return False
    
    def save_vector_db(self):
        """Save the vector database to disk."""
        try:
            index_path = os.path.join(config.VECTOR_DB_PATH, "faiss_index.index")
            store_path = os.path.join(config.VECTOR_DB_PATH, "document_store.pkl")
            
            # Save FAISS index
            faiss.write_index(self.vector_index, index_path)
            
            # Save document store
            with open(store_path, 'wb') as f:
                pickle.dump(self.document_store, f)
            
            self.logger.info("Vector database saved successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save vector database: {e}")
            return False
    
    def retrieve_context(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve relevant context documents for a given query.
        
        Args:
            query: Search query text
            k: Number of documents to retrieve
            
        Returns:
            List of relevant documents with similarity scores
        """
        try:
            if not query.strip():
                return []
            
            # Generate query embedding
            if self.embedding_model is None:
                self.logger.error("Embedding model not initialized")
                return []
            query_embedding = self.embedding_model.encode([query])
            query_embedding = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)
            
            # Search vector index
            if self.vector_index is None:
                self.logger.error("Vector index not initialized")
                return []
            similarities, indices = self.vector_index.search(query_embedding.astype('float32'), k)  # type: ignore
            
            # Retrieve documents
            retrieved_docs = []
            for i, (score, idx) in enumerate(zip(similarities[0], indices[0])):
                if idx != -1 and idx in self.document_store:  # Valid index
                    doc = self.document_store[idx].copy()
                    doc['similarity_score'] = float(score)
                    doc['rank'] = i + 1
                    retrieved_docs.append(doc)
            
            self.logger.info(f"Retrieved {len(retrieved_docs)} documents for query: '{query[:50]}...'")
            return retrieved_docs
            
        except Exception as e:
            self.logger.error(f"Context retrieval failed: {e}")
            return []
    
    def enhance_concepts(self, concepts: List[Dict[str, Any]], 
                        context_docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Enhance extracted concepts with relevant context from knowledge base.
        
        Args:
            concepts: List of extracted concepts
            context_docs: Retrieved context documents
            
        Returns:
            Enhanced concepts with additional context
        """
        enhanced_concepts = []
        
        for concept in concepts:
            enhanced_concept = concept.copy()
            concept_text = concept.get("text", "").lower()
            
            # Find relevant context for this concept
            relevant_context = []
            for doc in context_docs:
                doc_content = doc.get("content", "").lower()
                doc_title = doc.get("title", "").lower()
                
                # Check for relevance
                concept_words = set(concept_text.split())
                doc_words = set(doc_content.split())
                title_words = set(doc_title.split())
                
                # Calculate overlap
                content_overlap = len(concept_words.intersection(doc_words))
                title_overlap = len(concept_words.intersection(title_words))
                
                if content_overlap >= 2 or title_overlap >= 1:
                    relevant_context.append({
                        "source": doc.get("title", ""),
                        "content": doc.get("content", "")[:200] + "...",  # Truncate
                        "subject": doc.get("subject", ""),
                        "similarity": doc.get("similarity_score", 0.0)
                    })
            
            # Add context to concept
            if relevant_context:
                enhanced_concept["related_context"] = relevant_context[:2]  # Max 2 contexts per concept
                enhanced_concept["enhanced"] = True
                
                # Increase importance if context is found
                current_importance = enhanced_concept.get("importance", 0.5)
                enhanced_concept["importance"] = min(current_importance + 0.1, 1.0)
            
            enhanced_concepts.append(enhanced_concept)
        
        return enhanced_concepts
    
    def _load_knowledge_base(self):
        """Load additional knowledge base files if available."""
        knowledge_files = list(Path(config.KNOWLEDGE_BASE_PATH).glob("*.json"))
        
        for file_path in knowledge_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                if isinstance(data, list):
                    for doc in data:
                        if self._validate_document(doc):
                            self.add_document(doc)
                elif isinstance(data, dict) and self._validate_document(data):
                    self.add_document(data)
                
                self.logger.info(f"Loaded knowledge base file: {file_path.name}")
                
            except Exception as e:
                self.logger.warning(f"Failed to load knowledge base file {file_path}: {e}")
        
        if knowledge_files:
            self.knowledge_base_loaded = True
            # Save updated vector database
            self.save_vector_db()
    
    def _validate_document(self, doc: Dict[str, Any]) -> bool:
        """Validate document structure."""
        required_fields = ["id", "content"]
        return all(field in doc for field in required_fields)
    
    def add_course_materials(self, course_data: Dict[str, Any]) -> bool:
        """Add course-specific materials to the knowledge base."""
        try:
            course_id = course_data.get("course_id", "unknown")
            materials = course_data.get("materials", [])
            
            added_count = 0
            for material in materials:
                # Create document from course material
                doc = {
                    "id": f"{course_id}_{material.get('id', added_count)}",
                    "title": material.get("title", "Course Material"),
                    "content": material.get("content", ""),
                    "subject": course_data.get("subject", "general"),
                    "type": material.get("type", "course_material"),
                    "course": course_id
                }
                
                if self.add_document(doc):
                    added_count += 1
            
            # Save updated database
            self.save_vector_db()
            
            self.logger.info(f"Added {added_count} course materials for {course_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to add course materials: {e}")
            return False
    
    def search_by_subject(self, subject: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search documents by subject area."""
        matching_docs = []
        
        for doc in self.document_store.values():
            if doc.get("subject", "").lower() == subject.lower():
                matching_docs.append(doc)
        
        return matching_docs[:limit]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the knowledge base."""
        subjects = {}
        types = {}
        
        for doc in self.document_store.values():
            subject = doc.get("subject", "unknown")
            doc_type = doc.get("type", "unknown")
            
            subjects[subject] = subjects.get(subject, 0) + 1
            types[doc_type] = types.get(doc_type, 0) + 1
        
        embedding_dim = 0
        if self.embedding_model is not None:
            try:
                embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
            except AttributeError:
                # Fallback for models without this method
                test_embedding = self.embedding_model.encode(["test"])
                embedding_dim = test_embedding.shape[1] if test_embedding is not None else 0
        
        return {
            "total_documents": len(self.document_store),
            "subjects": subjects,
            "document_types": types,
            "embedding_dimension": embedding_dim,
            "knowledge_base_loaded": self.knowledge_base_loaded
        }
    
    def create_sample_knowledge_files(self):
        """Create sample knowledge base files for testing."""
        sample_files = {
            "mathematics_concepts.json": [
                {
                    "id": "calc_limits",
                    "title": "Calculus - Limits",
                    "content": "A limit describes the behavior of a function as its input approaches a particular value. Limits are fundamental to calculus and are used to define derivatives and integrals. The limit of f(x) as x approaches a is denoted as lim(x→a) f(x).",
                    "subject": "mathematics",
                    "type": "definition"
                },
                {
                    "id": "algebra_vectors",
                    "title": "Linear Algebra - Vector Spaces",
                    "content": "A vector space is a collection of objects called vectors that can be added together and multiplied by scalars. Vector spaces must satisfy eight axioms including associativity, commutativity, and distributivity. Examples include Euclidean space and function spaces.",
                    "subject": "mathematics", 
                    "type": "concept"
                }
            ],
            "computer_science_fundamentals.json": [
                {
                    "id": "cs_recursion",
                    "title": "Recursion in Programming",
                    "content": "Recursion is a programming technique where a function calls itself to solve smaller instances of the same problem. Every recursive function needs a base case to stop recursion and a recursive case that reduces the problem size.",
                    "subject": "computer_science",
                    "type": "technique"
                }
            ]
        }
        
        try:
            for filename, data in sample_files.items():
                file_path = os.path.join(config.KNOWLEDGE_BASE_PATH, filename)
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                
                self.logger.info(f"Created sample knowledge file: {filename}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to create sample knowledge files: {e}")
            return False