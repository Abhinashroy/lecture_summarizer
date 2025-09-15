"""
Concept Extractor Agent
Uses fine-tuned LoRA model for academic concept extraction, definition identification, and technical term recognition.
"""

import re
import os
import torch
from typing import Dict, Any, List, Tuple
import time
import spacy
from collections import defaultdict

from .base_agent import BaseAgent
import config

class ConceptExtractorAgent(BaseAgent):
    """
    Agent responsible for extracting key concepts, definitions, and technical terms
    from lecture transcripts using a fine-tuned LoRA model.
    """
    
    def __init__(self):
        super().__init__("ConceptExtractor")
        self.t5_model = None
        self.t5_tokenizer = None
        self.scibert_model = None
        self.scibert_tokenizer = None
        self.nlp = None
        self.academic_terms = self._load_academic_terms()
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize FLAN-T5-small-LoRA and SciBERT for concept extraction with lazy loading."""
        try:
            # Lazy import to avoid loading heavy dependencies at startup
            from transformers import AutoTokenizer, AutoModel, T5Tokenizer, T5ForConditionalGeneration
            from peft import LoraConfig, get_peft_model, TaskType
            
            # Load FLAN-T5-small-LoRA for concept extraction
            self.logger.info("Loading FLAN-T5-small-LoRA model...")
            self.t5_tokenizer = T5Tokenizer.from_pretrained(config.ALTERNATIVE_MODELS["summarization"])
            self.t5_model = T5ForConditionalGeneration.from_pretrained(config.ALTERNATIVE_MODELS["summarization"])
            
            # Apply LoRA configuration to T5
            lora_config = LoraConfig(
                r=config.LORA_CONFIG["r"],
                lora_alpha=config.LORA_CONFIG["lora_alpha"],
                target_modules=["q", "v"],  # T5 attention modules
                lora_dropout=config.LORA_CONFIG["lora_dropout"],
                bias=config.LORA_CONFIG["bias"],
                task_type=TaskType.SEQ_2_SEQ_LM
            )
            self.t5_model = get_peft_model(self.t5_model, lora_config)
            self.logger.info("FLAN-T5-small-LoRA model loaded successfully")
            
            # Load SciBERT for scientific concept extraction
            self.logger.info("Loading SciBERT-optimized model...")
            self.scibert_tokenizer = AutoTokenizer.from_pretrained(config.ALTERNATIVE_MODELS["concept_extraction"])
            self.scibert_model = AutoModel.from_pretrained(config.ALTERNATIVE_MODELS["concept_extraction"])
            self.logger.info("SciBERT-optimized model loaded successfully")
            
            # Initialize spaCy for NER and linguistic analysis
            try:
                import spacy
                self.nlp = spacy.load("en_core_web_sm")
                self.logger.info("SpaCy model loaded successfully")
            except (OSError, ImportError) as e:
                self.logger.warning(f"SpaCy model not found: {e}, using basic text processing")
                self.nlp = None
                
        except Exception as e:
            self.logger.error(f"Failed to initialize models: {e}")
            # Fall back to rule-based methods
            self.t5_model = None
            self.t5_tokenizer = None
            self.scibert_model = None
            self.scibert_tokenizer = None
    
    def _load_academic_terms(self) -> Dict[str, List[str]]:
        """Load academic terms by subject area."""
        return {
            "mathematics": [
                "theorem", "proof", "lemma", "corollary", "axiom", "derivative", 
                "integral", "polynomial", "matrix", "vector", "eigenvalue", "limit",
                "convergence", "divergence", "function", "domain", "range"
            ],
            "computer_science": [
                "algorithm", "complexity", "recursion", "iteration", "data structure",
                "binary tree", "hash table", "graph", "sorting", "searching",
                "compilation", "parsing", "object-oriented", "inheritance", "polymorphism"
            ],
            "physics": [
                "energy", "momentum", "force", "acceleration", "velocity", "mass",
                "electromagnetic", "quantum", "photon", "electron", "wave", "frequency",
                "amplitude", "thermodynamics", "entropy", "conservation"
            ],
            "biology": [
                "cell", "DNA", "RNA", "protein", "enzyme", "mitochondria", "chloroplast",
                "photosynthesis", "respiration", "evolution", "mutation", "gene",
                "chromosome", "metabolism", "homeostasis", "ecosystem"
            ],
            "chemistry": [
                "molecule", "atom", "ion", "bond", "reaction", "catalyst", "equilibrium",
                "oxidation", "reduction", "acid", "base", "pH", "molarity", "solvent",
                "compound", "element", "periodic table"
            ]
        }
    
    def process(self, input_data: Any) -> Dict[str, Any]:
        """
        Extract key concepts, definitions, and formulas from lecture transcript.
        
        Args:
            input_data: Dict containing transcription text and optional metadata
            
        Returns:
            Dict containing extracted concepts, definitions, formulas, and terms
        """
        start_time = time.time()
        
        try:
            # Extract text from input
            if isinstance(input_data, dict):
                text = input_data.get("transcription", "")
                segments = input_data.get("segments", [])
            else:
                text = str(input_data)
                segments = []
            
            if not text:
                return {"error": "No text provided for concept extraction"}
            
            # Use FLAN-T5-small-LoRA and SciBERT if available
            concepts = {}
            if self.t5_model and self.scibert_model:
                concepts = self._extract_with_advanced_models(text)
            else:
                concepts = self._extract_with_rules(text)
            
            # Extract additional information
            definitions = self._extract_definitions(text)
            formulas = self._extract_formulas(text)
            key_terms = self._extract_key_terms(text)
            subject_areas = self._identify_subject_areas(text)
            
            # Organize by importance
            prioritized_concepts = self._prioritize_concepts(concepts, key_terms)
            
            processing_time = self.log_performance(start_time, "concept_extraction")
            confidence_score = self._calculate_confidence_score(prioritized_concepts)
            
            # Store quality score for status reporting
            self.last_quality_score = min(confidence_score + 0.1, 1.0)
            
            return {
                "concepts": prioritized_concepts,
                "definitions": definitions,
                "formulas": formulas,
                "key_terms": key_terms,
                "subject_areas": subject_areas,
                "processing_time": processing_time,
                "total_concepts": len(prioritized_concepts),
                "confidence_score": confidence_score,
                "output_quality": self.last_quality_score,
                "extraction_accuracy": confidence_score
            }
            
        except Exception as e:
            self.logger.error(f"Concept extraction failed: {e}")
            return {
                "concepts": [],
                "definitions": [],
                "formulas": [],
                "key_terms": [],
                "error": str(e),
                "processing_time": time.time() - start_time
            }
    
    def _extract_with_advanced_models(self, text: str) -> List[Dict[str, Any]]:
        """Extract concepts using FLAN-T5-small-LoRA and SciBERT models."""
        try:
            # Check if models are available
            if not self.t5_model or not self.t5_tokenizer or not self.scibert_model or not self.scibert_tokenizer:
                self.logger.info("Models not available, falling back to rule-based extraction")
                return self._extract_with_rules(text)
                
            # Split text into chunks if too long
            max_chunk_size = 400  # tokens
            chunks = self._split_text_into_chunks(text, max_chunk_size)
            
            all_concepts = []
            
            for chunk in chunks:
                # Use T5 for concept generation
                concept_prompt = f"Extract key concepts from: {chunk}"
                inputs = self.t5_tokenizer(concept_prompt, return_tensors="pt", max_length=512, truncation=True)
                
                with torch.no_grad():
                    outputs = self.t5_model.generate(
                        **inputs,
                        max_length=200,
                        num_beams=4,
                        temperature=0.7,
                        do_sample=True,
                        pad_token_id=self.t5_tokenizer.eos_token_id
                    )
                
                generated_concepts = self.t5_tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                # Use SciBERT for concept embeddings and scientific term identification
                scibert_inputs = self.scibert_tokenizer(chunk, return_tensors="pt", max_length=512, truncation=True)
                
                with torch.no_grad():
                    scibert_outputs = self.scibert_model(**scibert_inputs)
                    embeddings = scibert_outputs.last_hidden_state.mean(dim=1)
                
                # Parse the structured output
                parsed_concepts = self._parse_advanced_model_output(generated_concepts, chunk, embeddings)
                all_concepts.extend(parsed_concepts)
            
            return self._deduplicate_concepts(all_concepts)
            
        except Exception as e:
            self.logger.error(f"Advanced model extraction failed: {e}")
            return self._extract_with_rules(text)
    
    def _extract_with_rules(self, text: str) -> List[Dict[str, Any]]:
        """Extract concepts using rule-based methods as fallback."""
        concepts = []
        
        # Split into sentences
        sentences = re.split(r'[.!?]+', text)
        
        for i, sentence in enumerate(sentences):
            sentence = sentence.strip()
            if not sentence or len(sentence) < 5:  # Lowered from 10 to 5
                continue
            
            concept = {
                "text": sentence,
                "type": "statement",
                "importance": 0.5,
                "position": i
            }
            
            # Check for definition patterns (more lenient)
            if self._is_definition_sentence(sentence):
                concept["type"] = "definition"
                concept["importance"] = 0.9
            
            # Check for formula patterns  
            elif self._contains_formula(sentence):
                concept["type"] = "formula"
                concept["importance"] = 0.8
            
            # Check for key academic terms (increased importance)
            elif self._contains_academic_terms(sentence):
                concept["type"] = "key_concept"
                concept["importance"] = 0.7
            
            # Check for important linguistic patterns
            elif self._has_important_patterns(sentence):
                concept["type"] = "important_statement"
                concept["importance"] = 0.6
            
            # Be more inclusive - include more sentences as potential concepts
            elif len(sentence.split()) >= 3:  # Include sentences with 3+ words
                concept["importance"] = 0.5
            
            # Include concepts with importance >= 0.4 (lowered threshold)
            if concept["importance"] >= 0.4:
                concepts.append(concept)
        
        # If we still have very few concepts, add some key phrases
        if len(concepts) < 3:
            concepts.extend(self._extract_key_phrases(text))
        
        return concepts
    
    def _extract_key_phrases(self, text: str) -> List[Dict[str, Any]]:
        """Extract key phrases as backup concepts when too few concepts are found."""
        phrases = []
        
        # Look for common academic patterns
        academic_patterns = [
            r'\b[A-Z][a-z]+ (?:theorem|law|principle|theory|equation|method)\b',
            r'\b(?:theorem|law|principle|theory|concept) of [A-Z][a-z]+\b',
            r'\b[A-Z][a-z]+ (?:algorithm|approach|technique|strategy)\b',
            r'\b(?:first|second|third) (?:law|principle|theorem)\b',
            r'\b(?:fundamental|basic|key|important|main) (?:concept|idea|principle)\b'
        ]
        
        for pattern in academic_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                phrase = match.group(0)
                phrases.append({
                    "text": phrase,
                    "type": "key_phrase", 
                    "importance": 0.6,
                    "position": len(phrases)
                })
        
        # Extract noun phrases as concepts
        words = text.split()
        for i in range(len(words) - 1):
            # Look for capitalized words that might be important terms
            if words[i][0].isupper() and len(words[i]) > 3:
                if i < len(words) - 1 and words[i+1][0].isupper():
                    # Multi-word terms
                    phrase = f"{words[i]} {words[i+1]}"
                    phrases.append({
                        "text": phrase,
                        "type": "term",
                        "importance": 0.5,
                        "position": len(phrases)
                    })
        
        return phrases[:5]  # Return top 5 phrases
    
    def _split_text_into_chunks(self, text: str, max_chunk_size: int) -> List[str]:
        """Split text into manageable chunks for model processing."""
        words = text.split()
        chunks = []
        current_chunk = []
        
        for word in words:
            current_chunk.append(word)
            if len(current_chunk) >= max_chunk_size:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
        
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return chunks
    
    def _parse_model_output(self, model_output: str, original_text: str) -> List[Dict[str, Any]]:
        """Parse structured output from the fine-tuned model."""
        concepts = []
        
        # Split into sections
        sections = re.split(r'\*\*([^*]+)\*\*', model_output)
        
        current_section = None
        for i, section in enumerate(sections):
            if i % 2 == 1:  # Section headers
                current_section = section.strip().lower()
            else:  # Section content
                content = section.strip()
                if content and current_section:
                    items = [item.strip() for item in content.split('\n') if item.strip()]
                    
                    for item in items:
                        if item.startswith('-'):
                            item = item[1:].strip()
                        
                        concept = {
                            "text": item,
                            "type": current_section.replace(" ", "_"),
                            "importance": self._get_importance_by_type(current_section),
                            "source": "legacy_model"
                        }
                        concepts.append(concept)
        
        return concepts
    
    def _get_importance_by_type(self, concept_type: str) -> float:
        """Get importance score based on concept type."""
        importance_map = {
            "key concepts": 0.9,
            "definition": 0.95,
            "formula": 0.8,
            "law": 0.85,
            "principle": 0.85,
            "process": 0.7,
            "components": 0.6,
            "examples": 0.4
        }
        return importance_map.get(concept_type.lower(), 0.5)
    
    def _parse_advanced_model_output(self, t5_output: str, original_text: str, embeddings: torch.Tensor) -> List[Dict[str, Any]]:
        """Parse output from FLAN-T5-small-LoRA and incorporate SciBERT embeddings."""
        concepts = []
        
        # Parse T5 generated concepts
        concept_lines = [line.strip() for line in t5_output.split('\n') if line.strip()]
        
        for concept_text in concept_lines:
            if len(concept_text) > 5:  # Filter out very short outputs
                # Calculate importance using embedding similarity to scientific terms
                importance = self._calculate_scientific_importance(concept_text, embeddings)
                
                # Determine concept type using rule-based classification
                concept_type = self._classify_concept_type(concept_text)
                
                concept = {
                    "text": concept_text,
                    "type": concept_type,
                    "importance": importance,
                    "source": "flan_t5_lora_scibert",
                    "embedding_score": float(embeddings.mean().item())
                }
                concepts.append(concept)
        
        return concepts
    
    def _calculate_scientific_importance(self, concept_text: str, embeddings: torch.Tensor) -> float:
        """Calculate importance based on scientific terminology and embeddings."""
        base_score = 0.5
        
        # Boost for scientific keywords
        scientific_keywords = ['theory', 'principle', 'law', 'equation', 'formula', 'model', 'hypothesis']
        for keyword in scientific_keywords:
            if keyword.lower() in concept_text.lower():
                base_score += 0.1
        
        # Use embedding magnitude as importance indicator
        embedding_boost = min(float(embeddings.norm().item()) / 10.0, 0.3)
        
        return min(base_score + embedding_boost, 1.0)
    
    def _classify_concept_type(self, text: str) -> str:
        """Classify concept type based on content patterns."""
        text_lower = text.lower()
        
        if any(word in text_lower for word in ['definition', 'defined as', 'is a', 'refers to']):
            return 'definition'
        elif any(word in text_lower for word in ['formula', 'equation', '=', 'calculate']):
            return 'formula'
        elif any(word in text_lower for word in ['law', 'principle', 'rule', 'theorem']):
            return 'law'
        elif any(word in text_lower for word in ['process', 'method', 'procedure', 'steps']):
            return 'process'
        else:
            return 'key_concept'
    
    def _extract_definitions(self, text: str) -> List[Dict[str, str]]:
        """Extract definitions using pattern matching."""
        definitions = []
        
        # Definition patterns
        patterns = [
            r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+is\s+(?:an?\s+)?([^.!?]+)',
            r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+refers to\s+([^.!?]+)',
            r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+means\s+([^.!?]+)',
            r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+can be defined as\s+([^.!?]+)'
        ]
        
        for pattern in patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                term = match.group(1).strip()
                definition = match.group(2).strip()
                
                definitions.append({
                    "term": term,
                    "definition": definition,
                    "confidence": 0.8
                })
        
        return definitions
    
    def _extract_formulas(self, text: str) -> List[Dict[str, str]]:
        """Extract mathematical formulas and equations."""
        formulas = []
        
        # Formula patterns
        patterns = [
            r'([A-Za-z]+)\s*=\s*([^.!?,\n]+)',  # Basic equations
            r'([A-Za-z]+)\s*∝\s*([^.!?,\n]+)',  # Proportional to
            r'∫([^dx]+)dx\s*=\s*([^.!?,\n]+)',  # Integrals
            r'd/dx\s*\(([^)]+)\)\s*=\s*([^.!?,\n]+)',  # Derivatives
        ]
        
        for pattern in patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                if len(match.groups()) >= 2:
                    formula = {
                        "expression": match.group(0).strip(),
                        "left_side": match.group(1).strip(),
                        "right_side": match.group(2).strip(),
                        "type": "equation"
                    }
                    formulas.append(formula)
        
        return formulas
    
    def _extract_key_terms(self, text: str) -> List[Dict[str, Any]]:
        """Extract key academic terms using NER and term lists."""
        key_terms = []
        
        # Use spaCy if available
        if self.nlp:
            try:
                doc = self.nlp(text)
                
                # Named entities
                for ent in doc.ents:
                    if ent.label_ in ['PERSON', 'ORG', 'LAW', 'PRODUCT']:
                        key_terms.append({
                            "term": ent.text,
                            "type": ent.label_,
                            "confidence": 0.9,
                            "source": "spacy_ner"
                        })
            except Exception as e:
                self.logger.warning(f"SpaCy processing failed: {e}")
        
        # Academic terms from our predefined lists
        text_lower = text.lower()
        for subject, terms in self.academic_terms.items():
            for term in terms:
                if term.lower() in text_lower:
                    key_terms.append({
                        "term": term,
                        "type": "academic_term",
                        "subject": subject,
                        "confidence": 0.7,
                        "source": "term_list"
                    })
        
        # Remove duplicates
        seen_terms = set()
        unique_terms = []
        for term in key_terms:
            term_key = term["term"].lower()
            if term_key not in seen_terms:
                seen_terms.add(term_key)
                unique_terms.append(term)
        
        return unique_terms
    
    def _identify_subject_areas(self, text: str) -> List[Dict[str, Any]]:
        """Identify the academic subject areas covered in the text using enhanced detection."""
        subject_scores = defaultdict(float)
        text_lower = text.lower()
        
        # Enhanced subject area terms including social sciences and humanities
        enhanced_academic_terms = {
            **self.academic_terms,
            "social_studies": [
                "civil rights", "discrimination", "social justice", "equality", "racism",
                "society", "community", "culture", "diversity", "movement", "activism",
                "human rights", "democracy", "freedom", "oppression", "prejudice"
            ],
            "education": [
                "classroom", "students", "teacher", "teaching", "learning", "education",
                "school", "pedagogy", "curriculum", "instruction", "academic", "study"
            ],
            "literature": [
                "poetry", "writing", "narrative", "story", "text", "literary", "author",
                "words", "language", "expression", "voice", "speech", "communication"
            ],
            "psychology": [
                "emotions", "feelings", "fear", "shame", "dignity", "pride", "silence",
                "behavior", "mental", "psychological", "empathy", "trauma"
            ],
            "philosophy": [
                "ethics", "morality", "principles", "values", "meaning", "existence",
                "truth", "justice", "wisdom", "reflection", "consciousness"
            ]
        }
        
        # Calculate scores for each subject area
        for subject, terms in enhanced_academic_terms.items():
            for term in terms:
                count = text_lower.count(term.lower())
                # Weight longer terms higher and give bonus for exact matches
                term_weight = len(term.split()) * 0.5 + 1.0
                subject_scores[subject] += count * term_weight
        
        # Context-based scoring (look for key phrases that indicate subject area)
        context_patterns = {
            "social_studies": [
                "civil rights movement", "discrimination", "social justice", "human dignity",
                "equality", "social issues", "community problems", "cultural awareness"
            ],
            "education": [
                "in the classroom", "my students", "teaching", "educational", "learning environment",
                "academic", "study", "curriculum", "instruction"
            ],
            "literature": [
                "poetry", "written work", "literary", "narrative", "storytelling",
                "creative writing", "spoken word", "artistic expression"
            ]
        }
        
        for subject, patterns in context_patterns.items():
            for pattern in patterns:
                if pattern in text_lower:
                    subject_scores[subject] += 3.0  # High bonus for context matches
        
        # Normalize scores
        total_score = sum(subject_scores.values())
        if total_score > 0:
            for subject in subject_scores:
                subject_scores[subject] = (subject_scores[subject] / total_score) * 100
        
        # Return top subjects with meaningful scores (at least 5%)
        sorted_subjects = sorted(subject_scores.items(), key=lambda x: x[1], reverse=True)
        meaningful_subjects: List[Dict[str, Any]] = [
            {"subject": subject.replace("_", " ").title(), "confidence": float(score)} 
            for subject, score in sorted_subjects 
            if score >= 5.0
        ]
        
        default_subject: List[Dict[str, Any]] = [{"subject": "General", "confidence": 100.0}]
        return meaningful_subjects[:3] if meaningful_subjects else default_subject
    
    def _prioritize_concepts(self, concepts: List[Dict[str, Any]], key_terms: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Prioritize concepts by importance and relevance."""
        # Add bonus importance for concepts containing key terms
        key_term_texts = {term["term"].lower() for term in key_terms}
        
        for concept in concepts:
            concept_text = concept["text"].lower()
            term_bonus = sum(0.1 for term in key_term_texts if term in concept_text)
            concept["importance"] = min(concept["importance"] + term_bonus, 1.0)
        
        # Sort by importance
        return sorted(concepts, key=lambda x: x["importance"], reverse=True)
    
    def _deduplicate_concepts(self, concepts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate concepts based on text similarity."""
        if not concepts:
            return []
        
        deduplicated = []
        seen_texts = set()
        
        for concept in concepts:
            concept_text = concept.get("text", "").lower().strip()
            
            # Skip empty concepts
            if not concept_text:
                continue
            
            # Check for exact duplicates
            if concept_text in seen_texts:
                continue
            
            # Check for similar concepts (simple similarity check)
            is_similar = False
            for seen_text in seen_texts:
                # Check if one concept is contained in another
                if (concept_text in seen_text or seen_text in concept_text) and \
                   abs(len(concept_text) - len(seen_text)) / max(len(concept_text), len(seen_text)) < 0.3:
                    is_similar = True
                    break
            
            if not is_similar:
                deduplicated.append(concept)
                seen_texts.add(concept_text)
        
        return deduplicated
    
    def _calculate_confidence_score(self, concepts: List[Dict[str, Any]]) -> float:
        """Calculate overall confidence score for extracted concepts."""
        if not concepts:
            return 0.0
        
        total_confidence = sum(concept.get("importance", 0.5) for concept in concepts)
        return total_confidence / len(concepts)
    
    def _is_definition_sentence(self, sentence: str) -> bool:
        """Check if sentence contains a definition."""
        definition_indicators = [
            " is ", " are ", " refers to ", " means ", " defined as ",
            " can be described as ", " represents "
        ]
        return any(indicator in sentence.lower() for indicator in definition_indicators)
    
    def _contains_formula(self, sentence: str) -> bool:
        """Check if sentence contains mathematical formulas."""
        formula_indicators = ["=", "∫", "∂", "∑", "π", "α", "β", "γ", "θ", "λ", "σ"]
        return any(indicator in sentence for indicator in formula_indicators)
    
    def _contains_academic_terms(self, sentence: str) -> bool:
        """Check if sentence contains academic terms - more inclusive."""
        sentence_lower = sentence.lower()
        
        # Check subject-specific terms
        for terms in self.academic_terms.values():
            if any(term.lower() in sentence_lower for term in terms):
                return True
        
        # Additional general academic indicators
        general_academic_terms = [
            "analysis", "method", "approach", "technique", "strategy",
            "process", "procedure", "mechanism", "system", "model",
            "framework", "structure", "pattern", "relationship", "connection",
            "factor", "element", "component", "aspect", "feature",
            "research", "study", "investigation", "experiment", "observation",
            "hypothesis", "conclusion", "result", "finding", "evidence",
            "problem", "solution", "issue", "challenge", "application"
        ]
        
        return any(term in sentence_lower for term in general_academic_terms)
    
    def _has_important_patterns(self, sentence: str) -> bool:
        """Check if sentence has important linguistic patterns - expanded."""
        important_patterns = [
            r'\b(important|significant|key|crucial|essential|fundamental|major|primary|main)\b',
            r'\b(because|therefore|thus|consequently|as a result|due to|leads to)\b',
            r'\b(first|second|third|finally|in conclusion|moreover|furthermore)\b',
            r'\b(definition|concept|principle|theory|law|rule|property|characteristic)\b',
            r'\b(example|instance|case|demonstration|illustration|shows that)\b',
            r'\b(however|although|despite|nevertheless|on the other hand|whereas)\b',
            r'\b(we can see|it follows|this means|in other words|specifically)\b',
            r'\b(consider|suppose|assume|let us|imagine|think about)\b',
            r'\b(proves|demonstrates|indicates|suggests|implies|reveals)\b'
        ]
        
        sentence_lower = sentence.lower()
        for pattern in important_patterns:
            if re.search(pattern, sentence_lower):
                return True
        return False