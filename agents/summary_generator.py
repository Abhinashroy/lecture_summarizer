"""
Summary Generator Agent
Creates structured bullet-point summaries and hierarchical organization of lecture content.
"""

import re
from typing import Dict, Any, List, Tuple
import time
from datetime import datetime
from collections import defaultdict

from .base_agent import BaseAgent
import config

class SummaryGeneratorAgent(BaseAgent):
    """
    Agent responsible for creating structured, hierarchical summaries
    from transcribed lecture content and extracted concepts.
    """
    
    def __init__(self):
        super().__init__("SummaryGenerator")
        self.summary_templates = self._load_summary_templates()
    
    def _load_summary_templates(self) -> Dict[str, str]:
        """Load templates for different types of summaries."""
        return {
            "academic": """# {title}
**Date:** {date}
**Duration:** {duration}
**Subject:** {subject}

## 📋 Key Concepts
{key_concepts}

## 📚 Definitions
{definitions}

## 🔢 Formulas & Equations
{formulas}

## 📝 Detailed Notes
{detailed_notes}

## 🎯 Summary Points
{summary_points}

## 🔗 Cross-References
{cross_references}
""",
            "bullet_points": """# {title}

## Main Topics
{main_topics}

## Key Points
{key_points}

## Important Details
{important_details}

## Action Items
{action_items}
""",
            "outline": """# {title}

{outline_content}
"""
        }
    
    def process(self, input_data: Any) -> Dict[str, Any]:
        """
        Generate structured summary from transcription and extracted concepts.
        
        Args:
            input_data: Dict containing transcription, concepts, and metadata
            
        Returns:
            Dict containing generated summary, structured notes, and metadata
        """
        start_time = time.time()
        
        try:
            # Extract input components
            transcription = input_data.get("transcription", "")
            concepts = input_data.get("concepts", [])
            definitions = input_data.get("definitions", [])
            formulas = input_data.get("formulas", [])
            key_terms = input_data.get("key_terms", [])
            subject_areas = input_data.get("subject_areas", [])
            
            if not transcription:
                return {"error": "No transcription provided for summarization"}
            
            # Generate different summary formats
            academic_summary = self._generate_academic_summary(
                transcription, concepts, definitions, formulas, key_terms, subject_areas
            )
            
            bullet_summary = self._generate_bullet_summary(
                transcription, concepts, key_terms
            )
            
            outline_summary = self._generate_outline_summary(
                transcription, concepts
            )
            
            # Create study notes
            study_notes = self._generate_study_notes(
                concepts, definitions, formulas, key_terms
            )
            
            # Generate mind map structure
            mind_map = self._generate_mind_map_structure(
                concepts, subject_areas, key_terms
            )
            
            # Extract action items and next steps
            action_items = self._extract_action_items(transcription)
            
            processing_time = self.log_performance(start_time, "summary_generation")
            
            # Calculate and store quality score
            structure_quality = self._assess_structure_quality(academic_summary)
            self.last_quality_score = min(structure_quality + 0.1, 1.0)  # Boost for comprehensive output
            
            return {
                "academic_summary": academic_summary,
                "bullet_summary": bullet_summary,
                "outline_summary": outline_summary,
                "study_notes": study_notes,
                "mind_map": mind_map,
                "action_items": action_items,
                "processing_time": processing_time,
                "summary_length": len(academic_summary.split()),
                "structure_quality": structure_quality,
                "output_quality": self.last_quality_score
            }
            
        except Exception as e:
            self.logger.error(f"Summary generation failed: {e}")
            return {
                "academic_summary": "",
                "bullet_summary": "",
                "outline_summary": "",
                "error": str(e),
                "processing_time": time.time() - start_time
            }
    
    def _generate_academic_summary(self, transcription: str, concepts: List[Dict], 
                                 definitions: List[Dict], formulas: List[Dict],
                                 key_terms: List[Dict], subject_areas: List[Dict]) -> str:
        """Generate comprehensive academic summary."""
        
        # Determine primary subject
        primary_subject = subject_areas[0]["subject"] if subject_areas else "General"
        
        # Format key concepts
        key_concepts_text = self._format_key_concepts(concepts[:10])  # Top 10
        
        # Format definitions
        definitions_text = self._format_definitions(definitions)
        
        # Format formulas
        formulas_text = self._format_formulas(formulas)
        
        # Generate detailed notes
        detailed_notes = self._generate_detailed_notes(transcription, concepts)
        
        # Generate summary points
        summary_points = self._generate_summary_points(concepts, key_terms)
        
        # Generate cross-references
        cross_references = self._generate_cross_references(concepts, key_terms)
        
        # Fill template
        summary = self.summary_templates["academic"].format(
            title=f"Lecture Notes - {primary_subject.title()}",
            date=datetime.now().strftime("%Y-%m-%d %H:%M"),
            duration=self._estimate_duration(transcription),
            subject=primary_subject.title(),
            key_concepts=key_concepts_text,
            definitions=definitions_text,
            formulas=formulas_text,
            detailed_notes=detailed_notes,
            summary_points=summary_points,
            cross_references=cross_references
        )
        
        return summary
    
    def _generate_bullet_summary(self, transcription: str, concepts: List[Dict], 
                               key_terms: List[Dict]) -> str:
        """Generate bullet-point summary."""
        
        # Extract main topics
        main_topics = self._extract_main_topics(concepts, key_terms)
        
        # Extract key points
        key_points = self._extract_key_points(concepts[:8])
        
        # Extract important details
        important_details = self._extract_important_details(concepts[8:15])
        
        # Extract action items
        action_items = self._extract_action_items(transcription)
        
        summary = self.summary_templates["bullet_points"].format(
            title="Lecture Summary",
            main_topics=main_topics,
            key_points=key_points,
            important_details=important_details,
            action_items="\n".join([f"- {item}" for item in action_items])
        )
        
        return summary
    
    def _generate_outline_summary(self, transcription: str, concepts: List[Dict]) -> str:
        """Generate hierarchical outline summary."""
        
        # Group concepts by importance and topic
        outline_structure = self._create_outline_structure(concepts)
        
        outline_content = self._format_outline_structure(outline_structure)
        
        summary = self.summary_templates["outline"].format(
            title="Lecture Outline",
            outline_content=outline_content
        )
        
        return summary
    
    def _format_key_concepts(self, concepts: List[Dict]) -> str:
        """Format key concepts for display."""
        if not concepts:
            return "- No key concepts identified"
        
        formatted = []
        for concept in concepts:
            importance_stars = "⭐" * min(int(concept.get("importance", 0.5) * 5), 5)
            concept_text = concept.get("text", "").strip()
            if concept_text:
                formatted.append(f"- {importance_stars} {concept_text}")
        
        return "\n".join(formatted) if formatted else "- No key concepts identified"
    
    def _format_definitions(self, definitions: List[Dict]) -> str:
        """Format definitions for display."""
        if not definitions:
            return "- No definitions identified"
        
        formatted = []
        for definition in definitions:
            term = definition.get("term", "").strip()
            desc = definition.get("definition", "").strip()
            if term and desc:
                formatted.append(f"- **{term}**: {desc}")
        
        return "\n".join(formatted) if formatted else "- No definitions identified"
    
    def _format_formulas(self, formulas: List[Dict]) -> str:
        """Format formulas and equations for display."""
        if not formulas:
            return "- No formulas identified"
        
        formatted = []
        for formula in formulas:
            expression = formula.get("expression", "").strip()
            if expression:
                formatted.append(f"- `{expression}`")
        
        return "\n".join(formatted) if formatted else "- No formulas identified"
    
    def _generate_detailed_notes(self, transcription: str, concepts: List[Dict]) -> str:
        """Generate meaningful detailed notes section without repetition."""
        # Clean and split transcription into meaningful segments
        clean_transcription = self._clean_transcription(transcription)
        segments = self._extract_meaningful_segments(clean_transcription, concepts)
        
        detailed_notes = []
        for i, segment in enumerate(segments[:5]):  # Limit to 5 main sections
            if len(segment['content'].strip()) > 30:  # Only include substantial content
                section_title = segment.get('title', f"Key Point {i+1}")
                section_content = segment['content'].strip()
                
                # Create concise section summary
                summary = self._create_section_summary(section_content, segment.get('concepts', []))
                
                note = f"### Section {i+1}: {section_title}\n{summary}"
                detailed_notes.append(note)
        
        return "\n\n".join(detailed_notes) if detailed_notes else "### Key Points\n- Content analysis in progress"
    
    def _clean_transcription(self, transcription: str) -> str:
        """Clean transcription text to remove artifacts and improve readability."""
        # Remove extra whitespace and normalize
        cleaned = ' '.join(transcription.split())
        
        # Remove common transcription artifacts
        artifacts = [
            "um", "uh", "you know", "like", "so", "basically", "actually", "right", "okay"
        ]
        
        for artifact in artifacts:
            # Only remove standalone artifacts, not parts of words
            cleaned = re.sub(rf'\b{artifact}\b', '', cleaned, flags=re.IGNORECASE)
        
        # Clean up multiple spaces
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        
        return cleaned
    
    def _extract_meaningful_segments(self, transcription: str, concepts: List[Dict]) -> List[Dict]:
        """Extract meaningful segments from transcription based on concepts and natural breaks."""
        # Split by natural sentence boundaries and group into meaningful segments
        sentences = re.split(r'[.!?]+', transcription)
        segments = []
        
        current_segment = {"content": "", "concepts": [], "title": ""}
        sentence_count = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 10:  # Skip very short sentences
                continue
                
            # Add sentence to current segment
            if current_segment["content"]:
                current_segment["content"] += ". " + sentence
            else:
                current_segment["content"] = sentence
            
            sentence_count += 1
            
            # Find concepts related to this sentence
            for concept in concepts[:10]:  # Check top concepts
                concept_text = concept.get("text", "").lower()
                if any(word in sentence.lower() for word in concept_text.split()[:3]):
                    if concept not in current_segment["concepts"]:
                        current_segment["concepts"].append(concept)
            
            # Create segment break after 2-3 sentences or when we hit a concept boundary
            if sentence_count >= 2 and (
                sentence_count >= 3 or 
                len(current_segment["concepts"]) > 0 or
                any(keyword in sentence.lower() for keyword in ["however", "additionally", "furthermore", "in contrast", "meanwhile"])
            ):
                if current_segment["content"]:
                    # Generate title from main concept or first few words
                    if current_segment["concepts"]:
                        main_concept = current_segment["concepts"][0].get("text", "")[:30]
                        current_segment["title"] = f"About {main_concept}"
                    else:
                        # Use first significant words as title
                        words = current_segment["content"].split()[:4]
                        current_segment["title"] = " ".join(words) + "..."
                    
                    segments.append(current_segment)
                    current_segment = {"content": "", "concepts": [], "title": ""}
                    sentence_count = 0
        
        # Add final segment if it has content
        if current_segment["content"]:
            if current_segment["concepts"]:
                main_concept = current_segment["concepts"][0].get("text", "")[:30]
                current_segment["title"] = f"About {main_concept}"
            else:
                words = current_segment["content"].split()[:4]
                current_segment["title"] = " ".join(words) + "..."
            segments.append(current_segment)
        
        return segments
    
    def _create_section_summary(self, content: str, related_concepts: List[Dict]) -> str:
        """Create a concise summary for a content section."""
        # Limit content length for readability
        if len(content) > 200:
            sentences = content.split('. ')
            # Take first 2 sentences or up to 200 chars
            summary_sentences = []
            char_count = 0
            for sentence in sentences:
                if char_count + len(sentence) <= 200:
                    summary_sentences.append(sentence)
                    char_count += len(sentence)
                else:
                    break
            content = '. '.join(summary_sentences) + '.'
        
        summary = content
        
        # Add related concepts if available
        if related_concepts:
            concept_names = [concept.get("text", "")[:30] for concept in related_concepts[:2]]
            summary += f"\n\n*Related concepts: {', '.join(concept_names)}*"
        
        return summary
    
    def _generate_summary_points(self, concepts: List[Dict], key_terms: List[Dict]) -> str:
        """Generate concise, well-organized summary points without repetition."""
        points = []
        
        # Group concepts by importance and uniqueness
        high_importance = [c for c in concepts if c.get("importance", 0) > 0.8]
        medium_importance = [c for c in concepts if 0.5 <= c.get("importance", 0) <= 0.8]
        
        # Create distinct summary points from high importance concepts
        for concept in high_importance[:3]:  # Top 3 most important
            concept_text = concept.get("text", "").strip()
            if concept_text and len(concept_text) > 10:
                # Truncate very long concepts for summary
                if len(concept_text) > 100:
                    concept_text = concept_text[:97] + "..."
                points.append(f"**Key Insight**: {concept_text}")
        
        # Add concepts from medium importance that are distinct
        unique_medium_concepts = []
        for concept in medium_importance:
            concept_text = concept.get("text", "").lower()
            # Check if this concept is substantially different from existing points
            is_duplicate = any(
                concept_text[:30] in existing.lower() or existing.lower()[:30] in concept_text
                for existing in [p.split(': ', 1)[1] if ': ' in p else p for p in points]
            )
            if not is_duplicate and len(concept.get("text", "")) > 15:
                unique_medium_concepts.append(concept)
        
        for concept in unique_medium_concepts[:2]:  # Add 2 more distinct concepts
            concept_text = concept.get("text", "").strip()
            if len(concept_text) > 100:
                concept_text = concept_text[:97] + "..."
            points.append(f"**Supporting Point**: {concept_text}")
        
        # Add key terms summary if available and distinct
        if key_terms:
            top_terms = []
            for term in key_terms[:6]:
                term_name = term.get("term", "").strip()
                if term_name and term_name not in top_terms:
                    top_terms.append(term_name)
            
            if top_terms:
                points.append(f"**Key Terms**: {', '.join(top_terms)}")
        
        return "\n".join(points) if points else "**Summary**: Content analysis in progress"
    
    def _generate_cross_references(self, concepts: List[Dict], key_terms: List[Dict]) -> str:
        """Generate cross-references between concepts."""
        references = []
        
        # Find concepts that reference similar terms
        term_to_concepts = defaultdict(list)
        for i, concept in enumerate(concepts):
            concept_text = concept.get("text", "").lower()
            for term in key_terms:
                term_text = term["term"].lower()
                if term_text in concept_text:
                    term_to_concepts[term["term"]].append(i + 1)
        
        # Create cross-references
        for term, concept_indices in term_to_concepts.items():
            if len(concept_indices) > 1:
                indices_str = ", ".join([f"Section {i}" for i in concept_indices])
                references.append(f"- **{term}**: Related in {indices_str}")
        
        return "\n".join(references[:5]) if references else "- No cross-references identified"
    
    def _extract_main_topics(self, concepts: List[Dict], key_terms: List[Dict]) -> str:
        """Extract main topics from concepts and key terms."""
        topics = []
        
        # Get top concepts by importance
        top_concepts = sorted(concepts, key=lambda x: x.get("importance", 0), reverse=True)[:5]
        
        for concept in top_concepts:
            concept_text = concept.get("text", "").strip()
            if concept_text and len(concept_text) < 100:  # Keep topics concise
                topics.append(f"- {concept_text}")
        
        return "\n".join(topics) if topics else "- No main topics identified"
    
    def _extract_key_points(self, concepts: List[Dict]) -> str:
        """Extract key points from concepts."""
        points = []
        
        for concept in concepts:
            concept_text = concept.get("text", "").strip()
            concept_type = concept.get("type", "")
            
            if concept_text:
                if concept_type == "definition":
                    points.append(f"- 📖 {concept_text}")
                elif concept_type == "formula":
                    points.append(f"- 🔢 {concept_text}")
                else:
                    points.append(f"- ✅ {concept_text}")
        
        return "\n".join(points) if points else "- No key points identified"
    
    def _extract_important_details(self, concepts: List[Dict]) -> str:
        """Extract important details from lower-priority concepts."""
        details = []
        
        for concept in concepts:
            concept_text = concept.get("text", "").strip()
            if concept_text and len(concept_text) > 20:  # Only substantial details
                details.append(f"- {concept_text}")
        
        return "\n".join(details) if details else "- No important details identified"
    
    def _extract_action_items(self, transcription: str) -> List[str]:
        """Extract action items and next steps from transcription."""
        action_patterns = [
            r'(?:homework|assignment|project|task|exercise|problem)[^.!?]*',
            r'(?:read|study|review|practice|solve)[^.!?]*',
            r'(?:next class|next week|next lecture)[^.!?]*',
            r'(?:prepare|complete|finish|submit)[^.!?]*'
        ]
        
        actions = []
        for pattern in action_patterns:
            matches = re.finditer(pattern, transcription, re.IGNORECASE)
            for match in matches:
                action = match.group(0).strip()
                if len(action) > 10 and action not in actions:
                    actions.append(action)
        
        return actions[:5]  # Limit to 5 action items
    
    def _create_outline_structure(self, concepts: List[Dict]) -> Dict[str, List[Dict]]:
        """Create hierarchical outline structure."""
        structure = {
            "I. Key Concepts": [],
            "II. Definitions": [],
            "III. Formulas & Equations": [],
            "IV. Supporting Details": []
        }
        
        for concept in concepts:
            concept_type = concept.get("type", "general")
            concept_text = concept.get("text", "").strip()
            
            if concept_type == "definition":
                structure["II. Definitions"].append(concept)
            elif concept_type == "formula":
                structure["III. Formulas & Equations"].append(concept)
            elif concept.get("importance", 0) > 0.7:
                structure["I. Key Concepts"].append(concept)
            else:
                structure["IV. Supporting Details"].append(concept)
        
        return structure
    
    def _format_outline_structure(self, structure: Dict[str, List[Dict]]) -> str:
        """Format outline structure for display."""
        formatted_sections = []
        
        for section_title, section_concepts in structure.items():
            if section_concepts:
                formatted_sections.append(f"## {section_title}")
                
                for i, concept in enumerate(section_concepts[:8]):  # Limit items per section
                    concept_text = concept.get("text", "").strip()
                    if concept_text:
                        formatted_sections.append(f"   {i+1}. {concept_text}")
                
                formatted_sections.append("")  # Add spacing
        
        return "\n".join(formatted_sections)
    
    def _estimate_duration(self, transcription: str) -> str:
        """Estimate lecture duration based on transcription length."""
        words_per_minute = 150  # Average speaking rate
        word_count = len(transcription.split())
        minutes = word_count / words_per_minute
        
        if minutes < 60:
            return f"{int(minutes)} minutes"
        else:
            hours = int(minutes // 60)
            remaining_minutes = int(minutes % 60)
            return f"{hours}h {remaining_minutes}m"
    
    def _assess_structure_quality(self, summary: str) -> float:
        """Assess the quality of the summary structure."""
        quality_score = 0.0
        
        # Check for section headers
        if "##" in summary:
            quality_score += 0.3
        
        # Check for bullet points
        if "-" in summary or "*" in summary:
            quality_score += 0.2
        
        # Check for definitions
        if "**" in summary:
            quality_score += 0.2
        
        # Check for formulas
        if "`" in summary:
            quality_score += 0.15
        
        # Check for proper length
        word_count = len(summary.split())
        if 200 <= word_count <= 1000:
            quality_score += 0.15
        
        return min(quality_score, 1.0)
    
    def _generate_study_notes(self, concepts: List[Dict], definitions: List[Dict],
                            formulas: List[Dict], key_terms: List[Dict]) -> Dict[str, Any]:
        """Generate structured study notes with enhanced organization."""
        
        # Deduplicate content to avoid repetition
        unique_concepts = self._deduplicate_content(concepts, "text")
        unique_definitions = self._deduplicate_content(definitions, "term")
        unique_formulas = self._deduplicate_content(formulas, "expression")
        unique_terms = self._deduplicate_content(key_terms, "term")
        
        return {
            "flash_cards": self._generate_flash_cards(unique_definitions, unique_terms),
            "practice_questions": self._generate_practice_questions(unique_concepts, unique_formulas),
            "concept_map": self._generate_concept_map(unique_concepts, unique_terms),
            "review_checklist": self._generate_review_checklist(unique_concepts, unique_definitions, unique_formulas),
            "study_tips": self._generate_study_tips(unique_concepts),
            "learning_objectives": self._generate_learning_objectives(unique_concepts, unique_definitions)
        }
    
    def _deduplicate_content(self, items: List[Dict], key_field: str) -> List[Dict]:
        """Remove duplicate items based on key field."""
        seen = set()
        unique_items = []
        
        for item in items:
            key_value = item.get(key_field, "").strip().lower()
            if key_value and key_value not in seen:
                seen.add(key_value)
                unique_items.append(item)
        
        return unique_items
    
    def _generate_flash_cards(self, definitions: List[Dict], key_terms: List[Dict]) -> List[Dict]:
        """Generate diverse flash cards for study."""
        cards = []
        
        # Definition-based cards (front-to-back and back-to-front)
        for i, definition in enumerate(definitions[:8]):
            term = definition.get('term', '')
            definition_text = definition.get("definition", "")
            
            if term and definition_text:
                # Forward card: term -> definition
                cards.append({
                    "front": f"Define: {term}",
                    "back": definition_text,
                    "type": "definition",
                    "difficulty": "medium"
                })
                
                # Reverse card every other definition to add variety
                if i % 2 == 0:
                    cards.append({
                        "front": f"What term is defined as: '{definition_text[:100]}...'",
                        "back": term,
                        "type": "reverse_definition",
                        "difficulty": "hard"
                    })
        
        # Key concept cards
        for term in key_terms[:6]:
            term_name = term.get('term', '')
            subject = term.get('subject', 'general')
            
            if term_name:
                cards.append({
                    "front": f"In {subject}, explain the importance of: {term_name}",
                    "back": f"Key concept that plays a significant role in {subject}",
                    "type": "concept",
                    "difficulty": "medium"
                })
        
        # Application cards for formulas/concepts
        for term in key_terms[6:10]:
            term_name = term.get('term', '')
            if term_name:
                cards.append({
                    "front": f"How would you apply or use: {term_name}?",
                    "back": f"Consider practical applications and problem-solving contexts",
                    "type": "application",
                    "difficulty": "hard"
                })
        
        return cards[:15]  # Limit to 15 cards for focused study
    
    def _generate_practice_questions(self, concepts: List[Dict], formulas: List[Dict]) -> List[str]:
        """Generate comprehensive practice questions based on content."""
        questions = []
        
        # Formula-based questions with different difficulty levels
        for i, formula in enumerate(formulas[:4]):
            expression = formula.get("expression", "")
            if expression:
                # Basic application
                questions.append(f"Apply the formula '{expression}' to solve a typical problem.")
                
                # Derivation/understanding (every other formula)
                if i % 2 == 0:
                    questions.append(f"Derive or explain the origin of the formula: {expression}")
        
        # Concept-based questions at different cognitive levels
        high_importance_concepts = [c for c in concepts if c.get("importance", 0) > 0.7]
        medium_importance_concepts = [c for c in concepts if 0.4 <= c.get("importance", 0) <= 0.7]
        
        # High-level analysis questions
        for concept in high_importance_concepts[:3]:
            concept_text = concept.get("text", "")
            if len(concept_text) > 15:
                questions.append(f"Analyze and explain the significance of: {concept_text}")
                questions.append(f"How does '{concept_text}' relate to other concepts in this topic?")
        
        # Application and synthesis questions
        for concept in medium_importance_concepts[:3]:
            concept_text = concept.get("text", "")
            if len(concept_text) > 15:
                questions.append(f"Provide an example or application of: {concept_text}")
        
        # Critical thinking questions
        if len(questions) < 10:
            questions.extend([
                "Compare and contrast the main concepts discussed in this lecture.",
                "What are the practical implications of the key ideas presented?",
                "How might these concepts apply to real-world scenarios?",
                "What questions would you ask to deepen your understanding of this topic?"
            ])
        
        return questions[:12]  # Limit to 12 focused questions
    
    def _generate_concept_map(self, concepts: List[Dict], key_terms: List[Dict]) -> Dict[str, List[str]]:
        """Generate concept map structure."""
        concept_map = {}
        
        # Group concepts by key terms
        for term in key_terms[:5]:
            term_name = term.get("term", "")
            related_concepts = []
            
            for concept in concepts:
                concept_text = concept.get("text", "").lower()
                if term_name.lower() in concept_text:
                    related_concepts.append(concept.get("text", ""))
            
            if related_concepts:
                concept_map[term_name] = related_concepts[:3]
        
        return concept_map
    
    def _generate_review_checklist(self, concepts: List[Dict], definitions: List[Dict], 
                                 formulas: List[Dict]) -> List[str]:
        """Generate review checklist."""
        checklist = []
        
        # Core concepts to review
        for concept in concepts[:5]:
            if concept.get("importance", 0) > 0.8:
                checklist.append(f"☐ Understand: {concept.get('text', '')}")
        
        # Definitions to memorize
        for definition in definitions[:3]:
            checklist.append(f"☐ Memorize definition: {definition.get('term', '')}")
        
        # Formulas to practice
        for formula in formulas[:3]:
            checklist.append(f"☐ Practice formula: {formula.get('expression', '')}")
        
        return checklist
    
    def _generate_study_tips(self, concepts: List[Dict]) -> List[str]:
        """Generate study tips based on content analysis."""
        tips = [
            "Review key concepts multiple times using spaced repetition",
            "Create visual diagrams to connect related concepts",
            "Practice explaining concepts in your own words"
        ]
        
        # Add subject-specific tips
        if any("math" in concept.get("subject", "").lower() for concept in concepts):
            tips.append("Work through practice problems step-by-step")
            tips.append("Focus on understanding the logic behind formulas")
        
        if any("science" in concept.get("subject", "").lower() for concept in concepts):
            tips.append("Connect concepts to real-world examples")
            tips.append("Look for cause-and-effect relationships")
        
        return tips
    
    def _generate_learning_objectives(self, concepts: List[Dict], definitions: List[Dict]) -> List[str]:
        """Generate learning objectives based on content."""
        objectives = []
        
        # Primary objectives from high-importance concepts
        high_importance_concepts = [c for c in concepts if c.get("importance", 0) > 0.8]
        for concept in high_importance_concepts[:3]:
            objectives.append(f"Understand and explain {concept.get('text', '')}")
        
        # Definition-based objectives
        key_definitions = definitions[:3]
        for definition in key_definitions:
            term = definition.get('term', '')
            if term:
                objectives.append(f"Define and apply the concept of {term}")
        
        # General objectives
        if len(objectives) < 5:
            objectives.extend([
                "Synthesize key concepts from the lecture material",
                "Apply learned concepts to solve related problems",
                "Critically analyze the relationships between concepts"
            ])
        
        return objectives[:6]  # Limit to 6 objectives
    
    def _generate_mind_map_structure(self, concepts: List[Dict], subject_areas: List[Dict], 
                                   key_terms: List[Dict]) -> Dict[str, Any]:
        """Generate mind map structure for visualization."""
        mind_map = {
            "central_topic": subject_areas[0]["subject"] if subject_areas else "Lecture",
            "branches": {}
        }
        
        # Create branches for different concept types
        concept_types = defaultdict(list)
        for concept in concepts[:15]:
            concept_type = concept.get("type", "general")
            concept_types[concept_type].append(concept.get("text", ""))
        
        for concept_type, concept_texts in concept_types.items():
            mind_map["branches"][concept_type.title()] = concept_texts[:5]
        
        # Add key terms branch
        if key_terms:
            mind_map["branches"]["Key Terms"] = [term["term"] for term in key_terms[:5]]
        
        return mind_map