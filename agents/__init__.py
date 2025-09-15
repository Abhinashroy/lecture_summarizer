"""
Agents package for the Lecture Summarizer & Note Enhancer.
Multi-agent system for processing lecture audio and generating structured notes.
"""

from .base_agent import BaseAgent
from .audio_transcriber import AudioTranscriberAgent
from .concept_extractor import ConceptExtractorAgent
from .summary_generator import SummaryGeneratorAgent
from .orchestrator import LectureAgentOrchestrator

__all__ = [
    "BaseAgent",
    "AudioTranscriberAgent", 
    "ConceptExtractorAgent",
    "SummaryGeneratorAgent",
    "LectureAgentOrchestrator"
]