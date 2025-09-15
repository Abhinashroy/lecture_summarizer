#!/usr/bin/env python3
"""
Lecture Summarizer & Note Enhancer
Main application entry point for the AI agent system.
"""

import argparse
import os
import sys
import time
from pathlib import Path

# Add current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agents.orchestrator import LectureAgentOrchestrator
from evaluation.metrics import EvaluationMetrics
from ui.web_app import create_app
import config

def main():
    """Main application entry point."""
    parser = argparse.ArgumentParser(description="Lecture Summarizer & Note Enhancer")
    parser.add_argument("--audio_file", type=str, help="Path to audio file to process")
    parser.add_argument("--web", action="store_true", help="Launch web interface")
    parser.add_argument("--train", action="store_true", help="Train the fine-tuned model")
    parser.add_argument("--evaluate", action="store_true", help="Run evaluation metrics")
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    
    if args.web:
        # Launch web interface
        print("🚀 Starting Lecture Summarizer Web Interface...")
        app, socketio = create_app()
        socketio.run(app, host=config.FLASK_HOST, port=config.FLASK_PORT, debug=config.FLASK_DEBUG)
        
    elif args.train:
        # Train the fine-tuned model
        print("🔧 Training fine-tuned model...")
        from models.fine_tuning import FineTuningPipeline
        
        trainer = FineTuningPipeline()
        trainer.prepare_training_data()
        trainer.fine_tune_model()
        print("✅ Model training completed!")
        
    elif args.evaluate:
        # Run evaluation metrics
        print("📊 Running evaluation metrics...")
        evaluator = EvaluationMetrics()
        results = evaluator.run_full_evaluation()
        print("✅ Evaluation completed!")
        print(f"Results: {results}")
        
    elif args.audio_file:
        # Process single audio file
        if not os.path.exists(args.audio_file):
            print(f"❌ Error: Audio file '{args.audio_file}' not found!")
            return
            
        print(f"🎵 Processing audio file: {args.audio_file}")
        
        # Initialize orchestrator
        orchestrator = LectureAgentOrchestrator()
        
        # Process the lecture
        start_time = time.time()
        result = orchestrator.process_lecture(args.audio_file)
        processing_time = time.time() - start_time
        
        # Save results
        output_file = os.path.join(config.OUTPUT_DIR, 
                                  f"lecture_notes_{int(time.time())}.md")
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(result['summary'])
        
        print(f"✅ Processing completed in {processing_time:.2f} seconds")
        print(f"📝 Notes saved to: {output_file}")
        print(f"🎯 Key concepts identified: {len(result['concepts'])}")
        
    else:
        # Show help and demo options
        print("🎓 Lecture Summarizer & Note Enhancer")
        print("=====================================")
        print()
        print("Usage options:")
        print("  --audio_file FILE    Process a specific audio file")
        print("  --web               Launch web interface")
        print("  --train             Train the fine-tuned model")
        print("  --evaluate          Run evaluation metrics")
        print()
        print("Example:")
        print("  python main.py --audio_file data/audio/sample_lecture.wav")
        print("  python main.py --web")

if __name__ == "__main__":
    main()