"""
Web Application for Lecture Summarizer & Note Enhancer
Simple Flask-based interface for uploading audio files and displaying generated notes.
"""

import os
import sys
import time
import json
import threading
import numpy as np
from pathlib import Path
from typing import Dict, Any
from flask_socketio import SocketIO, emit, join_room, leave_room

# Add parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
from flask_cors import CORS
from werkzeug.utils import secure_filename

from agents.orchestrator import LectureAgentOrchestrator
from evaluation.metrics import EvaluationMetrics
import config

def make_json_serializable(obj):
    """Convert any object to JSON-serializable format."""
    if isinstance(obj, dict):
        return {key: make_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [make_json_serializable(item) for item in obj]
    elif isinstance(obj, tuple):
        return [make_json_serializable(item) for item in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif hasattr(obj, '__dict__'):
        return make_json_serializable(obj.__dict__)
    else:
        return obj

def create_app():
    """Create and configure the Flask application."""
    app = Flask(__name__)
    app.secret_key = 'lecture_summarizer_secret_key_change_in_production'
    
    # Enable CORS for cross-origin requests
    CORS(app)
    
    # Initialize SocketIO
    socketio = SocketIO(app, cors_allowed_origins="*")
    
    # Configure upload settings
    app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size
    app.config['UPLOAD_FOLDER'] = os.path.join('data', 'audio')
    
    # Ensure upload directory exists
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    
    # Initialize system components
    orchestrator = LectureAgentOrchestrator()
    evaluator = EvaluationMetrics()
    
    # Allowed file extensions
    ALLOWED_EXTENSIONS = {'.wav', '.mp3', '.m4a', '.flac', '.ogg'}
    
    def allowed_file(filename):
        """Check if file extension is allowed."""
        return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS
    
    def progress_callback(session_id, progress, message):
        """Progress callback for real-time updates."""
        print(f"Sending progress update to session {session_id}: {progress}% - {message}")
        try:
            # Ensure progress and message are JSON serializable
            progress = float(progress) if progress is not None else 0
            message = str(message) if message is not None else ""
            
            socketio.emit('progress_update', {
                'progress': progress,
                'message': message,
                'timestamp': time.time()
            }, to=session_id)
        except Exception as e:
            print(f"Error sending progress update: {e}")
            # Send a simple fallback update
            socketio.emit('progress_update', {
                'progress': 0,
                'message': "Processing...",
                'timestamp': time.time()
            }, to=session_id)
    
    @app.route('/')
    def index():
        """Main page with upload form."""
        return render_template('index.html')
    
    @app.route('/upload', methods=['POST'])
    def upload_file():
        """Handle file upload and processing."""
        try:
            # Check if file was uploaded
            if 'audio_file' not in request.files:
                return jsonify({'error': 'No file selected'}), 400
            
            file = request.files['audio_file']
            session_id = request.form.get('session_id', str(int(time.time())))
            
            if file.filename == '':
                return jsonify({'error': 'No file selected'}), 400
            
            if not allowed_file(file.filename):
                return jsonify({'error': f'File type not supported. Allowed: {", ".join(ALLOWED_EXTENSIONS)}'}), 400
            
            # Save uploaded file
            filename = secure_filename(file.filename) if file.filename else 'unknown_file'
            timestamp = int(time.time())
            filename = f"{timestamp}_{filename}"
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            
            file.save(file_path)
            
            # Process the lecture with progress callback
            def process_with_progress():
                try:
                    def progress_update(progress, message):
                        progress_callback(session_id, progress, message)
                    
                    result = orchestrator.process_lecture(file_path, progress_callback=progress_update)
                    
                    if not result or not result.get('success', False):
                        error_msg = result.get('error', 'Unknown processing error') if result else 'No result returned'
                        socketio.emit('processing_error', {'error': error_msg}, to=session_id)
                        return
                    
                    # Emit final result with enhanced orchestrator information
                    result_data = {
                        'success': True,
                        'workflow_id': result.get('workflow_id'),
                        'processing_time': result.get('processing_time', 0),
                        'transcription': result.get('transcription', ''),
                        'summary': result.get('summary', ''),
                        'concepts': result.get('concepts', []),
                        'key_terms': result.get('key_terms', []),
                        'definitions': result.get('definitions', []),
                        'formulas': result.get('formulas', []),
                        'study_notes': result.get('study_notes', {}),
                        'quality_score': result.get('overall_quality', 0),
                        # Enhanced orchestrator information
                        'stage_durations': result.get('stage_durations', {}),
                        'agent_performance': result.get('agent_performance', {}),
                        'quality_checks': result.get('quality_checks', {}),
                        # Fix key mismatches - extract from study_notes structure
                        'flashcards': result.get('study_notes', {}).get('flash_cards', []),
                        'questions': result.get('study_notes', {}).get('practice_questions', [])
                    }
                    
                    # Make all data JSON serializable
                    result_data = make_json_serializable(result_data)
                    
                    # Send to specific session
                    print(f"Sending processing_complete to session {session_id}")
                    try:
                        socketio.emit('processing_complete', result_data, to=session_id)
                        print(f"Successfully sent processing_complete to session {session_id}")
                    except Exception as emit_error:
                        print(f"Error emitting processing_complete: {emit_error}")
                        # Send a simplified error message
                        socketio.emit('processing_error', {
                            'error': f'Result serialization failed: {str(emit_error)}'
                        }, to=session_id)
                    
                except Exception as e:
                    socketio.emit('processing_error', {'error': str(e)}, to=session_id)
                
                finally:
                    # Clean up uploaded file
                    try:
                        if os.path.exists(file_path):
                            os.remove(file_path)
                    except:
                        pass
            
            # Start processing in background thread
            thread = threading.Thread(target=process_with_progress)
            thread.daemon = True
            thread.start()
            
            return jsonify({
                'success': True,
                'session_id': session_id,
                'message': 'Processing started'
            })
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @app.route('/status')
    def system_status():
        """Get system status and health information."""
        try:
            status = orchestrator.get_system_status()
            return jsonify({
                'status': 'healthy',
                'system_info': status,
                'uptime': time.time(),
                'version': '1.0.0'
            })
        except Exception as e:
            return jsonify({'status': 'error', 'error': str(e)}), 500
    
    @app.route('/workflow-history')
    def workflow_history():
        """Get workflow processing history."""
        try:
            history = orchestrator.get_workflow_history()
            return jsonify({
                'history': history[-10:],  # Last 10 workflows
                'total_count': len(history)
            })
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @app.route('/evaluate', methods=['POST'])
    def run_evaluation():
        """Run system evaluation."""
        try:
            evaluation_results = evaluator.run_full_evaluation()
            return jsonify(evaluation_results)
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @app.route('/demo')
    def demo_page():
        """Demo page with sample processing."""
        return render_template('demo.html')
    
    @app.route('/demo/process', methods=['POST'])
    def process_demo():
        """Process demo text input."""
        try:
            # Handle both JSON and form data
            if request.is_json and request.json:
                demo_text = request.json.get('text', '')
            else:
                demo_text = request.form.get('text', '')
            
            if not demo_text.strip():
                return jsonify({'error': 'No text provided'}), 400
            
            # Create a mock transcription result for demo
            mock_transcription = {
                'transcription': demo_text,
                'segments': [],
                'confidence': 0.85,
                'word_count': len(demo_text.split())
            }
            
            # Process with concept extractor and summary generator
            from agents.concept_extractor import ConceptExtractorAgent
            from agents.summary_generator import SummaryGeneratorAgent
            
            concept_extractor = ConceptExtractorAgent()
            summary_generator = SummaryGeneratorAgent()
            
            # Extract concepts
            concept_result = concept_extractor.process(mock_transcription)
            
            # Generate summary
            combined_data = {**mock_transcription, **concept_result}
            summary_result = summary_generator.process(combined_data)
            
            return jsonify({
                'success': True,
                'transcription': demo_text,
                'concepts': concept_result.get('concepts', []),
                'definitions': concept_result.get('definitions', []),
                'formulas': concept_result.get('formulas', []),
                'key_terms': concept_result.get('key_terms', []),
                'summary': summary_result.get('academic_summary', ''),
                'bullet_summary': summary_result.get('bullet_summary', ''),
                'study_notes': summary_result.get('study_notes', {})
            })
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/health')
    def health_check():
        """Simple health check endpoint."""
        return jsonify({'status': 'healthy', 'timestamp': time.time()})
    
    # Error handlers
    @app.errorhandler(413)
    def too_large(e):
        return jsonify({'error': 'File too large. Maximum size is 100MB.'}), 413
    
    @app.errorhandler(404)
    def not_found(e):
        return jsonify({'error': 'Endpoint not found'}), 404
    
    @app.errorhandler(500)
    def internal_error(e):
        return jsonify({'error': 'Internal server error'}), 500
    
    # WebSocket event handlers
    @socketio.on('connect')
    def handle_connect():
        pass
        
    @socketio.on('disconnect')
    def handle_disconnect():
        pass
        
    @socketio.on('join_session')
    def handle_join_session(data):
        session_id = data.get('session_id')
        if session_id:
            join_room(session_id)
            emit('session_joined', {'session_id': session_id})
            print(f"Client joined session room: {session_id}")
    
    return app, socketio

if __name__ == '__main__':
    app, socketio = create_app()
    print("🎓 Lecture Summarizer Web Interface")
    print("=" * 50)
    print(f"✅ Server starting on http://{config.FLASK_HOST}:{config.FLASK_PORT}")
    print("✅ All models loaded and ready")
    print("✅ WebSocket support enabled for real-time progress")
    print("=" * 50)
    
    socketio.run(
        app,
        host=config.FLASK_HOST, 
        port=config.FLASK_PORT, 
        debug=config.FLASK_DEBUG,
        use_reloader=False,  # Disable auto-reloader to prevent constant restarts
        allow_unsafe_werkzeug=True  # For development
    )