"""
main.py (Updated) - Flask App with Config
Edward He - CS Student, University of Ottawa

Now loads all settings from config.yml
"""

from flask import Flask, request, jsonify
from dotenv import load_dotenv  # ✅ FIXED: was from flask.cli, should be from dotenv
from flask_cors import CORS
import logging
import os
from datetime import datetime
import uuid
import yaml

from ai import (
    load_model,
    generate_response,
    validate_message,
    clean_response,
    build_conversation_context
)

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")  # Currently unused, but kept for future use

# Load configuration
config_path = os.path.join(os.path.dirname(__file__), "config.yml")

try:
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
except FileNotFoundError:
    print(f"ERROR: config.yml not found at {config_path}")
    raise
except yaml.YAMLError as e:
    print(f"ERROR: Invalid YAML in config.yml: {e}")
    raise

model_config = config["model"]
inference_config = config["inference"]
server_config = config["server"]
conversation_config = config["conversation"]
api_config = config["api"]
logging_config = config["logging"]

# Setup logging
logging.basicConfig(
    level=logging_config.get("level", "INFO"),
    format=logging_config.get("format", "%(asctime)s - %(levelname)s - %(message)s"),
    handlers=[
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Flask App
app = Flask(__name__)

# Configure CORS
if server_config.get("cors_enabled", True):
    cors_origins = server_config.get("cors_origins", ["*"])
    CORS(app, origins=cors_origins)
    logger.info(f"CORS enabled for origins: {cors_origins}")
else:
    logger.info("CORS is disabled")

# Preload model on startup if configured (avoids long first-request latency)
if config.get("cache", {}).get("preload_model", False):
    logger.info("Preloading model on startup (cache.preload_model=true)...")
    load_model()

# Global State
conversations = {}

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_session_id():
    """Get session ID from request header or generate new one"""
    return request.headers.get("X-Session-ID", str(uuid.uuid4()))

def store_conversation(session_id: str, messages: list):  # ✅ FIXED: was sesson_id
    """Store conversation in memory"""
    conversations[session_id] = messages

def get_conversation(session_id: str):  # ✅ FIXED: was sesson_id
    """Retrieve conversation history"""
    return conversations.get(session_id, [])

def clear_conversation(session_id: str):  # ✅ FIXED: was sesson_id
    """Clear conversation history"""
    conversations.pop(session_id, None)

# ============================================================================
# ROUTES
# ============================================================================

@app.route("/", methods=["GET"])
def root():
    """API information endpoint"""
    response = {
        "name": "Open Source LLM Chatbot Server Route Handler",
        "version": "1.0",
        "model": model_config["id"],
        "endpoints": {
            "GET /": "This message",
            "GET /status": "Server status",
            "POST /chat": "Chat endpoint",
            "POST /clear": "Clear conversation",
            "GET /health": "Health check"
        }
    }

    if api_config.get("include_model_info", True):
        response["config"] = {
            "model_id": model_config["id"],
            "max_tokens": inference_config["max_tokens"],
            "temperature": inference_config["temperature"],
        }

    return jsonify(response), 200

@app.route('/status', methods=['GET'])
def status():
    """Server status endpoint"""
    data = {
        "status": "online",
        "model": model_config["id"],
        "active_sessions": len(conversations),
    }

    if api_config.get("include_timestamp", True):
        data["timestamp"] = datetime.now().isoformat()

    return jsonify(data), 200

@app.route('/health', methods=['GET'])
def health_check():
    """Health check for orchestration"""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat() if api_config.get("include_timestamp", True) else None
    }), 200

@app.route('/chat', methods=['POST'])
def chat():
    """Chat endpoint - main API endpoint"""
    try:
        data = request.get_json()

        if not data:
            return jsonify({"error": "No JSON data provided"}), 400

        # Extract parameters
        user_message = data.get('message')
        session_id = data.get('session_id', get_session_id())
        max_tokens = data.get('max_tokens')

        # Validate message
        is_valid, error_msg = validate_message(user_message)
        if not is_valid:
            return jsonify({"error": error_msg}), 400

        # Validate max_tokens
        if max_tokens is not None:
            if not isinstance(max_tokens, int) or max_tokens < 1 or max_tokens > 4096:
                return jsonify({"error": "max_tokens must be between 1 and 4096"}), 400

        logger.info(f"[{session_id}] Received message: {len(user_message)} chars")

        # Load model
        load_model()

        # Get conversation history
        messages = get_conversation(session_id)

        # Build context
        messages = build_conversation_context(user_message, messages)

        # Generate response
        logger.info(f"[{session_id}] Generating response...")
        response = generate_response(messages, max_tokens)
        response = clean_response(response)

        # Store conversation
        messages.append({"role": "assistant", "content": response})
        store_conversation(session_id, messages)

        logger.info(f"[{session_id}] Response generated: {len(response)} chars")

        # Build response
        response_data = {
            "response": response,
            "session_id": session_id,
            "message_count": len(messages),
        }

        if api_config.get("include_timestamp", True):
            response_data["timestamp"] = datetime.now().isoformat()

        return jsonify(response_data), 200

    except Exception as e:
        logger.error(f"Error in /chat: {str(e)}")
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500

@app.route('/clear', methods=['POST'])
def clear():
    """Clear conversation endpoint"""
    try:
        data = request.get_json(silent=True) or {}

        session_id = data.get('session_id', get_session_id())
        clear_conversation(session_id)
        logger.info(f"[{session_id}] Conversation cleared")

        return jsonify({"status": "cleared", "session_id": session_id}), 200

    except Exception as e:
        logger.error(f"Error in /clear: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/history/<session_id>', methods=['GET'])
def get_history(session_id: str):
    """Get conversation history endpoint"""
    try:
        messages = get_conversation(session_id)

        if not messages:
            return jsonify({"error": "Session not found"}), 404

        return jsonify({
            "session_id": session_id,
            "messages": messages,
            "message_count": len(messages)
        }), 200

    except Exception as e:
        logger.error(f"Error in /history: {str(e)}")
        return jsonify({"error": str(e)}), 500

# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(405)
def method_not_allowed(error):
    return jsonify({"error": "Method not allowed"}), 405

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500

# ============================================================================
# HOOKS
# ============================================================================

@app.before_request
def before_request():
    logger.debug(f"{request.method} {request.path}")

@app.after_request
def after_request(response):
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'DENY'
    return response

# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    port = int(os.getenv("PORT", server_config.get("port", 8080)))
    host = server_config.get("host", "0.0.0.0")
    debug = server_config.get("debug", False)

    print(f"""
    ╔═══════════════════════════════════════════════════════════╗
    │     Open Source Chatbot API - Flask with Config           │
    │     Edward He - CS Student, University of Ottawa         │
    ├───────────────────────────────────────────────────────────┤
    │  Configuration File: config.yml                           │
    │  Model: {model_config['id']:<42}│
    │  Max Tokens: {inference_config['max_tokens']:<37}│
    │  Temperature: {inference_config['temperature']:<35}│
    │                                                           │
    │  Server: http://{host}:{port:<49}│
    │  Debug Mode: {str(debug):<41}│
    │                                                           │
    │  Press Ctrl+C to stop                                    │
    ╚═══════════════════════════════════════════════════════════╝
    """)

    app.run(host=host, port=port, debug=debug)