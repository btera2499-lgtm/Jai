from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import requests
import json
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Get API keys from environment variables
NVIDIA_API_KEY = os.environ.get('NVIDIA_API_KEY', '')
DEEPSEEK_API_KEY = os.environ.get('DEEPSEEK_API_KEY', '')

NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"

def get_api_config(model):
    """Determine which API to use based on model name"""
    if 'deepseek' in model.lower():
        return {
            'base_url': DEEPSEEK_BASE_URL,
            'api_key': DEEPSEEK_API_KEY,
            'provider': 'deepseek'
        }
    else:
        return {
            'base_url': NVIDIA_BASE_URL,
            'api_key': NVIDIA_API_KEY,
            'provider': 'nvidia'
        }

@app.route('/v1/chat/completions', methods=['POST', 'OPTIONS'])
def chat_completions():
    # Handle preflight request
    if request.method == 'OPTIONS':
        return '', 204
    
    try:
        data = request.get_json()
        
        # Extract parameters
        messages = data.get('messages', [])
        model = data.get('model', 'deepseek-chat')
        temperature = data.get('temperature', 0.7)
        max_tokens = data.get('max_tokens', 2048)
        stream = data.get('stream', False)
        
        # Get API configuration based on model
        api_config = get_api_config(model)
        
        print(f"[INFO] Request for model: {model}, provider: {api_config['provider']}")
        
        if not api_config['api_key']:
            print(f"[ERROR] API key for {api_config['provider']} not configured")
            return jsonify({
                "error": {
                    "message": f"API key for {api_config['provider']} not configured",
                    "type": "auth_error",
                    "code": 401
                }
            }), 401
        
        # Prepare request payload
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream
        }
        
        headers = {
            "Authorization": f"Bearer {api_config['api_key']}",
            "Content-Type": "application/json"
        }
        
        # Make request to appropriate API
        if stream:
            return handle_streaming(payload, headers, api_config['base_url'], api_config['provider'])
        else:
            print(f"[INFO] Sending request to {api_config['base_url']}/chat/completions")
            response = requests.post(
                f"{api_config['base_url']}/chat/completions",
                headers=headers,
                json=payload,
                timeout=60
            )
            
            print(f"[INFO] Response status: {response.status_code}")
            
            if response.status_code == 200:
                return jsonify(response.json())
            else:
                print(f"[ERROR] API error: {response.text}")
                return jsonify({
                    "error": {
                        "message": f"{api_config['provider'].upper()} API error: {response.text}",
                        "type": "api_error",
                        "code": response.status_code
                    }
                }), response.status_code
                
    except requests.exceptions.Timeout:
        print("[ERROR] Request timeout")
        return jsonify({
            "error": {
                "message": "Request timeout",
                "type": "timeout_error"
            }
        }), 504
    except requests.exceptions.ConnectionError as e:
        print(f"[ERROR] Connection error: {str(e)}")
        return jsonify({
            "error": {
                "message": f"Connection error: {str(e)}",
                "type": "connection_error"
            }
        }), 503
    except Exception as e:
        print(f"[ERROR] Unexpected error: {str(e)}")
        return jsonify({
            "error": {
                "message": str(e),
                "type": "internal_error"
            }
        }), 500

def handle_streaming(payload, headers, base_url, provider):
    def generate():
        try:
            print(f"[INFO] Starting streaming request to {provider}")
            response = requests.post(
                f"{base_url}/chat/completions",
                headers=headers,
                json=payload,
                stream=True,
                timeout=60
            )
            
            for line in response.iter_lines():
                if line:
                    yield line.decode('utf-8') + '\n'
                    
        except Exception as e:
            print(f"[ERROR] Streaming error: {str(e)}")
            error_data = {
                "error": {
                    "message": str(e),
                    "type": "stream_error"
                }
            }
            yield f"data: {json.dumps(error_data)}\n\n"
    
    return Response(generate(), mimetype='text/event-stream')

@app.route('/v1/models', methods=['GET'])
def list_models():
    """List available models from both providers"""
    models_list = []
    
    # Add NVIDIA models if API key is configured
    if NVIDIA_API_KEY:
        nvidia_models = [
            {"id": "meta/llama-3.1-405b-instruct", "object": "model", "provider": "nvidia"},
            {"id": "meta/llama-3.1-70b-instruct", "object": "model", "provider": "nvidia"},
            {"id": "meta/llama-3.1-8b-instruct", "object": "model", "provider": "nvidia"},
            {"id": "mistralai/mixtral-8x7b-instruct-v0.1", "object": "model", "provider": "nvidia"}
        ]
        models_list.extend(nvidia_models)
    
    # Add DeepSeek models if API key is configured
    if DEEPSEEK_API_KEY:
        deepseek_models = [
            {"id": "deepseek-chat", "object": "model", "provider": "deepseek"},
            {"id": "deepseek-reasoner", "object": "model", "provider": "deepseek"}
        ]
        models_list.extend(deepseek_models)
    
    return jsonify({
        "object": "list",
        "data": models_list
    })

@app.route('/health', methods=['GET'])
def health():
    config_status = {
        "nvidia": bool(NVIDIA_API_KEY),
        "deepseek": bool(DEEPSEEK_API_KEY)
    }
    return jsonify({
        "status": "healthy",
        "configured_providers": config_status
    })

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        "message": "OpenAI-Compatible Multi-Provider Proxy",
        "supported_providers": ["NVIDIA NIM", "DeepSeek"],
        "endpoints": {
            "chat": "/v1/chat/completions",
            "models": "/v1/models",
            "health": "/health"
        },
        "usage": {
            "nvidia_models": ["meta/llama-3.1-405b-instruct", "meta/llama-3.1-70b-instruct"],
            "deepseek_models": ["deepseek-chat (V3)", "deepseek-reasoner (R1)"]
        }
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)        models_list.extend(deepseek_models)
    
    return jsonify({
        "object": "list",
        "data": models_list
    })

@app.route('/health', methods=['GET'])
def health():
    config_status = {
        "nvidia": bool(NVIDIA_API_KEY),
        "deepseek": bool(DEEPSEEK_API_KEY)
    }
    return jsonify({
        "status": "healthy",
        "configured_providers": config_status
    })

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        "message": "OpenAI-Compatible Multi-Provider Proxy",
        "supported_providers": ["NVIDIA NIM", "DeepSeek"],
        "endpoints": {
            "chat": "/v1/chat/completions",
            "models": "/v1/models",
            "health": "/health"
        },
        "usage": {
            "nvidia_models": ["meta/llama-3.1-405b-instruct", "meta/llama-3.1-70b-instruct"],
            "deepseek_models": ["deepseek-chat (V3)", "deepseek-reasoner (R1)"]
        }
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)
