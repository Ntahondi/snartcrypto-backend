from flask import Flask, request, jsonify
from snailguard.core.detector import SnailGuardDetector
from snailguard.core.license_manager import LicenseManager
import logging

app = Flask(__name__)
license_manager = LicenseManager()

@app.route('/v1/analyze', methods=['POST'])
def analyze_request():
    """Universal API endpoint for all programming languages"""
    try:
        # Get API key from header
        api_key = request.headers.get('X-API-Key')
        if not api_key:
            return jsonify({'error': 'API key required'}), 401
        
        # Validate license
        is_valid, license_info = license_manager.validate_key(api_key)
        if not is_valid:
            return jsonify({'error': 'Invalid API key'}), 403
        
        # Parse request data from any language
        request_data = request.get_json()
        
        # Analyze with SnailGuard
        detector = SnailGuardDetector(api_key=api_key)
        result = detector.analyze_request(request_data)
        
        return jsonify({
            'is_threat': result.is_threat,
            'confidence': result.confidence,
            'phase': result.phase.value,
            'actions': result.actions,
            'total_attacker_cost': sum(a['estimated_attacker_cost'] for a in result.actions)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/v1/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'service': 'SnailGuard AI'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=False)