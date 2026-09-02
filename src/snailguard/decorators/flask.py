from functools import wraps
from typing import Callable, Any
import logging
from flask import request, jsonify

from snailguard.core.detector import SnailGuardDetector, LicenseError

logger = logging.getLogger(__name__)

def protect_flask(api_key: str = None, **config):
    """
    Flask decorator with API key validation
    """
    
    def decorator(f: Callable) -> Callable:
        @wraps(f)
        def decorated_function(*args, **kwargs):
            try:
                # Initialize detector with API key
                local_detector = SnailGuardDetector(api_key=api_key, config=config)
                
                # Extract request data for analysis
                request_data = {
                    'method': request.method,
                    'path': request.path,
                    'headers': dict(request.headers),
                    'query_params': dict(request.args),
                    'body': request.get_json(silent=True) or {},
                    'client_ip': request.remote_addr,
                    'timestamp': request.environ.get('REQUEST_TIME', None)
                }
                
                # Analyze request with SnailGuard AI
                result = local_detector.analyze_request(request_data)
                
                if result.is_threat:
                    logger.warning(
                        f"Blocked threat with confidence {result.confidence:.3f} "
                        f"at phase {result.phase.value}"
                    )
                    
                    # Return economic warfare response
                    return jsonify({
                        'error': 'Request blocked by SnailGuard AI',
                        'threat_detected': True,
                        'confidence': result.confidence,
                        'phase': result.phase.value,
                        'actions_taken': result.actions
                    }), 429
                
                # Request is safe - proceed
                logger.debug(
                    f"Safe request processed with confidence {result.confidence:.3f} "
                    f"at phase {result.phase.value}"
                )
                
                return f(*args, **kwargs)
                
            except LicenseError as e:
                logger.error(f"License error: {e}")
                return jsonify({
                    'error': 'SnailGuard AI license error',
                    'message': str(e),
                    'get_api_key': 'https://snailguard.ai/api-keys'
                }), 403
            except Exception as e:
                logger.error(f"SnailGuard analysis error: {e}")
                # Fail open - allow request through on errors
                return f(*args, **kwargs)
        
        return decorated_function
    
    return decorator