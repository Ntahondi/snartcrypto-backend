from functools import wraps
from typing import Callable, Any
import logging
from fastapi import Request, HTTPException, status

from snailguard.core.detector import SnailGuardDetector

logger = logging.getLogger(__name__)

def protect_fastapi(detector: SnailGuardDetector = None, **config):
    """
    FastAPI decorator for protecting API endpoints with SnailGuard AI
    
    Usage:
        @app.get('/api/protected')
        @protect_fastapi()
        async def protected_endpoint():
            return {"message": "Safe access"}
    """
    
    def decorator(f: Callable) -> Callable:
        @wraps(f)
        async def decorated_function(request: Request, *args, **kwargs):
            # Initialize detector if not provided
            local_detector = detector or SnailGuardDetector(config=config)
            
            try:
                # Extract request data for analysis
                body = await request.body()
                try:
                    import json
                    body_data = json.loads(body) if body else {}
                except:
                    body_data = {}
                
                request_data = {
                    'method': request.method,
                    'path': request.url.path,
                    'headers': dict(request.headers),
                    'query_params': dict(request.query_params),
                    'body': body_data,
                    'client_ip': request.client.host if request.client else None,
                    'timestamp': None
                }
                
                # Analyze request with SnailGuard AI
                result = local_detector.analyze_request(request_data)
                
                if result.is_threat:
                    logger.warning(
                        f"Blocked threat with confidence {result.confidence:.3f} "
                        f"at phase {result.phase.value}"
                    )
                    
                    # Return economic warfare response
                    raise HTTPException(
                        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                        detail={
                            'error': 'Request blocked by SnailGuard AI',
                            'threat_detected': True,
                            'confidence': result.confidence,
                            'phase': result.phase.value,
                            'actions_taken': result.actions
                        }
                    )
                
                # Request is safe - proceed
                logger.debug(
                    f"Safe request processed with confidence {result.confidence:.3f} "
                    f"at phase {result.phase.value}"
                )
                
                return await f(request, *args, **kwargs)
                
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"SnailGuard analysis error: {e}")
                # Fail open - allow request through on errors
                return await f(request, *args, **kwargs)
        
        return decorated_function
    
    return decorator