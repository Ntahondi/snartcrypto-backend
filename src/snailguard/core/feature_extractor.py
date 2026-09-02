import numpy as np
from typing import Dict, Any
import hashlib
import time
import re

class FeatureExtractor:
    """Extracts 6 real-time features from HTTP requests for AI analysis"""
    
    def extract(self, request_data: Dict[str, Any]) -> np.ndarray:
        """
        Extract the 6 core real-time features used by our trained models
        """
        features = np.zeros(6)
        
        # Feature 1: sophistication_level
        features[0] = float(self._calculate_sophistication_level(request_data) or 0.0)
        
        # Feature 2: intelligence_value  
        features[1] = float(self._calculate_intelligence_value(request_data) or 0.0)
        
        # Feature 3: is_edge_case
        features[2] = float(self._detect_edge_case(request_data) or 0.0)
        
        # Feature 4: request_estimated_cost
        features[3] = float(self._estimate_request_cost(request_data) or 0.0)
        
        # Feature 5: request_sophistication_level
        features[4] = float(self._calculate_request_sophistication(request_data) or 0.0)
        
        # Feature 6: request_is_edge_case
        features[5] = float(self._detect_request_edge_case(request_data) or 0.0)
        
        return features
    
    def _calculate_sophistication_level(self, request_data: Dict[str, Any]) -> float:
        """Calculate overall request sophistication with attack pattern detection"""
        sophistication = 0.0
        
        # Analyze headers complexity
        headers = request_data.get('headers', {})
        sophistication += min(len(headers) * 0.05, 0.2)
        
        # Analyze payload complexity and attack patterns
        body = request_data.get('body', {})
        path = request_data.get('path', '')
        method = request_data.get('method', '')
        
        # Detect SQL injection patterns
        if self._detect_sql_injection(body) or self._detect_sql_injection(path):
            sophistication += 0.4
        
        # Detect XSS patterns
        if self._detect_xss(body) or self._detect_xss(headers):
            sophistication += 0.4
            
        # Detect path traversal
        if self._detect_path_traversal(path):
            sophistication += 0.4
            
        # Detect command injection
        if self._detect_command_injection(body):
            sophistication += 0.4
        
        # Analyze authentication complexity
        if headers.get('authorization') or headers.get('x-api-key'):
            sophistication += 0.1
        
        return min(sophistication, 1.0)
    
    def _calculate_intelligence_value(self, request_data: Dict[str, Any]) -> float:
        """Calculate intelligence value based on request patterns - IMPROVED"""
        intelligence = 0.0
        
        # User agent analysis - MORE COMPREHENSIVE
        user_agent = request_data.get('headers', {}).get('user-agent', '')
        user_agent_lower = user_agent.lower()
        
        # Expanded list of suspicious tools
        suspicious_tools = [
            'sqlmap', 'nmap', 'burp', 'metasploit', 'hydra', 
            'nikto', 'wpscan', 'gobuster', 'dirb', 'dirbuster',
            'wapiti', 'arachni', 'skipfish', 'nessus', 'openvas',
            'acunetix', 'appscan', 'webinspect', 'netsparker',
            'scanner', 'crawler', 'bot'
        ]
        
        if any(tool in user_agent_lower for tool in suspicious_tools):
            intelligence += 0.9  # MAJOR boost for known attack tools
        
        # Suspicious header patterns
        headers = request_data.get('headers', {})
        suspicious_headers = [
            'x-sql-injection', 'x-xss-payload', 'x-path-traversal', 
            'x-attack-signature', 'x-scanner', 'x-proxy'
        ]
        
        # Check both header names and values
        for header_name, header_value in headers.items():
            header_name_lower = header_name.lower()
            header_value_lower = str(header_value).lower()
            
            # Check if header name is suspicious
            if any(suspicious in header_name_lower for suspicious in suspicious_headers):
                intelligence += 0.8
                
            # Check if header value contains attack patterns
            if (self._detect_sql_injection(header_value_lower) or 
                self._detect_xss(header_value_lower)):
                intelligence += 0.7
        
        # Request frequency analysis
        client_ip = request_data.get('client_ip', '')
        intelligence += self._analyze_request_frequency(client_ip) * 0.2
        
        return min(intelligence, 1.0)
    
    def _detect_edge_case(self, request_data: Dict[str, Any]) -> float:
        """Detect if request represents an edge case scenario"""
        method = request_data.get('method', 'GET')
        path = request_data.get('path', '')
        
        # Unusual method-path combinations
        unusual_combinations = [
            ('POST', '/api/query'),
            ('GET', '/api/mutation'), 
            ('DELETE', '/api/login'),
            ('PUT', '/api/config/system')
        ]
        
        if (method, path) in unusual_combinations:
            return 1.0
        
        # Attack patterns are edge cases
        body = request_data.get('body', {})
        headers = request_data.get('headers', {})
        
        if (self._detect_sql_injection(body) or self._detect_sql_injection(path) or
            self._detect_xss(body) or self._detect_xss(headers) or
            self._detect_path_traversal(path)):
            return 1.0
            
        return 0.0
    
    def _estimate_request_cost(self, request_data: Dict[str, Any]) -> float:
        """Estimate computational cost of processing this request"""
        cost = 0.0
        
        # Body size cost
        body_size = len(str(request_data.get('body', '')))
        cost += min(body_size / 1000000, 1.0) * 0.3
        
        # Attack patterns increase cost
        body = request_data.get('body', {})
        path = request_data.get('path', '')
        if (self._detect_sql_injection(body) or self._detect_sql_injection(path) or
            self._detect_xss(body) or self._detect_path_traversal(path)):
            cost += 0.5
        
        # Query complexity cost
        query_params = request_data.get('query_params', {})
        cost += min(len(query_params) * 0.1, 0.2)
        
        # Header complexity cost
        headers = request_data.get('headers', {})
        cost += min(len(headers) * 0.05, 0.1)
        
        return min(cost, 1.0)
    
    def _calculate_request_sophistication(self, request_data: Dict[str, Any]) -> float:
        """Request-specific sophistication calculation"""
        return self._calculate_sophistication_level(request_data) * 0.9
    
    def _detect_request_edge_case(self, request_data: Dict[str, Any]) -> float:
        """Request-specific edge case detection"""
        return self._detect_edge_case(request_data) * 0.9
    
    # ===== ATTACK PATTERN DETECTION METHODS =====
    
    def _detect_sql_injection(self, data: Any) -> bool:
        """Detect SQL injection patterns - IMPROVED VERSION"""
        if isinstance(data, str):
            patterns = [
                r"'\s+or\s+['\"]?1['\"]?\s*=\s*['\"]?1",      # Basic SQL injection: admin' OR '1'='1
                r"'\s+or\s+true\b",                             # ' OR TRUE
                r"\bunion\s+all\s+select\b|\bunion\s+select\b", # Union select
                r"\binsert\s+into\b.*\bvalues\b",              # Insert statements
                r"\bdrop\s+table\b|\bdrop\s+database\b",       # Drop table
                r";\s*--|\bwaitfor\s+delay\b|\bsleep\(\d+\)",   # Stacked query / time delay
                r"\bbenchmark\(\d+,",                         # MySQL benchmark
                r"\bexec\s*\(|\bxp_cmdshell\b",                # Execution commands
                r"\bload_file\s*\(",                           # File reading
                r"\binto\s+outfile\b|\binto\s+dumpfile\b",     # File writing
            ]
            text = data.lower()
            return any(re.search(pattern, text, re.IGNORECASE) for pattern in patterns)
        elif isinstance(data, dict):
            return any(self._detect_sql_injection(str(v)) for v in data.values())
        elif isinstance(data, list):
            return any(self._detect_sql_injection(item) for item in data)
        return False
    
    def _detect_xss(self, data: Any) -> bool:
        """Detect XSS patterns"""
        if isinstance(data, str):
            patterns = [
                r"<script>",
                r"javascript:",
                r"onerror=",
                r"onload=",
                r"onclick=",
                r"alert\(",
                r"document\.cookie"
            ]
            text = data.lower()
            return any(re.search(pattern, text) for pattern in patterns)
        elif isinstance(data, dict):
            return any(self._detect_xss(str(v)) for v in data.values())
        elif isinstance(data, list):
            return any(self._detect_xss(item) for item in data)
        return False
    
    def _detect_path_traversal(self, path: str) -> bool:
        """Detect path traversal patterns"""
        patterns = [
            r"\.\./",
            r"\.\.\\",
            r"etc/passwd",
            r"win\.ini",
            r"\.\.%.*\.\.",
            r"\.\.0x2f"
        ]
        return any(re.search(pattern, path.lower()) for pattern in patterns)
    
    def _detect_command_injection(self, data: Any) -> bool:
        """Detect command injection patterns"""
        if isinstance(data, str):
            patterns = [
                r";\s*(rm|ls|cat|echo|wget|curl)",
                r"\|\s*(rm|ls|cat|echo|wget|curl)",
                r"&\s*(rm|ls|cat|echo|wget|curl)",
                r"`.*`",
                r"\$\(.*\)"
            ]
            text = data.lower()
            return any(re.search(pattern, text) for pattern in patterns)
        elif isinstance(data, dict):
            return any(self._detect_command_injection(str(v)) for v in data.values())
        return False
    
    # Helper methods
    def _calculate_object_complexity(self, obj: Any, depth: int = 0) -> int:
        if depth > 5:
            return 0
        if isinstance(obj, dict):
            return sum(self._calculate_object_complexity(v, depth + 1) for v in obj.values()) + len(obj)
        elif isinstance(obj, list):
            return sum(self._calculate_object_complexity(item, depth + 1) for item in obj) + len(obj)
        else:
            return 1
    
    def _analyze_timing_pattern(self, timestamp: float) -> float:
        if timestamp is None:
            return 0.5
        current_time = time.time()
        time_diff = current_time - timestamp
        return max(0.0, min(1.0, 1.0 - (time_diff / 3600)))
    
    def _analyze_request_frequency(self, client_ip: str) -> float:
        return 0.5
    
    def _analyze_behavioral_patterns(self, request_data: Dict[str, Any]) -> float:
        return 0.3