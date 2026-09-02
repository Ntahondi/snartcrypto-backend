import time
import random
import hashlib
import json
from typing import Dict, Any, List
import string

class DeceptiveResponseEngine:
    """
    Generates real-time deceptive responses to waste attacker time and resources
    """
    
    def __init__(self):
        self.response_templates = self._generate_templates()
    
    def generate_slow_deceptive_response(self, original_request: Dict, severity: str) -> Dict[str, Any]:
        """
        Generate deceptive response with intentional delays to maximize suffering
        """
        # Phase 1: Initial delay (frustration building)
        initial_delay = self._calculate_delay(severity, 'initial')
        time.sleep(initial_delay)
        
        # Phase 2: Generate deceptive data (CPU waste)
        deceptive_data = self._generate_deceptive_payload(original_request, severity)
        
        # Phase 3: Add processing simulation delay
        processing_delay = self._calculate_delay(severity, 'processing')
        time.sleep(processing_delay)
        
        # Phase 4: Add random jitter (unpredictable suffering)
        jitter_delay = random.uniform(0.5, 2.0)
        time.sleep(jitter_delay)
        
        return {
            'response': deceptive_data,
            'metadata': {
                'processing_time': initial_delay + processing_delay + jitter_delay,
                'server_load': random.randint(70, 95),
                'database_hits': random.randint(1000, 5000),
                'cache_misses': random.randint(50, 200),
                'suffering_level': severity
            }
        }
    
    def _calculate_delay(self, severity: str, phase: str) -> float:
        """Calculate intentional delays based on severity"""
        base_delays = {
            'low': {'initial': 1.0, 'processing': 2.0},
            'medium': {'initial': 3.0, 'processing': 5.0},
            'high': {'initial': 7.0, 'processing': 10.0},
            'nuclear': {'initial': 15.0, 'processing': 20.0}
        }
        
        base = base_delays.get(severity, base_delays['medium'])[phase]
        
        # Add random variation to make it unpredictable
        variation = random.uniform(0.7, 1.3)
        
        return base * variation
    
    def _generate_deceptive_payload(self, request: Dict, severity: str) -> Dict[str, Any]:
        """Generate realistic but fake response data in real-time"""
        
        method = request.get('method', 'GET')
        path = request.get('path', '')
        client_ip = request.get('client_ip', '0.0.0.0')
        
        # Analyze request to generate context-aware deception
        if any(sql_indicator in str(request).lower() for sql_indicator in ['select', 'union', 'drop', 'insert']):
            return self._generate_sql_deception(method, path, severity)
        elif any(xss_indicator in str(request).lower() for xss_indicator in ['script', 'alert', 'javascript']):
            return self._generate_xss_deception(method, path, severity)
        elif any(traversal_indicator in path for traversal_indicator in ['..', 'etc/', 'passwd', 'win.ini']):
            return self._generate_traversal_deception(method, path, severity)
        else:
            return self._generate_general_deception(method, path, severity)
    
    def _generate_sql_deception(self, method: str, path: str, severity: str) -> Dict[str, Any]:
        """Generate SQL injection deception"""
        # Simulate database structure discovery
        fake_tables = self._generate_fake_table_structure()
        fake_users = self._generate_fake_user_data()
        fake_errors = self._generate_fake_sql_errors()
        
        return {
            'success': random.choice([True, False]),
            'data': random.choice([fake_tables, fake_users]),
            'error': fake_errors if random.random() > 0.7 else None,
            'query_time': random.uniform(0.1, 2.5),
            'affected_rows': random.randint(1, 1000),
            'database': {
                'type': random.choice(['MySQL', 'PostgreSQL', 'SQLite', 'MongoDB']),
                'version': f"{random.randint(5, 15)}.{random.randint(0, 9)}.{random.randint(0, 99)}",
                'charset': random.choice(['utf8', 'utf8mb4', 'latin1'])
            },
            'debug_info': {
                'query': f"SELECT * FROM {random.choice(['users', 'admins', 'config'])}",
                'explain_plan': self._generate_fake_explain_plan(),
                'index_used': random.choice([True, False])
            }
        }
    
    def _generate_xss_deception(self, method: str, path: str, severity: str) -> Dict[str, Any]:
        """Generate XSS deception"""
        return {
            'status': 'success',
            'message': 'Comment submitted successfully',
            'user_input': f"<script>alert('XSS')</script>",
            'sanitized': '&lt;script&gt;alert(&#39;XSS&#39;)&lt;/script&gt;',
            'security': {
                'xss_protection': 'enabled',
                'content_security_policy': "default-src 'self'",
                'input_validation': 'strict'
            },
            'session': {
                'id': self._generate_random_hash(),
                'expires_in': random.randint(300, 3600),
                'secure': True
            }
        }
    
    def _generate_traversal_deception(self, method: str, path: str, severity: str) -> Dict[str, Any]:
        """Generate path traversal deception"""
        fake_files = self._generate_fake_file_structure()
        fake_permissions = self._generate_fake_permissions()
        
        return {
            'file_exists': random.choice([True, False]),
            'content': fake_files if random.random() > 0.5 else None,
            'permissions': fake_permissions,
            'filesystem': {
                'type': random.choice(['ext4', 'NTFS', 'APFS', 'FAT32']),
                'free_space': random.randint(1024, 1048576),  # MB
                'total_space': random.randint(1048576, 104857600)  # MB
            },
            'access_log': [
                f"{random.randint(1, 255)}.{random.randint(1, 255)}.{random.randint(1, 255)}.{random.randint(1, 255)} - - [{random.randint(1, 28)}/{random.choice(['Jan', 'Feb', 'Mar'])}/2024:{random.randint(0, 23)}:{random.randint(0, 59)}:{random.randint(0, 59)} +0000] \"GET {path} HTTP/1.1\" 200 {random.randint(100, 9999)}"
                for _ in range(random.randint(3, 10))
            ]
        }
    
    def _generate_general_deception(self, method: str, path: str, severity: str) -> Dict[str, Any]:
        """Generate general API deception"""
        return {
            'status': 'success',
            'data': {
                'id': random.randint(1, 10000),
                'name': self._generate_random_name(),
                'email': self._generate_random_email(),
                'created_at': f"2024-{random.randint(1, 12):02d}-{random.randint(1, 28):02d}T{random.randint(0, 23):02d}:{random.randint(0, 59):02d}:{random.randint(0, 59):02d}Z",
                'updated_at': f"2024-{random.randint(1, 12):02d}-{random.randint(1, 28):02d}T{random.randint(0, 23):02d}:{random.randint(0, 59):02d}:{random.randint(0, 59):02d}Z"
            },
            'pagination': {
                'page': random.randint(1, 10),
                'per_page': random.choice([10, 25, 50, 100]),
                'total': random.randint(100, 10000),
                'total_pages': random.randint(10, 100)
            },
            'rate_limit': {
                'limit': random.randint(100, 1000),
                'remaining': random.randint(0, 100),
                'reset': int(time.time()) + random.randint(300, 3600)
            }
        }
    
    def _generate_fake_table_structure(self) -> List[Dict]:
        """Generate fake database table structure"""
        tables = ['users', 'admins', 'sessions', 'config', 'logs', 'products', 'orders']
        return [
            {
                'table_name': table,
                'columns': [
                    {
                        'name': col,
                        'type': random.choice(['VARCHAR(255)', 'INT', 'TEXT', 'DATETIME', 'BOOLEAN']),
                        'nullable': random.choice([True, False]),
                        'key': random.choice(['PRI', 'MUL', 'UNI', '']) if i == 0 else ''
                    }
                    for i, col in enumerate(['id', 'name', 'email', 'password_hash', 'created_at', 'updated_at'])
                ],
                'row_count': random.randint(1000, 100000)
            }
            for table in random.sample(tables, random.randint(2, 4))
        ]
    
    def _generate_fake_user_data(self) -> List[Dict]:
        """Generate fake user data"""
        return [
            {
                'id': i,
                'username': self._generate_random_name().lower(),
                'email': self._generate_random_email(),
                'password_hash': self._generate_random_hash(),
                'created_at': f"2023-{random.randint(1, 12):02d}-{random.randint(1, 28):02d} {random.randint(0, 23):02d}:{random.randint(0, 59):02d}:{random.randint(0, 59):02d}",
                'last_login': f"2024-{random.randint(1, 12):02d}-{random.randint(1, 28):02d} {random.randint(0, 23):02d}:{random.randint(0, 59):02d}:{random.randint(0, 59):02d}" if random.random() > 0.3 else None,
                'is_active': random.choice([True, False])
            }
            for i in range(random.randint(5, 20))
        ]
    
    def _generate_fake_sql_errors(self) -> str:
        """Generate realistic SQL errors"""
        errors = [
            "Table 'database.users' doesn't exist",
            "You have an error in your SQL syntax",
            "Access denied for user 'root'@'localhost'",
            "Lock wait timeout exceeded",
            "Deadlock found when trying to get lock",
            "Can't connect to MySQL server on 'localhost' (10061)"
        ]
        return random.choice(errors)
    
    def _generate_fake_explain_plan(self) -> List[Dict]:
        """Generate fake SQL explain plan"""
        return [
            {
                'id': i,
                'select_type': random.choice(['SIMPLE', 'PRIMARY', 'SUBQUERY']),
                'table': random.choice(['users', 'orders', 'products']),
                'type': random.choice(['ALL', 'index', 'range', 'ref']),
                'possible_keys': 'PRIMARY,email_index,username_index',
                'key': random.choice(['PRIMARY', 'email_index', 'username_index']),
                'key_len': random.randint(10, 100),
                'rows': random.randint(100, 10000),
                'Extra': random.choice(['Using where', 'Using index', 'Using temporary', 'Using filesort'])
            }
            for i in range(random.randint(1, 3))
        ]
    
    def _generate_fake_file_structure(self) -> Dict[str, Any]:
        """Generate fake file system structure"""
        return {
            '/etc/passwd': f"root:x:0:0:root:/root:/bin/bash\n" +
                          f"daemon:x:1:1:daemon:/usr/sbin:/usr/sbin/nologin\n" +
                          f"bin:x:2:2:bin:/bin:/usr/sbin/nologin\n" +
                          f"sys:x:3:3:sys:/dev:/usr/sbin/nologin\n" +
                          f"sync:x:4:65534:sync:/bin:/bin/sync\n",
            '/etc/shadow': f"root:{self._generate_random_hash()}:18577:0:99999:7:::\n" +
                          f"daemon:*:18577:0:99999:7:::\n" +
                          f"bin:*:18577:0:99999:7:::\n",
            '/var/log/auth.log': f"Jan {random.randint(1, 28)} {random.randint(0, 23)}:{random.randint(0, 59)}:{random.randint(0, 59)} server sshd[{random.randint(1000, 9999)}]: Accepted password for root from {random.randint(1, 255)}.{random.randint(1, 255)}.{random.randint(1, 255)}.{random.randint(1, 255)} port {random.randint(1000, 9999)}\n"
        }
    
    def _generate_fake_permissions(self) -> Dict[str, str]:
        """Generate fake file permissions"""
        return {
            '/etc/passwd': '-rw-r--r-- 1 root root',
            '/etc/shadow': '-rw-r----- 1 root shadow',
            '/var/log/auth.log': '-rw-r--r-- 1 root adm'
        }
    
    def _generate_random_name(self) -> str:
        """Generate random names"""
        first_names = ['john', 'jane', 'alice', 'bob', 'charlie', 'diana', 'eve', 'frank']
        last_names = ['smith', 'johnson', 'williams', 'brown', 'jones', 'miller', 'davis']
        return f"{random.choice(first_names)} {random.choice(last_names)}"
    
    def _generate_random_email(self) -> str:
        """Generate random emails"""
        domains = ['gmail.com', 'yahoo.com', 'hotmail.com', 'company.com', 'example.org']
        return f"{self._generate_random_name().replace(' ', '.').lower()}@{random.choice(domains)}"
    
    def _generate_random_hash(self) -> str:
        """Generate random hash"""
        return hashlib.sha256(str(time.time() + random.random()).encode()).hexdigest()
    
    def _generate_templates(self) -> Dict[str, Any]:
        """Generate response templates"""
        return {
            'success': {'status': 'success', 'message': 'Operation completed successfully'},
            'error': {'status': 'error', 'message': 'An error occurred'},
            'pending': {'status': 'pending', 'message': 'Processing your request'}
        }