from snailguard import SnailGuardDetector
import threading
import time
import random

def massive_attack_simulation():
    print("💥 MASSIVE ATTACK SIMULATION - STRESS TEST")
    print("==========================================")
    
    detector = SnailGuardDetector(
        api_key="SG-ENTERPRISE-ibzP6qoq7Lv1bOGQvDOnU4TVwAeNP4wV",
        config={'enable_economic_warfare': True}
    )
    
    attack_patterns = [
        # SQL Injection patterns
        "admin' OR '1'='1'--",
        "1 UNION SELECT 1,2,3,4",
        "'; DROP TABLE users--",
        "' OR 1=1--",
        
        # XSS patterns
        "<script>alert('XSS')</script>",
        "<img src=x onerror=alert(1)>",
        "javascript:alert(document.cookie)",
        "<svg onload=alert(1)>",
        
        # Path traversal
        "../../../etc/passwd",
        "..\\..\\windows\\system32\\config\\sam",
        "....//....//....//etc/passwd",
        
        # Command injection
        "; cat /etc/passwd",
        "| ls -la",
        "&& whoami",
        "`id`"
    ]
    
    results = {
        'total_attacks': 0,
        'threats_detected': 0,
        'total_economic_damage': 0,
        'attack_threads': []
    }
    
    def attack_worker(worker_id, num_attacks):
        worker_results = {
            'threats': 0,
            'damage': 0,
            'attacks': 0
        }
        
        for i in range(num_attacks):
            attack_pattern = random.choice(attack_patterns)
            attack_type = random.choice(['sql', 'xss', 'path', 'cmd'])
            
            request = {
                'method': random.choice(['GET', 'POST', 'PUT']),
                'path': f'/api/{random.choice(["login", "search", "files", "execute"])}',
                'headers': {'user-agent': f'AttackerBot-{worker_id}'},
                'body': {"input": attack_pattern} if random.choice([True, False]) else {},
                'client_ip': f'{random.randint(1,255)}.{random.randint(1,255)}.{random.randint(1,255)}.{random.randint(1,255)}'
            }
            
            result = detector.analyze_request(request)
            worker_results['attacks'] += 1
            
            if result.is_threat:
                worker_results['threats'] += 1
                damage = sum(a['estimated_attacker_cost'] for a in result.actions)
                worker_results['damage'] += damage
        
        return worker_results
    
    # Simulate massive concurrent attacks
    print("🚀 Launching 1000 concurrent attacks from 10 attacker threads...")
    start_time = time.time()
    
    threads = []
    for i in range(10):  # 10 attacker threads
        thread = threading.Thread(
            target=lambda i=i: results['attack_threads'].append(attack_worker(i, 100))
        )
        threads.append(thread)
        thread.start()
    
    for thread in threads:
        thread.join()
    
    total_time = time.time() - start_time
    
    # Aggregate results
    for thread_result in results['attack_threads']:
        results['total_attacks'] += thread_result['attacks']
        results['threats_detected'] += thread_result['threats']
        results['total_economic_damage'] += thread_result['damage']
    
    print(f"\n💥 MASSIVE ATTACK SIMULATION RESULTS")
    print("="*50)
    print(f"⏱️  Total simulation time: {total_time:.2f}s")
    print(f"🎯 Total attacks simulated: {results['total_attacks']}")
    print(f"🛡️  Threats detected: {results['threats_detected']}")
    print(f"📊 Detection rate: {(results['threats_detected']/results['total_attacks'])*100:.1f}%")
    print(f"💸 Total economic damage: ${results['total_economic_damage']:,}")
    print(f"⚡ Attacks per second: {results['total_attacks']/total_time:.1f}")
    print(f"💰 Economic damage per second: ${results['total_economic_damage']/total_time:,.0f}")
    
    print(f"\n🔥 ATTACKER SUFFERING METRICS:")
    print(f"  💀 Financial loss: ${results['total_economic_damage']:,}")
    print(f"  ⏳ Wasted attacker hours: {results['total_attacks'] * 0.5 / 3600:.1f} hours")
    print(f"  🔥 Cost per attack: ${results['total_economic_damage']/results['total_attacks']:,.0f}")
    print(f"  📉 Attack success rate: {((results['total_attacks']-results['threats_detected'])/results['total_attacks'])*100:.1f}%")

if __name__ == "__main__":
    massive_attack_simulation()