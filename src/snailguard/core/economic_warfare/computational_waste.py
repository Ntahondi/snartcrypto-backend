import time
import hashlib
import threading
from typing import List, Dict
import random

class ComputationalWaste:
    """Implements computational waste to drain attacker resources"""
    
    def __init__(self):
        self.waste_operations = []
    
    def trigger_waste_cycles(self, severity: str) -> Dict[str, any]:
        """Trigger computational waste based on threat severity"""
        if severity == 'low':
            cycles = 1000
            operation = self._light_computation
        elif severity == 'medium':
            cycles = 10000
            operation = self._medium_computation
        else:  # high/nuclear
            cycles = 100000
            operation = self._heavy_computation
        
        # Run in background thread to not block main request
        thread = threading.Thread(
            target=self._execute_waste_cycles,
            args=(cycles, operation)
        )
        thread.daemon = True
        thread.start()
        
        return {
            'type': 'computational_waste',
            'cycles_executed': cycles,
            'estimated_cpu_cost': cycles * 0.0001,  # $ per cycle
            'waste_thread_started': True
        }
    
    def _execute_waste_cycles(self, cycles: int, operation):
        """Execute waste cycles in background"""
        for i in range(cycles):
            operation(i)
            # Small delay to ensure CPU actually works
            if i % 1000 == 0:
                time.sleep(0.001)
    
    def _light_computation(self, i: int):
        """Light computational waste"""
        hashlib.sha256(f"waste_{i}_{random.random()}".encode()).hexdigest()
    
    def _medium_computation(self, i: int):
        """Medium computational waste"""
        for j in range(10):
            hashlib.sha512(f"waste_{i}_{j}_{random.random()}".encode()).hexdigest()
    
    def _heavy_computation(self, i: int):
        """Heavy computational waste - expensive!"""
        # Simulate matrix operations
        matrix_size = 50
        matrix_a = [[random.random() for _ in range(matrix_size)] for _ in range(matrix_size)]
        matrix_b = [[random.random() for _ in range(matrix_size)] for _ in range(matrix_size)]
        
        # Fake matrix multiplication (waste cycles)
        result = [[0 for _ in range(matrix_size)] for _ in range(matrix_size)]
        for x in range(matrix_size):
            for y in range(matrix_size):
                for z in range(matrix_size):
                    result[x][y] += matrix_a[x][z] * matrix_b[z][y]