"""
Data Storage Layer for SmartCrypto
Handles all database and file operations
"""

import json
import os
import sqlite3
import threading
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class DataStorage:
    """
    Low-level data storage handler with dual-mode support:
    - SQLite database (primary, recommended)
    - JSONL files (fallback, portable)
    """
    
    def __init__(self, storage_path: str = "data/", use_db: bool = True):
        self.storage_path = Path(storage_path)
        self.use_db = use_db
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Database path
        self.db_path = self.storage_path / "smartcrypto.db"
        
        # File paths (fallback)
        self.signals_file = self.storage_path / "signals.jsonl"
        self.pattern_drawings_file = self.storage_path / "pattern_drawings.jsonl"
        self.performance_file = self.storage_path / "performance.json"
        self.patterns_file = self.storage_path / "patterns.json"
        
        # Initialize
        if self.use_db:
            self._init_database()
        else:
            self._init_files()
        
        logger.info(f"📦 DataStorage initialized (mode: {'DB' if use_db else 'Files'})")
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # INITIALIZATION
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    def _init_database(self):
        """Initialize SQLite database schema"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # SIGNALS TABLE
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS signals (
                signal_id TEXT PRIMARY KEY,
                symbol TEXT NOT NULL,
                action TEXT NOT NULL,
                price REAL NOT NULL,
                confidence REAL NOT NULL,
                signal_strength REAL NOT NULL,
                timestamp TEXT NOT NULL,
                outcome TEXT DEFAULT 'OPEN',
                pnl_percentage REAL,
                pnl REAL,
                entry_price REAL,
                exit_price REAL,
                exit_time TEXT,
                position_id TEXT,
                strategy TEXT,  -- JSON
                analysis TEXT,  -- JSON
                direction_1h TEXT,
                direction_4h TEXT,
                direction_1d TEXT,
                stop_loss REAL,
                take_profit REAL,
                max_holding_hours INTEGER DEFAULT 4,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Indexes for performance
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_signal_symbol_time ON signals(symbol, timestamp DESC)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_signal_outcome ON signals(outcome)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_signal_action ON signals(action)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_signal_confidence ON signals(confidence DESC)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_signal_timestamp ON signals(timestamp DESC)')
        
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # PATTERN DRAWINGS TABLE
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS pattern_drawings (
                pattern_id TEXT PRIMARY KEY,
                signal_id TEXT,
                symbol TEXT NOT NULL,
                pattern_type TEXT NOT NULL,
                action TEXT NOT NULL,
                price REAL,
                confidence REAL,
                signal_strength REAL,
                drawing_data TEXT,  -- JSON
                pattern_description TEXT,
                timestamp TEXT NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (signal_id) REFERENCES signals(signal_id)
            )
        ''')
        
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_pattern_symbol ON pattern_drawings(symbol)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_pattern_type ON pattern_drawings(pattern_type)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_pattern_created ON pattern_drawings(created_at DESC)')
        
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # MARKET DATA TABLE
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS market_data (
                symbol TEXT,
                timestamp TEXT,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume REAL,
                PRIMARY KEY (symbol, timestamp)
            )
        ''')
        
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_market_symbol_time ON market_data(symbol, timestamp DESC)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_market_time ON market_data(timestamp DESC)')
        
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # PERFORMANCE METRICS TABLE
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS performance_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT,
                timestamp TEXT NOT NULL,
                total_signals INTEGER,
                total_wins INTEGER,
                total_losses INTEGER,
                win_rate REAL,
                total_pnl REAL,
                avg_pnl REAL,
                best_trade REAL,
                worst_trade REAL,
                sharpe_ratio REAL,
                max_drawdown REAL,
                timeframe_accuracy TEXT,  -- JSON
                signal_type_accuracy TEXT, -- JSON
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_metrics_symbol ON performance_metrics(symbol)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_metrics_time ON performance_metrics(timestamp DESC)')
        
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # PATTERN STATISTICS TABLE
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS pattern_stats (
                pattern_type TEXT PRIMARY KEY,
                total_occurrences INTEGER DEFAULT 0,
                total_wins INTEGER DEFAULT 0,
                win_rate REAL DEFAULT 0,
                avg_pnl REAL DEFAULT 0,
                symbols TEXT,  -- JSON array
                last_updated TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
        logger.info(f"✅ Database initialized at: {self.db_path}")
    
    def _init_files(self):
        """Initialize file-based storage"""
        # Create empty JSONL files
        for file_path in [self.signals_file, self.pattern_drawings_file]:
            if not file_path.exists():
                file_path.touch()
        
        # Initialize JSON files
        if not self.performance_file.exists() or self.performance_file.stat().st_size == 0:
            with open(self.performance_file, 'w', encoding='utf-8') as f:
                json.dump(self._get_default_performance(), f, indent=2)
        
        if not self.patterns_file.exists() or self.patterns_file.stat().st_size == 0:
            with open(self.patterns_file, 'w', encoding='utf-8') as f:
                json.dump(self._get_default_patterns(), f, indent=2)
    
    def _get_default_performance(self) -> Dict:
        """Default performance dictionary"""
        return {
            "overall_accuracy": {},
            "symbol_performance": {},
            "timeframe_accuracy": {},
            "signal_type_accuracy": {},
            "last_updated": datetime.utcnow().isoformat()
        }
    
    def _get_default_patterns(self) -> Dict:
        """Default patterns dictionary"""
        return {
            "common_patterns": {},
            "successful_setups": {},
            "pattern_win_rates": {},
            "last_analyzed": datetime.utcnow().isoformat()
        }
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # SIGNAL OPERATIONS
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    def save_signal(self, signal: Dict) -> bool:
        """Save a signal to storage"""
        with self._lock:
            try:
                if self.use_db:
                    return self._save_signal_db(signal)
                else:
                    return self._save_signal_file(signal)
            except Exception as e:
                logger.error(f"Error saving signal: {e}")
                return False
    
    def _save_signal_db(self, signal: Dict) -> bool:
        """Save signal to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO signals (
                signal_id, symbol, action, price, confidence, signal_strength,
                timestamp, outcome, pnl_percentage, pnl, entry_price,
                exit_price, exit_time, position_id, strategy, analysis,
                direction_1h, direction_4h, direction_1d, stop_loss,
                take_profit, max_holding_hours
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            signal.get('signal_id'),
            signal.get('symbol'),
            signal.get('action'),
            signal.get('price'),
            signal.get('confidence'),
            signal.get('signal_strength'),
            signal.get('timestamp'),
            signal.get('outcome', 'OPEN'),
            signal.get('pnl_percentage'),
            signal.get('pnl'),
            signal.get('entry_price'),
            signal.get('exit_price'),
            signal.get('exit_time'),
            signal.get('position_id'),
            json.dumps(signal.get('strategy', {})),
            json.dumps(signal.get('analysis', {})),
            signal.get('direction_1h'),
            signal.get('direction_4h'),
            signal.get('direction_1d'),
            signal.get('stop_loss'),
            signal.get('take_profit'),
            signal.get('max_holding_hours', 4)
        ))
        
        conn.commit()
        conn.close()
        return True
    
    def _save_signal_file(self, signal: Dict) -> bool:
        """Save signal to JSONL file"""
        with open(self.signals_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(signal) + '\n')
        return True
    
    def update_signal(self, signal: Dict) -> bool:
        """Update an existing signal"""
        with self._lock:
            try:
                if self.use_db:
                    return self._update_signal_db(signal)
                else:
                    return self._update_signal_file(signal)
            except Exception as e:
                logger.error(f"Error updating signal: {e}")
                return False
    
    def _update_signal_db(self, signal: Dict) -> bool:
        """Update signal in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE signals SET
                outcome = ?, pnl = ?, pnl_percentage = ?,
                exit_price = ?, exit_time = ?, position_id = ?,
                updated_at = CURRENT_TIMESTAMP
            WHERE signal_id = ?
        ''', (
            signal.get('outcome'),
            signal.get('pnl'),
            signal.get('pnl_percentage'),
            signal.get('exit_price'),
            signal.get('exit_time'),
            signal.get('position_id'),
            signal.get('signal_id')
        ))
        
        conn.commit()
        conn.close()
        return True
    
    def _update_signal_file(self, signal: Dict) -> bool:
        """Update signal in file (rewrite entire file)"""
        # Read all signals
        signals = self.get_all_signals()
        
        # Update the target signal
        for i, s in enumerate(signals):
            if s.get('signal_id') == signal.get('signal_id'):
                signals[i] = signal
                break
        
        # Rewrite file
        with open(self.signals_file, 'w', encoding='utf-8') as f:
            for s in signals:
                f.write(json.dumps(s) + '\n')
        
        return True
    
    def get_signal(self, signal_id: str) -> Optional[Dict]:
        """Get a single signal by ID"""
        try:
            if self.use_db:
                return self._get_signal_db(signal_id)
            else:
                return self._get_signal_file(signal_id)
        except Exception as e:
            logger.error(f"Error getting signal: {e}")
            return None
    
    def _get_signal_db(self, signal_id: str) -> Optional[Dict]:
        """Get signal from database"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM signals WHERE signal_id = ?', (signal_id,))
        row = cursor.fetchone()
        conn.close()
        
        if row:
            signal = dict(row)
            # Parse JSON fields
            for field in ['strategy', 'analysis']:
                if signal.get(field):
                    try:
                        signal[field] = json.loads(signal[field])
                    except:
                        signal[field] = {}
            return signal
        return None
    
    def _get_signal_file(self, signal_id: str) -> Optional[Dict]:
        """Get signal from file"""
        if self.signals_file.exists():
            with open(self.signals_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        try:
                            signal = json.loads(line.strip())
                            if signal.get('signal_id') == signal_id:
                                return signal
                        except json.JSONDecodeError:
                            continue
        return None
    
    def get_signals(self, symbol: Optional[str] = None, 
                   hours: int = 24, limit: int = 100,
                   include_closed: bool = True) -> List[Dict]:
        """Get signals with filters"""
        try:
            if self.use_db:
                return self._get_signals_db(symbol, hours, limit, include_closed)
            else:
                return self._get_signals_file(symbol, hours, limit, include_closed)
        except Exception as e:
            logger.error(f"Error getting signals: {e}")
            return []
    
    def _get_signals_db(self, symbol: Optional[str], hours: int,
                       limit: int, include_closed: bool) -> List[Dict]:
        """Get signals from database"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        query = '''
            SELECT * FROM signals 
            WHERE timestamp > datetime('now', ?)
        '''
        params = [f'-{hours} hours']
        
        if symbol:
            query += ' AND symbol = ?'
            params.append(symbol)
        
        if not include_closed:
            query += ' AND outcome = "OPEN"'
        
        query += ' ORDER BY timestamp DESC LIMIT ?'
        params.append(limit)
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()
        
        signals = []
        for row in rows:
            signal = dict(row)
            # Parse JSON fields
            for field in ['strategy', 'analysis']:
                if signal.get(field):
                    try:
                        signal[field] = json.loads(signal[field])
                    except:
                        signal[field] = {}
            signals.append(signal)
        
        return signals
    
    def _get_signals_file(self, symbol: Optional[str], hours: int,
                         limit: int, include_closed: bool) -> List[Dict]:
        """Get signals from file"""
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        signals = []
        
        if self.signals_file.exists():
            with open(self.signals_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        try:
                            signal = json.loads(line.strip())
                            
                            # Apply filters
                            if symbol and signal.get('symbol') != symbol:
                                continue
                            if not include_closed and signal.get('outcome') != 'OPEN':
                                continue
                            
                            # Time filter
                            try:
                                ts = datetime.fromisoformat(signal['timestamp'].replace('Z', ''))
                                if ts < cutoff:
                                    continue
                            except:
                                pass
                            
                            signals.append(signal)
                        except json.JSONDecodeError:
                            continue
        
        # Sort and limit
        signals.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
        return signals[:limit]
    
    def get_all_signals(self) -> List[Dict]:
        """Get all signals (used for file rewrites)"""
        signals = []
        if self.signals_file.exists():
            with open(self.signals_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        try:
                            signals.append(json.loads(line.strip()))
                        except json.JSONDecodeError:
                            continue
        return signals
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # PATTERN DRAWING OPERATIONS
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    def save_pattern_drawing(self, drawing: Dict) -> bool:
        """Save a pattern drawing"""
        with self._lock:
            try:
                if self.use_db:
                    return self._save_pattern_db(drawing)
                else:
                    return self._save_pattern_file(drawing)
            except Exception as e:
                logger.error(f"Error saving pattern drawing: {e}")
                return False
    
    def _save_pattern_db(self, drawing: Dict) -> bool:
        """Save pattern to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO pattern_drawings (
                pattern_id, signal_id, symbol, pattern_type,
                action, price, confidence, signal_strength,
                drawing_data, pattern_description, timestamp
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            drawing.get('pattern_id'),
            drawing.get('signal_id'),
            drawing.get('symbol'),
            drawing.get('pattern_type'),
            drawing.get('action'),
            drawing.get('price'),
            drawing.get('confidence'),
            drawing.get('signal_strength'),
            json.dumps(drawing.get('drawing', {})),
            drawing.get('pattern_description'),
            drawing.get('timestamp')
        ))
        
        conn.commit()
        conn.close()
        return True
    
    def _save_pattern_file(self, drawing: Dict) -> bool:
        """Save pattern to file"""
        with open(self.pattern_drawings_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(drawing) + '\n')
        return True
    
    def get_pattern_drawings(self, symbol: Optional[str] = None,
                            pattern_type: Optional[str] = None,
                            hours: int = 24, limit: int = 100) -> List[Dict]:
        """Get pattern drawings with filters"""
        try:
            if self.use_db:
                return self._get_patterns_db(symbol, pattern_type, hours, limit)
            else:
                return self._get_patterns_file(symbol, pattern_type, hours, limit)
        except Exception as e:
            logger.error(f"Error getting pattern drawings: {e}")
            return []
    
    def _get_patterns_db(self, symbol: Optional[str], pattern_type: Optional[str],
                        hours: int, limit: int) -> List[Dict]:
        """Get patterns from database"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        query = '''
            SELECT * FROM pattern_drawings 
            WHERE created_at > datetime('now', ?)
        '''
        params = [f'-{hours} hours']
        
        if symbol:
            query += ' AND symbol = ?'
            params.append(symbol)
        if pattern_type:
            query += ' AND pattern_type = ?'
            params.append(pattern_type)
        
        query += ' ORDER BY created_at DESC LIMIT ?'
        params.append(limit)
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()
        
        drawings = []
        for row in rows:
            drawing = dict(row)
            if drawing.get('drawing_data'):
                try:
                    drawing['drawing'] = json.loads(drawing['drawing_data'])
                except:
                    drawing['drawing'] = {}
                del drawing['drawing_data']
            drawings.append(drawing)
        
        return drawings
    
    def _get_patterns_file(self, symbol: Optional[str], pattern_type: Optional[str],
                          hours: int, limit: int) -> List[Dict]:
        """Get patterns from file"""
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        drawings = []
        
        if self.pattern_drawings_file.exists():
            with open(self.pattern_drawings_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        try:
                            drawing = json.loads(line.strip())
                            
                            # Apply filters
                            if symbol and drawing.get('symbol') != symbol:
                                continue
                            if pattern_type and drawing.get('pattern_type') != pattern_type:
                                continue
                            
                            # Time filter
                            try:
                                ts = datetime.fromisoformat(drawing['created_at'].replace('Z', ''))
                                if ts < cutoff:
                                    continue
                            except:
                                pass
                            
                            drawings.append(drawing)
                        except json.JSONDecodeError:
                            continue
        
        drawings.sort(key=lambda x: x.get('created_at', ''), reverse=True)
        return drawings[:limit]
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # MARKET DATA OPERATIONS
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    def save_market_data(self, symbol: str, df: pd.DataFrame) -> bool:
        """Save market data to database"""
        if not self.use_db:
            logger.warning("Market data only available in DB mode")
            return False
        
        with self._lock:
            try:
                conn = sqlite3.connect(self.db_path)
                
                # Prepare data
                df = df.copy()
                if 'timestamp' not in df.columns:
                    df['timestamp'] = df.index
                
                df['symbol'] = symbol
                
                # Insert in chunks
                chunk_size = 1000
                for i in range(0, len(df), chunk_size):
                    chunk = df.iloc[i:i+chunk_size]
                    chunk.to_sql('market_data', conn, if_exists='append', index=False)
                
                conn.commit()
                conn.close()
                logger.info(f"💾 Saved {len(df)} market records for {symbol}")
                return True
                
            except Exception as e:
                logger.error(f"Error saving market data: {e}")
                return False
    
    def load_market_data(self, symbol: str, hours: int = 24) -> pd.DataFrame:
        """Load market data from database"""
        if not self.use_db:
            return pd.DataFrame()
        
        try:
            conn = sqlite3.connect(self.db_path)
            
            query = '''
                SELECT * FROM market_data 
                WHERE symbol = ? 
                AND timestamp > datetime('now', ?)
                ORDER BY timestamp ASC
            '''
            
            df = pd.read_sql_query(query, conn, params=[symbol, f'-{hours} hours'])
            conn.close()
            
            if not df.empty:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.set_index('timestamp', inplace=True)
                df = df.drop('symbol', axis=1)
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading market data: {e}")
            return pd.DataFrame()
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # PERFORMANCE OPERATIONS
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    def save_performance_metrics(self, metrics: Dict) -> bool:
        """Save performance metrics"""
        if not self.use_db:
            return self._save_performance_file(metrics)
        
        with self._lock:
            try:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT INTO performance_metrics (
                        symbol, timestamp, total_signals, total_wins, total_losses,
                        win_rate, total_pnl, avg_pnl, best_trade, worst_trade,
                        sharpe_ratio, max_drawdown, timeframe_accuracy, signal_type_accuracy
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    metrics.get('symbol'),
                    metrics.get('timestamp', datetime.utcnow().isoformat()),
                    metrics.get('total_signals', 0),
                    metrics.get('total_wins', 0),
                    metrics.get('total_losses', 0),
                    metrics.get('win_rate', 0),
                    metrics.get('total_pnl', 0),
                    metrics.get('avg_pnl', 0),
                    metrics.get('best_trade', 0),
                    metrics.get('worst_trade', 0),
                    metrics.get('sharpe_ratio', 0),
                    metrics.get('max_drawdown', 0),
                    json.dumps(metrics.get('timeframe_accuracy', {})),
                    json.dumps(metrics.get('signal_type_accuracy', {}))
                ))
                
                conn.commit()
                conn.close()
                return True
                
            except Exception as e:
                logger.error(f"Error saving performance metrics: {e}")
                return False
    
    def _save_performance_file(self, metrics: Dict) -> bool:
        """Save performance to JSON file"""
        try:
            # Read existing
            if self.performance_file.exists() and self.performance_file.stat().st_size > 0:
                with open(self.performance_file, 'r', encoding='utf-8') as f:
                    performance = json.load(f)
            else:
                performance = self._get_default_performance()
            
            # Update
            symbol = metrics.get('symbol')
            if symbol:
                if symbol not in performance['symbol_performance']:
                    performance['symbol_performance'][symbol] = {
                        "total_signals": 0, "wins": 0, "losses": 0,
                        "total_pnl": 0.0, "avg_pnl": 0.0
                    }
                
                perf = performance['symbol_performance'][symbol]
                perf['total_signals'] = metrics.get('total_signals', 0)
                perf['wins'] = metrics.get('total_wins', 0)
                perf['losses'] = metrics.get('total_losses', 0)
                perf['total_pnl'] = metrics.get('total_pnl', 0)
                perf['avg_pnl'] = metrics.get('avg_pnl', 0)
            
            performance['last_updated'] = datetime.utcnow().isoformat()
            
            with open(self.performance_file, 'w', encoding='utf-8') as f:
                json.dump(performance, f, indent=2)
            
            return True
            
        except Exception as e:
            logger.error(f"Error saving performance file: {e}")
            return False
    
    def get_performance(self, symbol: Optional[str] = None) -> Dict:
        """Get performance metrics"""
        try:
            if self.use_db:
                return self._get_performance_db(symbol)
            else:
                return self._get_performance_file(symbol)
        except Exception as e:
            logger.error(f"Error getting performance: {e}")
            return {}
    
    def _get_performance_db(self, symbol: Optional[str]) -> Dict:
        """Get performance from database"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        if symbol:
            cursor.execute('''
                SELECT * FROM performance_metrics 
                WHERE symbol = ? 
                ORDER BY timestamp DESC LIMIT 1
            ''', (symbol,))
        else:
            cursor.execute('''
                SELECT * FROM performance_metrics 
                ORDER BY timestamp DESC LIMIT 1
            ''')
        
        row = cursor.fetchone()
        conn.close()
        
        if row:
            metrics = dict(row)
            # Parse JSON fields
            for field in ['timeframe_accuracy', 'signal_type_accuracy']:
                if metrics.get(field):
                    try:
                        metrics[field] = json.loads(metrics[field])
                    except:
                        metrics[field] = {}
            return metrics
        return {}
    
    def _get_performance_file(self, symbol: Optional[str]) -> Dict:
        """Get performance from file"""
        if self.performance_file.exists() and self.performance_file.stat().st_size > 0:
            with open(self.performance_file, 'r', encoding='utf-8') as f:
                performance = json.load(f)
            
            if symbol:
                return performance.get('symbol_performance', {}).get(symbol, {})
            return performance
        return {}
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # PATTERN STATISTICS
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    def save_pattern_stats(self, pattern_stats: Dict) -> bool:
        """Save pattern statistics"""
        with self._lock:
            try:
                if self.use_db:
                    return self._save_pattern_stats_db(pattern_stats)
                else:
                    return self._save_pattern_stats_file(pattern_stats)
            except Exception as e:
                logger.error(f"Error saving pattern stats: {e}")
                return False
    
    def _save_pattern_stats_db(self, pattern_stats: Dict) -> bool:
        """Save pattern stats to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for pattern_type, stats in pattern_stats.items():
            cursor.execute('''
                INSERT OR REPLACE INTO pattern_stats (
                    pattern_type, total_occurrences, total_wins,
                    win_rate, avg_pnl, symbols, last_updated
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                pattern_type,
                stats.get('total', 0),
                stats.get('wins', 0),
                stats.get('win_rate', 0),
                stats.get('avg_pnl', 0),
                json.dumps(stats.get('symbols', [])),
                datetime.utcnow().isoformat()
            ))
        
        conn.commit()
        conn.close()
        return True
    
    def _save_pattern_stats_file(self, pattern_stats: Dict) -> bool:
        """Save pattern stats to file"""
        try:
            with open(self.patterns_file, 'r', encoding='utf-8') as f:
                patterns = json.load(f)
        except:
            patterns = self._get_default_patterns()
        
        patterns['common_patterns'] = pattern_stats
        patterns['last_analyzed'] = datetime.utcnow().isoformat()
        
        with open(self.patterns_file, 'w', encoding='utf-8') as f:
            json.dump(patterns, f, indent=2)
        
        return True
    
    def get_pattern_stats(self) -> Dict:
        """Get pattern statistics"""
        try:
            if self.use_db:
                return self._get_pattern_stats_db()
            else:
                return self._get_pattern_stats_file()
        except Exception as e:
            logger.error(f"Error getting pattern stats: {e}")
            return {}
    
    def _get_pattern_stats_db(self) -> Dict:
        """Get pattern stats from database"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM pattern_stats')
        rows = cursor.fetchall()
        conn.close()
        
        stats = {}
        for row in rows:
            stat = dict(row)
            if stat.get('symbols'):
                try:
                    stat['symbols'] = json.loads(stat['symbols'])
                except:
                    stat['symbols'] = []
            stats[stat['pattern_type']] = stat
        
        return stats
    
    def _get_pattern_stats_file(self) -> Dict:
        """Get pattern stats from file"""
        if self.patterns_file.exists() and self.patterns_file.stat().st_size > 0:
            with open(self.patterns_file, 'r', encoding='utf-8') as f:
                patterns = json.load(f)
            return patterns.get('common_patterns', {})
        return {}
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # CLEANUP & MAINTENANCE
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    def cleanup(self, days: int = 30) -> int:
        """Clean up old data and return number of records removed"""
        with self._lock:
            try:
                if self.use_db:
                    return self._cleanup_db(days)
                else:
                    return self._cleanup_files(days)
            except Exception as e:
                logger.error(f"Error cleaning up: {e}")
                return 0
    
    def _cleanup_db(self, days: int) -> int:
        """Clean up database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cutoff = (datetime.utcnow() - timedelta(days=days)).isoformat()
        total_removed = 0
        
        # Remove old closed signals
        cursor.execute('''
            DELETE FROM signals 
            WHERE outcome IN ('WIN', 'LOSS') 
            AND timestamp < ?
        ''', (cutoff,))
        total_removed += cursor.rowcount
        
        # Remove old market data (keep 90 days)
        cutoff_90 = (datetime.utcnow() - timedelta(days=90)).isoformat()
        cursor.execute('DELETE FROM market_data WHERE timestamp < ?', (cutoff_90,))
        total_removed += cursor.rowcount
        
        # Remove old pattern drawings
        cursor.execute('DELETE FROM pattern_drawings WHERE created_at < ?', (cutoff,))
        total_removed += cursor.rowcount
        
        # Remove old metrics
        cursor.execute('DELETE FROM performance_metrics WHERE timestamp < ?', (cutoff,))
        total_removed += cursor.rowcount
        
        conn.commit()
        conn.close()
        
        logger.info(f"🧹 Cleaned up {total_removed} records older than {days} days")
        return total_removed
    
    def _cleanup_files(self, days: int) -> int:
        """Clean up file-based storage"""
        cutoff = datetime.utcnow() - timedelta(days=days)
        removed = 0
        
        # Clean up signals file
        if self.signals_file.exists():
            keep = []
            with open(self.signals_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        try:
                            signal = json.loads(line.strip())
                            # Keep recent or open signals
                            try:
                                ts = datetime.fromisoformat(signal['timestamp'].replace('Z', ''))
                                if ts >= cutoff or signal.get('outcome') == 'OPEN':
                                    keep.append(signal)
                                else:
                                    removed += 1
                            except:
                                keep.append(signal)
                        except json.JSONDecodeError:
                            keep.append({})
            
            with open(self.signals_file, 'w', encoding='utf-8') as f:
                for signal in keep:
                    if signal:
                        f.write(json.dumps(signal) + '\n')
        
        return removed
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # UTILITY METHODS
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    def health_check(self) -> Dict:
        """Check storage health"""
        issues = []
        
        if self.use_db:
            try:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                cursor.execute('SELECT COUNT(*) FROM signals')
                count = cursor.fetchone()[0]
                conn.close()
            except Exception as e:
                issues.append(f"Database error: {e}")
        else:
            for file_path in [self.signals_file, self.pattern_drawings_file]:
                if not file_path.exists():
                    issues.append(f"Missing file: {file_path}")
        
        return {
            'healthy': len(issues) == 0,
            'issues': issues,
            'mode': 'database' if self.use_db else 'files',
            'db_path': str(self.db_path) if self.use_db else None
        }
    
    def export_data(self) -> Dict:
        """Export all data as dictionary"""
        return {
            'signals': self.get_signals(hours=8760, limit=10000),  # 1 year
            'pattern_drawings': self.get_pattern_drawings(hours=8760, limit=10000),
            'performance': self.get_performance(),
            'pattern_stats': self.get_pattern_stats(),
            'export_time': datetime.utcnow().isoformat(),
            'version': '1.0'
        }
    
    def import_data(self, data: Dict) -> int:
        """Import data from dictionary and return count"""
        count = 0
        for signal in data.get('signals', []):
            if self.save_signal(signal):
                count += 1
        return count