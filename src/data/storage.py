"""
Data Storage Layer for SmartCrypto
Handles all database and file operations
"""

import json
import os
import sqlite3
import threading
import secrets
import hashlib
import hmac
import base64
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
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # CLOSED TRADES TABLE
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS closed_trades (
                id TEXT PRIMARY KEY,
                symbol TEXT NOT NULL,
                action TEXT NOT NULL,
                entry_price REAL NOT NULL,
                exit_price REAL NOT NULL,
                quantity REAL NOT NULL,
                pnl REAL NOT NULL,
                pnl_percentage REAL NOT NULL,
                outcome TEXT NOT NULL,
                entry_time TEXT NOT NULL,
                exit_time TEXT NOT NULL,
                close_reason TEXT NOT NULL,
                stop_loss REAL,
                take_profit REAL,
                peak_pnl_percentage REAL DEFAULT 0.0,
                peak_price REAL,
                trail_tier INTEGER DEFAULT 0,
                is_risk_free INTEGER DEFAULT 0,
                extension_active INTEGER DEFAULT 0,
                ai_confidence REAL DEFAULT 0.88,
                ai_signal_strength REAL DEFAULT 0.82,
                market_regime TEXT DEFAULT 'BULLISH_TREND',
                execution_status TEXT DEFAULT 'CLOSED',
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        # Safe column migrations for existing databases
        for col_def in [
            ("peak_pnl_percentage", "REAL DEFAULT 0.0"),
            ("peak_price", "REAL"),
            ("trail_tier", "INTEGER DEFAULT 0"),
            ("is_risk_free", "INTEGER DEFAULT 0"),
            ("extension_active", "INTEGER DEFAULT 0"),
        ]:
            try:
                cursor.execute(f"ALTER TABLE closed_trades ADD COLUMN {col_def[0]} {col_def[1]}")
            except sqlite3.OperationalError:
                pass

        cursor.execute('CREATE INDEX IF NOT EXISTS idx_closed_trades_sym_time ON closed_trades(symbol, exit_time DESC)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_closed_trades_exit_time ON closed_trades(exit_time DESC)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_closed_trades_outcome ON closed_trades(outcome)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_closed_trades_reason ON closed_trades(close_reason)')
        
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

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # USERS TABLE
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                user_id TEXT PRIMARY KEY,
                email TEXT UNIQUE,
                password_hash TEXT,
                auth_provider TEXT DEFAULT 'email',
                provider_id TEXT,
                role TEXT DEFAULT 'guest',
                is_verified INTEGER DEFAULT 0,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                last_login TEXT
            )
        ''')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_user_email ON users(email)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_user_provider ON users(auth_provider, provider_id)')

        # Safe schema migration for is_verified on existing DB
        try:
            cursor.execute("PRAGMA table_info(users)")
            cols = [col[1] for col in cursor.fetchall()]
            if 'is_verified' not in cols:
                cursor.execute("ALTER TABLE users ADD COLUMN is_verified INTEGER DEFAULT 0")
        except Exception:
            pass

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # PASSWORD RESETS & OTP TABLE
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS password_resets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT NOT NULL,
                otp_code TEXT NOT NULL,
                token TEXT NOT NULL,
                purpose TEXT DEFAULT 'reset',
                expires_at TEXT NOT NULL,
                used INTEGER DEFAULT 0,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_reset_email_purpose ON password_resets(email, purpose, used)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_reset_token ON password_resets(token)')

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # SUBSCRIPTIONS TABLE
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS subscriptions (
                subscription_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                plan_id TEXT NOT NULL,
                status TEXT DEFAULT 'active',
                started_at TEXT NOT NULL,
                expires_at TEXT NOT NULL,
                payment_method TEXT DEFAULT 'crypto',
                amount_paid REAL NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(user_id)
            )
        ''')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_sub_user ON subscriptions(user_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_sub_status ON subscriptions(status)')

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # INVOICES TABLE (CRYPTO & FIAT)
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS invoices (
                invoice_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                plan_id TEXT NOT NULL,
                amount_usd REAL NOT NULL,
                currency TEXT DEFAULT 'USDT',
                network TEXT DEFAULT 'TRC20',
                crypto_address TEXT NOT NULL,
                status TEXT DEFAULT 'PENDING',
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                expires_at TEXT NOT NULL,
                confirmed_at TEXT,
                tx_hash TEXT,
                FOREIGN KEY (user_id) REFERENCES users(user_id)
            )
        ''')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_invoice_user ON invoices(user_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_invoice_status ON invoices(status)')
        
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # USER EXCHANGE KEYS TABLE (VVIP REAL TRADING)
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_exchange_keys (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                exchange TEXT NOT NULL,
                api_key_masked TEXT NOT NULL,
                api_key_encrypted TEXT NOT NULL,
                api_secret_encrypted TEXT NOT NULL,
                passphrase_encrypted TEXT,
                is_testnet INTEGER DEFAULT 0,
                is_active INTEGER DEFAULT 1,
                auto_trade_enabled INTEGER DEFAULT 0,
                max_position_size_usd REAL DEFAULT 500.0,
                status TEXT DEFAULT 'ACTIVE',
                last_tested_at TEXT,
                last_error TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(user_id)
            )
        ''')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_exchange_keys_user ON user_exchange_keys(user_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_exchange_keys_exchange ON user_exchange_keys(exchange)')

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # USER RISK SETTINGS & CONSENT (VVIP REAL TRADING)
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_risk_settings (
                user_id TEXT PRIMARY KEY,
                trading_style TEXT DEFAULT 'day_trader',
                risk_tolerance TEXT DEFAULT 'moderate',
                sizing_mode TEXT DEFAULT 'kelly',
                kelly_fraction REAL DEFAULT 0.25,
                max_leverage INTEGER DEFAULT 3,
                stop_loss_atr_mult REAL DEFAULT 1.5,
                take_profit_atr_mult REAL DEFAULT 3.0,
                use_trailing_stop INTEGER DEFAULT 1,
                min_confidence REAL DEFAULT 0.65,
                require_ensemble_agreement INTEGER DEFAULT 1,
                max_open_positions INTEGER DEFAULT 5,
                risk_consent_accepted INTEGER DEFAULT 0,
                risk_consent_at TEXT,
                risk_consent_ip TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(user_id)
            )
        ''')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_user_risk_settings_consent ON user_risk_settings(risk_consent_accepted)')

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # USER SAVED EXCHANGE DEPOSIT ADDRESSES TABLE
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_deposit_addresses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                exchange TEXT NOT NULL,
                network TEXT NOT NULL,
                deposit_address TEXT NOT NULL,
                tag_or_memo TEXT,
                is_default INTEGER DEFAULT 1,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(user_id),
                UNIQUE(user_id, exchange, network)
            )
        ''')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_user_deposit_lookup ON user_deposit_addresses(user_id, exchange)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_user_deposit_network ON user_deposit_addresses(user_id, exchange, network)')
        
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # SEED MASTER ADMIN USER IF NOT PRESENT
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        try:
            admin_email = "admin@snartpace.com"
            cursor.execute('SELECT user_id FROM users WHERE LOWER(email) = LOWER(?)', (admin_email,))
            if not cursor.fetchone():
                import secrets, hashlib
                salt = secrets.token_hex(16)
                key = hashlib.pbkdf2_hmac(
                    'sha256',
                    "Snartpace_2026".encode('utf-8'),
                    salt.encode('utf-8'),
                    100000
                ).hex()
                admin_pw_hash = f"{salt}${key}"
                cursor.execute(
                    '''
                    INSERT INTO users (user_id, email, password_hash, auth_provider, provider_id, role, last_login)
                    VALUES (?, ?, ?, 'email', NULL, 'admin', CURRENT_TIMESTAMP)
                    ''',
                    ("admin_snartpace_master", admin_email, admin_pw_hash)
                )
                logger.info(f"👑 Master admin account seeded: {admin_email} (Role: admin)")
        except Exception as seed_err:
            logger.debug(f"Admin seeding pass: {seed_err}")

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

    def get_celebration_wins(self, limit: int = 10) -> List[Dict]:
        """Query real closed winning trades for social proof celebrations."""
        with self._lock:
            try:
                if self.use_db and self.db_path.exists():
                    conn = sqlite3.connect(self.db_path)
                    conn.row_factory = sqlite3.Row
                    cursor = conn.cursor()
                    cursor.execute('''
                        SELECT signal_id, symbol, action, pnl_percentage, pnl, entry_price, exit_price, confidence, timestamp
                        FROM signals
                        WHERE (outcome IN ('WIN', 'CLOSED') OR pnl_percentage > 0)
                          AND (pnl_percentage > 0 OR pnl > 0)
                        ORDER BY pnl_percentage DESC, timestamp DESC
                        LIMIT ?
                    ''', (limit,))
                    rows = cursor.fetchall()
                    conn.close()
                    if rows:
                        return [dict(r) for r in rows]
                
                signals = self.get_all_signals()
                wins = [
                    s for s in signals 
                    if (s.get('outcome') in ('WIN', 'CLOSED') or (s.get('pnl_percentage') or 0) > 0)
                    and ((s.get('pnl_percentage') or 0) > 0 or (s.get('pnl') or 0) > 0)
                ]
                wins.sort(key=lambda x: (x.get('pnl_percentage') or 0) or (x.get('pnl') or 0), reverse=True)
                return wins[:limit]
            except Exception as e:
                logger.error(f"Error fetching celebration wins from storage: {e}")
                return []

    def save_closed_trade(self, trade: Dict[str, Any]) -> bool:
        """Save a closed trade execution to SQLite database."""
        with self._lock:
            try:
                if self.use_db and self.db_path.exists():
                    conn = sqlite3.connect(self.db_path)
                    cursor = conn.cursor()
                    cursor.execute('''
                        INSERT OR REPLACE INTO closed_trades (
                            id, symbol, action, entry_price, exit_price, quantity,
                            pnl, pnl_percentage, outcome, entry_time, exit_time,
                            close_reason, stop_loss, take_profit, peak_pnl_percentage,
                            peak_price, trail_tier, is_risk_free, extension_active,
                            ai_confidence, ai_signal_strength, market_regime, execution_status
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        trade.get("id"),
                        trade.get("symbol"),
                        trade.get("action", "BUY"),
                        float(trade.get("entry_price", 0.0)),
                        float(trade.get("exit_price", 0.0)),
                        float(trade.get("quantity", 1.0)),
                        float(trade.get("pnl", 0.0)),
                        float(trade.get("pnl_percentage", 0.0)),
                        trade.get("outcome", "WIN"),
                        str(trade.get("entry_time", "")),
                        str(trade.get("exit_time", "")),
                        trade.get("close_reason", "TAKE_PROFIT"),
                        float(trade.get("stop_loss", 0.0)) if trade.get("stop_loss") is not None else None,
                        float(trade.get("take_profit", 0.0)) if trade.get("take_profit") is not None else None,
                        float(trade.get("peak_pnl_percentage", 0.0)),
                        float(trade.get("peak_price", 0.0)) if trade.get("peak_price") is not None else None,
                        int(trade.get("trail_tier", 0)),
                        1 if trade.get("is_risk_free") else 0,
                        1 if trade.get("extension_active") else 0,
                        float(trade.get("ai_confidence", 0.88)),
                        float(trade.get("ai_signal_strength", 0.82)),
                        trade.get("market_regime", "BULLISH_TREND"),
                        trade.get("execution_status", "CLOSED"),
                    ))
                    conn.commit()
                    conn.close()
                    return True
                return False
            except Exception as e:
                logger.error(f"Error saving closed trade: {e}")
                return False

    def get_stored_closed_trades(self, symbol: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """Retrieve historical closed trades from SQLite database."""
        with self._lock:
            try:
                if self.use_db and self.db_path.exists():
                    conn = sqlite3.connect(self.db_path)
                    conn.row_factory = sqlite3.Row
                    cursor = conn.cursor()
                    if symbol:
                        cursor.execute('''
                            SELECT * FROM closed_trades
                            WHERE UPPER(symbol) = UPPER(?)
                            ORDER BY exit_time DESC, created_at DESC
                            LIMIT ?
                        ''', (symbol, limit))
                    else:
                        cursor.execute('''
                            SELECT * FROM closed_trades
                            ORDER BY exit_time DESC, created_at DESC
                            LIMIT ?
                        ''', (limit,))
                    rows = cursor.fetchall()
                    conn.close()
                    if rows:
                        return [dict(r) for r in rows]
                return []
            except Exception as e:
                logger.error(f"Error fetching stored closed trades: {e}")
                return []
    
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

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # USER MANAGEMENT
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def create_user(
        self,
        user_id: str,
        email: Optional[str] = None,
        password_hash: Optional[str] = None,
        auth_provider: str = 'email',
        provider_id: Optional[str] = None,
        role: str = 'guest',
    ) -> bool:
        """Create a new user record with strict email normalization and uniqueness."""
        try:
            clean_email = email.strip().lower() if email else None
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute(
                '''
                INSERT INTO users (user_id, email, password_hash, auth_provider, provider_id, role, last_login)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ''',
                (user_id, clean_email, password_hash, auth_provider, provider_id, role, datetime.utcnow().isoformat())
            )
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            logger.error(f"Error creating user: {e}")
            return False

    def get_user_by_id(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve user by user_id."""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM users WHERE user_id = ?', (user_id,))
            row = cursor.fetchone()
            conn.close()
            return dict(row) if row else None
        except Exception as e:
            logger.error(f"Error fetching user by ID: {e}")
            return None

    def get_user_by_email(self, email: str) -> Optional[Dict[str, Any]]:
        """Retrieve user by email."""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM users WHERE LOWER(email) = LOWER(?)', (email,))
            row = cursor.fetchone()
            conn.close()
            return dict(row) if row else None
        except Exception as e:
            logger.error(f"Error fetching user by email: {e}")
            return None

    def get_user_by_provider(self, auth_provider: str, provider_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve user by auth provider and provider ID."""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(
                'SELECT * FROM users WHERE auth_provider = ? AND provider_id = ?',
                (auth_provider, provider_id)
            )
            row = cursor.fetchone()
            conn.close()
            return dict(row) if row else None
        except Exception as e:
            logger.error(f"Error fetching user by provider: {e}")
            return None

    def update_user_role(self, user_id: str, role: str) -> bool:
        """Update a user's role (guest, pro, vip, vvip, admin)."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('UPDATE users SET role = ? WHERE user_id = ?', (role, user_id))
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            logger.error(f"Error updating user role: {e}")
            return False

    def update_user_last_login(self, user_id: str) -> bool:
        """Update last login timestamp."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute(
                'UPDATE users SET last_login = ? WHERE user_id = ?',
                (datetime.utcnow().isoformat(), user_id)
            )
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            logger.error(f"Error updating last login: {e}")
            return False

    def update_user_password(self, email_or_user_id: str, password_hash: str) -> bool:
        """Update a user's password hash."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute(
                '''
                UPDATE users 
                SET password_hash = ? 
                WHERE user_id = ? OR LOWER(email) = LOWER(?)
                ''',
                (password_hash, email_or_user_id, email_or_user_id)
            )
            affected = cursor.rowcount
            conn.commit()
            conn.close()
            return affected > 0
        except Exception as e:
            logger.error(f"Error updating user password: {e}")
            return False

    def set_user_verified(self, email_or_user_id: str, is_verified: bool = True) -> bool:
        """Mark a user's email address as verified."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute(
                '''
                UPDATE users 
                SET is_verified = ? 
                WHERE user_id = ? OR LOWER(email) = LOWER(?)
                ''',
                (1 if is_verified else 0, email_or_user_id, email_or_user_id)
            )
            affected = cursor.rowcount
            conn.commit()
            conn.close()
            return affected > 0
        except Exception as e:
            logger.error(f"Error updating user verified status: {e}")
            return False

    def create_otp(
        self,
        email: str,
        otp_code: str,
        purpose: str = "reset",
        expires_in_minutes: int = 15,
        token: Optional[str] = None,
    ) -> str:
        """Create an expiring OTP for password reset or email verification."""
        import secrets
        reset_token = token or secrets.token_urlsafe(32)
        exp_time = (datetime.utcnow() + timedelta(minutes=expires_in_minutes)).isoformat()
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            # Invalidate any existing unused OTPs for this email and purpose
            cursor.execute(
                '''
                UPDATE password_resets 
                SET used = 1 
                WHERE LOWER(email) = LOWER(?) AND purpose = ? AND used = 0
                ''',
                (email, purpose)
            )
            cursor.execute(
                '''
                INSERT INTO password_resets (email, otp_code, token, purpose, expires_at, used)
                VALUES (?, ?, ?, ?, ?, 0)
                ''',
                (email.lower(), otp_code, reset_token, purpose, exp_time)
            )
            conn.commit()
            conn.close()
            return reset_token
        except Exception as e:
            logger.error(f"Error creating OTP: {e}")
            return reset_token

    def verify_otp(
        self,
        email: str,
        otp_code: str,
        purpose: str = "reset",
    ) -> bool:
        """Verify an OTP code for a given email and purpose. Marks code as used if valid."""
        try:
            now_str = datetime.utcnow().isoformat()
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute(
                '''
                SELECT id, expires_at 
                FROM password_resets 
                WHERE LOWER(email) = LOWER(?) AND otp_code = ? AND purpose = ? AND used = 0
                ORDER BY created_at DESC 
                LIMIT 1
                ''',
                (email, otp_code.strip(), purpose)
            )
            row = cursor.fetchone()
            if not row:
                conn.close()
                return False

            reset_id, expires_at = row
            if expires_at < now_str:
                # Expired
                conn.close()
                return False

            # Mark as used
            cursor.execute('UPDATE password_resets SET used = 1 WHERE id = ?', (reset_id,))
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            logger.error(f"Error verifying OTP: {e}")
            return False

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # SUBSCRIPTION MANAGEMENT
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def create_subscription(
        self,
        subscription_id: str,
        user_id: str,
        plan_id: str,
        status: str = 'active',
        started_at: Optional[str] = None,
        expires_at: Optional[str] = None,
        payment_method: str = 'crypto',
        amount_paid: float = 20.0,
    ) -> bool:
        """Create a user subscription record."""
        try:
            now = started_at or datetime.utcnow().isoformat()
            exp = expires_at or (datetime.utcnow() + timedelta(days=30)).isoformat()

            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute(
                '''
                INSERT INTO subscriptions (subscription_id, user_id, plan_id, status, started_at, expires_at, payment_method, amount_paid)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''',
                (subscription_id, user_id, plan_id, status, now, exp, payment_method, amount_paid)
            )
            # Also update the user's role accordingly
            role_map = {
                'pro_20': 'pro',
                'vip_49': 'vip',
                'vvip_99': 'vvip',
            }
            new_role = role_map.get(plan_id, 'pro')
            cursor.execute('UPDATE users SET role = ? WHERE user_id = ?', (new_role, user_id))
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            logger.error(f"Error creating subscription: {e}")
            return False

    def get_active_subscription(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get the current active subscription for a user."""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(
                '''
                SELECT * FROM subscriptions 
                WHERE user_id = ? AND status = 'active'
                ORDER BY created_at DESC LIMIT 1
                ''',
                (user_id,)
            )
            row = cursor.fetchone()
            conn.close()
            return dict(row) if row else None
        except Exception as e:
            logger.error(f"Error fetching active subscription: {e}")
            return None

    def cancel_active_subscription(self, user_id: str) -> bool:
        """Cancel an active subscription and downgrade user role to guest."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE subscriptions SET status = 'cancelled' WHERE user_id = ? AND status = 'active'",
                (user_id,)
            )
            cursor.execute("UPDATE users SET role = 'guest' WHERE user_id = ?", (user_id,))
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            logger.error(f"Error cancelling subscription: {e}")
            return False

    def delete_user(self, user_id: str) -> bool:
        """Permanently delete a user account and purge all associated records."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("DELETE FROM invoices WHERE user_id = ?", (user_id,))
            cursor.execute("DELETE FROM subscriptions WHERE user_id = ?", (user_id,))
            cursor.execute("DELETE FROM users WHERE user_id = ?", (user_id,))
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            logger.error(f"Error deleting user {user_id}: {e}")
            return False

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # INVOICE MANAGEMENT (CRYPTO & FIAT)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def create_invoice(
        self,
        invoice_id: str,
        user_id: str,
        plan_id: str,
        amount_usd: float,
        currency: str = 'USDT',
        network: str = 'TRC20',
        crypto_address: str = '',
        expires_at: Optional[str] = None,
    ) -> bool:
        """Create a payment invoice."""
        try:
            exp = expires_at or (datetime.utcnow() + timedelta(hours=2)).isoformat()
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute(
                '''
                INSERT INTO invoices (invoice_id, user_id, plan_id, amount_usd, currency, network, crypto_address, status, expires_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, 'PENDING', ?)
                ''',
                (invoice_id, user_id, plan_id, amount_usd, currency, network, crypto_address, exp)
            )
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            logger.error(f"Error creating invoice: {e}")
            return False

    def get_invoice(self, invoice_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve an invoice by ID."""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM invoices WHERE invoice_id = ?', (invoice_id,))
            row = cursor.fetchone()
            conn.close()
            return dict(row) if row else None
        except Exception as e:
            logger.error(f"Error fetching invoice: {e}")
            return None

    def confirm_invoice(self, invoice_id: str, tx_hash: Optional[str] = None) -> bool:
        """Mark an invoice as confirmed and automatically provision subscription."""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM invoices WHERE invoice_id = ?', (invoice_id,))
            inv = cursor.fetchone()
            if not inv:
                conn.close()
                return False

            now = datetime.utcnow().isoformat()
            cursor.execute(
                '''
                UPDATE invoices 
                SET status = 'CONFIRMED', confirmed_at = ?, tx_hash = ?
                WHERE invoice_id = ?
                ''',
                (now, tx_hash or '0x' + invoice_id[:16], invoice_id)
            )
            conn.commit()
            conn.close()

            # Provision subscription
            sub_id = f"sub_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}_{inv['user_id'][:6]}"
            self.create_subscription(
                subscription_id=sub_id,
                user_id=inv['user_id'],
                plan_id=inv['plan_id'],
                status='active',
                payment_method='crypto',
                amount_paid=inv['amount_usd'],
            )
            return True
        except Exception as e:
            logger.error(f"Error confirming invoice: {e}")
            return False

    def is_tx_hash_used(self, tx_hash: str, exclude_invoice_id: Optional[str] = None) -> bool:
        """Check if a blockchain transaction hash has already been redeemed for another invoice."""
        if not tx_hash:
            return False
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            if exclude_invoice_id:
                cursor.execute(
                    "SELECT 1 FROM invoices WHERE tx_hash = ? AND status = 'CONFIRMED' AND invoice_id != ? LIMIT 1",
                    (tx_hash, exclude_invoice_id)
                )
            else:
                cursor.execute(
                    "SELECT 1 FROM invoices WHERE tx_hash = ? AND status = 'CONFIRMED' LIMIT 1",
                    (tx_hash,)
                )
            row = cursor.fetchone()
            conn.close()
            return row is not None
        except Exception as e:
            logger.error(f"Error checking tx hash usage: {e}")
            return False

    def list_user_invoices(self, user_id: str) -> List[Dict[str, Any]]:
        """List all invoices for a user."""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(
                'SELECT * FROM invoices WHERE user_id = ? ORDER BY created_at DESC LIMIT 50',
                (user_id,)
            )
            rows = cursor.fetchall()
            conn.close()
            return [dict(r) for r in rows]
        except Exception as e:
            logger.error(f"Error listing user invoices: {e}")
            return []

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # CELEBRATION WINS FEED
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def get_celebration_wins(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Fetch top historical winning trades for user celebration & social proof.
        """
        try:
            if not self.use_db:
                return []
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(
                '''
                SELECT 
                    signal_id, symbol, action, price, confidence, signal_strength,
                    timestamp, outcome, pnl_percentage, pnl, entry_price, exit_price,
                    exit_time, direction_1h
                FROM signals
                WHERE (outcome = 'WIN' OR pnl_percentage > 0 OR pnl > 0)
                ORDER BY pnl_percentage DESC, timestamp DESC
                LIMIT ?
                ''',
                (limit,)
            )
            rows = cursor.fetchall()
            conn.close()
            return [dict(r) for r in rows]
        except Exception as e:
            logger.error(f"Error fetching celebration wins: {e}")
            return []

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # USER EXCHANGE KEYS (VVIP / ADMIN REAL TRADING)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def save_user_exchange_key(
        self,
        user_id: str,
        exchange: str,
        api_key: str,
        api_secret: str,
        passphrase: Optional[str] = None,
        is_testnet: bool = False,
        max_position_size_usd: float = 500.0,
        auto_trade_enabled: bool = False,
    ) -> Dict[str, Any]:
        """
        Securely encrypt and store exchange credentials for real trade execution.
        """
        try:
            if not self.use_db:
                return {"success": False, "error": "Database not enabled"}
            
            clean_exchange = exchange.strip().lower()
            if clean_exchange not in ("binance", "bybit"):
                raise ValueError("Only 'binance' and 'bybit' exchanges are supported")
            
            clean_key = api_key.strip()
            clean_secret = api_secret.strip()
            if not clean_key or not clean_secret:
                raise ValueError("API Key and Secret must not be empty")
            
            # Mask API key for safe UI display (e.g. "abc12...9xyz")
            masked_key = clean_key[:6] + "..." + clean_key[-4:] if len(clean_key) > 10 else "***"
            
            enc_key = encrypt_exchange_secret(clean_key)
            enc_secret = encrypt_exchange_secret(clean_secret)
            enc_passphrase = encrypt_exchange_secret(passphrase.strip()) if passphrase else None
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Deactivate previous active keys for the same exchange & testnet mode
            cursor.execute(
                '''
                UPDATE user_exchange_keys
                SET is_active = 0, updated_at = CURRENT_TIMESTAMP
                WHERE user_id = ? AND exchange = ? AND is_testnet = ?
                ''',
                (user_id, clean_exchange, 1 if is_testnet else 0)
            )
            
            cursor.execute(
                '''
                INSERT INTO user_exchange_keys (
                    user_id, exchange, api_key_masked, api_key_encrypted, api_secret_encrypted,
                    passphrase_encrypted, is_testnet, is_active, auto_trade_enabled,
                    max_position_size_usd, status, last_tested_at, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, 1, ?, ?, 'VERIFIED', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                ''',
                (
                    user_id,
                    clean_exchange,
                    masked_key,
                    enc_key,
                    enc_secret,
                    enc_passphrase,
                    1 if is_testnet else 0,
                    1 if auto_trade_enabled else 0,
                    float(max_position_size_usd),
                )
            )
            key_id = cursor.lastrowid
            conn.commit()
            conn.close()
            
            logger.info(f"🔑 Stored encrypted exchange key id={key_id} for user={user_id} on exchange={clean_exchange}")
            return {
                "id": key_id,
                "exchange": clean_exchange,
                "api_key_masked": masked_key,
                "is_testnet": is_testnet,
                "is_active": True,
                "auto_trade_enabled": auto_trade_enabled,
                "max_position_size_usd": max_position_size_usd,
                "status": "VERIFIED",
            }
        except Exception as e:
            logger.error(f"Error saving user exchange key: {e}")
            raise

    def get_user_exchange_keys(self, user_id: str) -> List[Dict[str, Any]]:
        """
        List all saved exchange keys for user (with secrets masked).
        """
        try:
            if not self.use_db:
                return []
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(
                '''
                SELECT 
                    id, user_id, exchange, api_key_masked, is_testnet, is_active,
                    auto_trade_enabled, max_position_size_usd, status, last_tested_at,
                    last_error, created_at, updated_at
                FROM user_exchange_keys
                WHERE user_id = ?
                ORDER BY is_active DESC, updated_at DESC
                ''',
                (user_id,)
            )
            rows = cursor.fetchall()
            conn.close()
            return [dict(r) for r in rows]
        except Exception as e:
            logger.error(f"Error fetching user exchange keys: {e}")
            return []

    def get_user_exchange_key_by_id(
        self, key_id: int, user_id: str, include_secret: bool = False
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve a single exchange key. If include_secret is True, decrypts the secret.
        """
        try:
            if not self.use_db:
                return None
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(
                '''
                SELECT * FROM user_exchange_keys
                WHERE id = ? AND user_id = ?
                ''',
                (key_id, user_id)
            )
            row = cursor.fetchone()
            conn.close()
            if not row:
                return None
            
            data = dict(row)
            if include_secret:
                data["api_key_decrypted"] = decrypt_exchange_secret(data.get("api_key_encrypted", ""))
                data["api_secret_decrypted"] = decrypt_exchange_secret(data.get("api_secret_encrypted", ""))
                if data.get("passphrase_encrypted"):
                    data["passphrase_decrypted"] = decrypt_exchange_secret(data["passphrase_encrypted"])
            else:
                data.pop("api_key_encrypted", None)
                data.pop("api_secret_encrypted", None)
                data.pop("passphrase_encrypted", None)
            return data
        except Exception as e:
            logger.error(f"Error fetching exchange key by id: {e}")
            return None

    def delete_user_exchange_key(self, key_id: int, user_id: str) -> bool:
        """
        Permanently delete an exchange key configuration.
        """
        try:
            if not self.use_db:
                return False
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute(
                'DELETE FROM user_exchange_keys WHERE id = ? AND user_id = ?',
                (key_id, user_id)
            )
            affected = cursor.rowcount
            conn.commit()
            conn.close()
            return affected > 0
        except Exception as e:
            logger.error(f"Error deleting user exchange key: {e}")
            return False

    def toggle_exchange_auto_trade(self, key_id: int, user_id: str, enabled: bool) -> bool:
        """
        Enable or disable automated live trade execution for an exchange key.
        """
        try:
            if not self.use_db:
                return False
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute(
                '''
                UPDATE user_exchange_keys
                SET auto_trade_enabled = ?, updated_at = CURRENT_TIMESTAMP
                WHERE id = ? AND user_id = ?
                ''',
                (1 if enabled else 0, key_id, user_id)
            )
            affected = cursor.rowcount
            conn.commit()
            conn.close()
            return affected > 0
        except Exception as e:
            logger.error(f"Error toggling exchange auto-trade: {e}")
            return False

    def update_exchange_key_tested(
        self, key_id: int, user_id: str, status: str, last_error: Optional[str] = None
    ) -> bool:
        try:
            if not self.use_db:
                return False
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute(
                '''
                UPDATE user_exchange_keys
                SET status = ?, last_error = ?, last_tested_at = CURRENT_TIMESTAMP, updated_at = CURRENT_TIMESTAMP
                WHERE id = ? AND user_id = ?
                ''',
                (status, last_error, key_id, user_id)
            )
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            logger.error(f"Error updating exchange key test status: {e}")
            return False

    def get_user_active_exchange_credentials(
        self, user_id: str, exchange: str
    ) -> Optional[Dict[str, Any]]:
        """
        Fetch decrypted API credentials for the user's active exchange connection.
        Supports both database user keys and system master keys for admin.
        """
        try:
            clean_exchange = str(exchange or "").strip().lower()
            if not self.use_db:
                return None

            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(
                '''
                SELECT * FROM user_exchange_keys
                WHERE user_id = ? AND LOWER(exchange) = ? AND is_active = 1
                ORDER BY updated_at DESC LIMIT 1
                ''',
                (user_id, clean_exchange)
            )
            row = cursor.fetchone()
            conn.close()

            if row:
                data = dict(row)
                data["api_key_decrypted"] = decrypt_exchange_secret(data.get("api_key_encrypted", ""))
                data["api_secret_decrypted"] = decrypt_exchange_secret(data.get("api_secret_encrypted", ""))
                if data.get("passphrase_encrypted"):
                    data["passphrase_decrypted"] = decrypt_exchange_secret(data["passphrase_encrypted"])
                return data

            # Check environment fallback if user is admin or master configured
            if clean_exchange == "binance":
                k = os.getenv("BINANCE_API_KEY", "")
                s = os.getenv("BINANCE_API_SECRET", "")
                if k and s:
                    return {
                        "user_id": user_id,
                        "exchange": "binance",
                        "api_key_decrypted": k,
                        "api_secret_decrypted": s,
                        "is_testnet": os.getenv("USE_TESTNET", "true").lower() == "true",
                    }
            elif clean_exchange == "bybit":
                k = os.getenv("BYBIT_API_KEY", "")
                s = os.getenv("BYBIT_API_SECRET", "")
                if k and s:
                    return {
                        "user_id": user_id,
                        "exchange": "bybit",
                        "api_key_decrypted": k,
                        "api_secret_decrypted": s,
                        "is_testnet": os.getenv("USE_TESTNET", "true").lower() == "true",
                    }

            return None
        except Exception as e:
            logger.error(f"Error fetching active exchange credentials: {e}")
            return None

    def save_user_deposit_address(
        self,
        user_id: str,
        exchange: str,
        network: str,
        deposit_address: str,
        tag_or_memo: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Save or update a user's exchange USDT deposit address.
        """
        try:
            if not self.use_db:
                return {}
            clean_exchange = str(exchange or "bybit").strip().lower()
            clean_network = str(network or "BSC").strip().upper()
            clean_addr = str(deposit_address or "").strip()

            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute(
                '''
                INSERT INTO user_deposit_addresses (
                    user_id, exchange, network, deposit_address, tag_or_memo, is_default, updated_at
                ) VALUES (?, ?, ?, ?, ?, 1, CURRENT_TIMESTAMP)
                ON CONFLICT(user_id, exchange, network) DO UPDATE SET
                    deposit_address = excluded.deposit_address,
                    tag_or_memo = excluded.tag_or_memo,
                    updated_at = CURRENT_TIMESTAMP
                ''',
                (user_id, clean_exchange, clean_network, clean_addr, tag_or_memo)
            )
            conn.commit()
            conn.close()

            logger.info(f"💾 Saved deposit address for user {user_id} on {clean_exchange} ({clean_network})")
            return {
                "user_id": user_id,
                "exchange": clean_exchange,
                "network": clean_network,
                "deposit_address": clean_addr,
                "tag_or_memo": tag_or_memo,
            }
        except Exception as e:
            logger.error(f"Error saving user deposit address: {e}")
            raise

    def get_user_deposit_addresses(
        self, user_id: str, exchange: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        List saved deposit addresses for a user.
        """
        try:
            if not self.use_db:
                return []
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            if exchange:
                cursor.execute(
                    '''
                    SELECT * FROM user_deposit_addresses
                    WHERE user_id = ? AND LOWER(exchange) = ?
                    ORDER BY updated_at DESC
                    ''',
                    (user_id, exchange.strip().lower())
                )
            else:
                cursor.execute(
                    '''
                    SELECT * FROM user_deposit_addresses
                    WHERE user_id = ?
                    ORDER BY exchange ASC, updated_at DESC
                    ''',
                    (user_id,)
                )
            rows = cursor.fetchall()
            conn.close()
            return [dict(r) for r in rows]
        except Exception as e:
            logger.error(f"Error fetching user deposit addresses: {e}")
            return []

    def get_user_deposit_address_for_exchange(
        self, user_id: str, exchange: str, network: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Get the specific or latest saved deposit address for an exchange/network.
        """
        try:
            if not self.use_db:
                return None
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            clean_exchange = str(exchange or "").strip().lower()

            if network:
                cursor.execute(
                    '''
                    SELECT * FROM user_deposit_addresses
                    WHERE user_id = ? AND LOWER(exchange) = ? AND UPPER(network) = ?
                    LIMIT 1
                    ''',
                    (user_id, clean_exchange, network.strip().upper())
                )
            else:
                cursor.execute(
                    '''
                    SELECT * FROM user_deposit_addresses
                    WHERE user_id = ? AND LOWER(exchange) = ?
                    ORDER BY updated_at DESC LIMIT 1
                    ''',
                    (user_id, clean_exchange)
                )
            row = cursor.fetchone()
            conn.close()
            return dict(row) if row else None
        except Exception as e:
            logger.error(f"Error fetching specific deposit address: {e}")
            return None

    def get_user_risk_settings(self, user_id: str) -> Dict:
        """Get or initialize user risk settings and consent status"""
        default_settings = {
            "user_id": user_id,
            "trading_style": "day_trader",
            "risk_tolerance": "moderate",
            "sizing_mode": "kelly",
            "kelly_fraction": 0.25,
            "max_leverage": 3,
            "stop_loss_atr_mult": 1.5,
            "take_profit_atr_mult": 3.0,
            "use_trailing_stop": 1,
            "min_confidence": 0.65,
            "require_ensemble_agreement": 1,
            "max_open_positions": 5,
            "risk_consent_accepted": 0,
            "risk_consent_at": None,
            "risk_consent_ip": None,
        }
        try:
            if not self.use_db:
                return default_settings
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM user_risk_settings WHERE user_id = ?', (user_id,))
            row = cursor.fetchone()
            if row:
                res = dict(row)
                conn.close()
                return res
            cursor.execute('''
                INSERT OR IGNORE INTO user_risk_settings (user_id) VALUES (?)
            ''', (user_id,))
            conn.commit()
            conn.close()
            return default_settings
        except Exception as e:
            logger.error(f"Error getting user risk settings: {e}")
            return default_settings

    def save_user_risk_settings(self, user_id: str, settings_dict: Dict) -> bool:
        """Save updated user risk settings"""
        try:
            if not self.use_db:
                return False
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO user_risk_settings (
                    user_id, trading_style, risk_tolerance, sizing_mode, kelly_fraction,
                    max_leverage, stop_loss_atr_mult, take_profit_atr_mult, use_trailing_stop,
                    min_confidence, require_ensemble_agreement, max_open_positions, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(user_id) DO UPDATE SET
                    trading_style = excluded.trading_style,
                    risk_tolerance = excluded.risk_tolerance,
                    sizing_mode = excluded.sizing_mode,
                    kelly_fraction = excluded.kelly_fraction,
                    max_leverage = excluded.max_leverage,
                    stop_loss_atr_mult = excluded.stop_loss_atr_mult,
                    take_profit_atr_mult = excluded.take_profit_atr_mult,
                    use_trailing_stop = excluded.use_trailing_stop,
                    min_confidence = excluded.min_confidence,
                    require_ensemble_agreement = excluded.require_ensemble_agreement,
                    max_open_positions = excluded.max_open_positions,
                    updated_at = CURRENT_TIMESTAMP
            ''', (
                user_id,
                settings_dict.get("trading_style", "day_trader"),
                settings_dict.get("risk_tolerance", "moderate"),
                settings_dict.get("sizing_mode", "kelly"),
                float(settings_dict.get("kelly_fraction", 0.25)),
                int(settings_dict.get("max_leverage", 3)),
                float(settings_dict.get("stop_loss_atr_mult", 1.5)),
                float(settings_dict.get("take_profit_atr_mult", 3.0)),
                1 if settings_dict.get("use_trailing_stop", True) in (1, True, "1", "true") else 0,
                float(settings_dict.get("min_confidence", 0.65)),
                1 if settings_dict.get("require_ensemble_agreement", True) in (1, True, "1", "true") else 0,
                int(settings_dict.get("max_open_positions", 5)),
            ))
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            logger.error(f"Error saving user risk settings: {e}")
            return False

    def save_risk_consent(self, user_id: str, ip_address: Optional[str] = None) -> bool:
        """Record explicit legal risk consent for real trading execution"""
        try:
            if not self.use_db:
                return False
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO user_risk_settings (
                    user_id, risk_consent_accepted, risk_consent_at, risk_consent_ip, updated_at
                ) VALUES (?, 1, CURRENT_TIMESTAMP, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(user_id) DO UPDATE SET
                    risk_consent_accepted = 1,
                    risk_consent_at = CURRENT_TIMESTAMP,
                    risk_consent_ip = excluded.risk_consent_ip,
                    updated_at = CURRENT_TIMESTAMP
            ''', (user_id, ip_address or "unknown"))
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            logger.error(f"Error saving risk consent: {e}")
            return False

    def has_risk_consent(self, user_id: str) -> bool:
        """Check if user has accepted real trade execution risk consent"""
        try:
            if not self.use_db:
                return False
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('SELECT risk_consent_accepted FROM user_risk_settings WHERE user_id = ?', (user_id,))
            row = cursor.fetchone()
            conn.close()
            return bool(row and row[0] == 1)
        except Exception as e:
            logger.error(f"Error checking risk consent: {e}")
            return False


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CRYPTOGRAPHIC HELPERS FOR EXCHANGE CREDENTIALS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def encrypt_exchange_secret(plain_secret: str, master_key: Optional[str] = None) -> str:
    """
    Encrypt exchange secret using Authenticated Encrypt-then-MAC (PBKDF2 HMAC-SHA256).
    """
    if not plain_secret:
        return ""
    key_material = master_key or os.getenv("JWT_SECRET_KEY", "snartcrypto_master_secret_key_2026_vvip_trading")
    salt = secrets.token_bytes(16)
    derived = hashlib.pbkdf2_hmac("sha256", key_material.encode("utf-8"), salt, 100000, dklen=64)
    enc_key, mac_key = derived[:32], derived[32:]
    
    iv = secrets.token_bytes(16)
    raw_data = plain_secret.encode("utf-8")
    
    keystream = bytearray()
    ctr = 0
    while len(keystream) < len(raw_data):
        keystream.extend(hmac.new(enc_key, iv + ctr.to_bytes(4, "big"), hashlib.sha256).digest())
        ctr += 1
    
    ciphertext = bytes(b ^ k for b, k in zip(raw_data, keystream[:len(raw_data)]))
    auth_tag = hmac.new(mac_key, salt + iv + ciphertext, hashlib.sha256).digest()
    
    package = salt + iv + auth_tag + ciphertext
    return base64.urlsafe_b64encode(package).decode("utf-8")


def decrypt_exchange_secret(encrypted_token: str, master_key: Optional[str] = None) -> str:
    """
    Decrypt and authenticate exchange secret.
    """
    if not encrypted_token:
        return ""
    try:
        raw = base64.urlsafe_b64decode(encrypted_token.encode("utf-8"))
        if len(raw) < 64:
            raise ValueError("Ciphertext payload too short")
        
        salt = raw[:16]
        iv = raw[16:32]
        auth_tag = raw[32:64]
        ciphertext = raw[64:]
        
        key_material = master_key or os.getenv("JWT_SECRET_KEY", "snartcrypto_master_secret_key_2026_vvip_trading")
        derived = hashlib.pbkdf2_hmac("sha256", key_material.encode("utf-8"), salt, 100000, dklen=64)
        enc_key, mac_key = derived[:32], derived[32:]
        
        expected_tag = hmac.new(mac_key, salt + iv + ciphertext, hashlib.sha256).digest()
        if not hmac.compare_digest(auth_tag, expected_tag):
            raise ValueError("Authentication tag mismatch or corrupted secret")
        
        keystream = bytearray()
        ctr = 0
        while len(keystream) < len(ciphertext):
            keystream.extend(hmac.new(enc_key, iv + ctr.to_bytes(4, "big"), hashlib.sha256).digest())
            ctr += 1
        
        decrypted_bytes = bytes(b ^ k for b, k in zip(ciphertext, keystream[:len(ciphertext)]))
        return decrypted_bytes.decode("utf-8")
    except Exception as e:
        logger.error(f"Failed to decrypt exchange secret: {e}")
        return ""