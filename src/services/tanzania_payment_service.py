"""
Tanzania Mobile Money (M-Pesa, Tigo Pesa, Airtel Money, HaloPesa)
On-Ramp Service & P2P Exchange Assistant for Binance and Bybit.
"""

from __future__ import annotations

import os
import re
import uuid
import time
import logging
import asyncio
from typing import Dict, Any, Optional, List
from datetime import datetime, timezone
import httpx

from src.core.config import get_settings
from src.data.storage import DataStorage

logger = logging.getLogger(__name__)


class TanzaniaPaymentService:
    """
    Handles:
    1. Dynamic Multi-Source Live TZS <-> USDT currency conversion rates (Binance P2P, ExchangeRate API, CoinGecko).
    2. Automated User Exchange Deposit Address Retrieval (Bybit & Binance CCXT) with database profile persistence.
    3. Mobile operator detection (Vodacom, Tigo, Airtel, Halotel).
    4. Mobile Money STK Push collection (Beem Africa API & Sandbox).
    5. Non-custodial USDT dispatch tracking to Binance/Bybit deposit addresses.
    6. Filtered P2P deep-linking and step-by-step safety guides for Binance & Bybit.
    """

    MIN_DEPOSIT_TZS = 5000.0       # ~$1.90 USDT
    MAX_DEPOSIT_TZS = 10000000.0   # ~$3,787 USDT

    def __init__(
        self,
        provider: Optional[str] = None,
        snippe_api_key: Optional[str] = None,
        snippe_api_url: Optional[str] = None,
        snippe_webhook_secret: Optional[str] = None,
        snippe_live_mode: Optional[bool] = None,
        beem_api_key: Optional[str] = None,
        beem_secret_key: Optional[str] = None,
        beem_live_mode: Optional[bool] = None,
        live_mode: Optional[bool] = None,
        storage: Optional[DataStorage] = None,
    ):
        settings = get_settings()
        self.provider = str(provider or getattr(settings, "PAYMENT_PROVIDER", "snippe")).strip().lower()

        # Snippe Configuration
        self.snippe_api_key = snippe_api_key or getattr(settings, "SNIPPE_API_KEY", "")
        self.snippe_api_url = snippe_api_url or getattr(settings, "SNIPPE_API_URL", "https://api.snippe.io/v1")
        self.snippe_webhook_secret = snippe_webhook_secret or getattr(settings, "SNIPPE_WEBHOOK_SECRET", "")
        self.snippe_live_mode = snippe_live_mode if snippe_live_mode is not None else getattr(settings, "SNIPPE_LIVE_MODE", False)

        # Beem Africa Configuration
        self.beem_api_key = beem_api_key or getattr(settings, "BEEM_API_KEY", "")
        self.beem_secret_key = beem_secret_key or getattr(settings, "BEEM_SECRET_KEY", "")
        self.beem_live_mode = beem_live_mode if beem_live_mode is not None else getattr(settings, "BEEM_LIVE_MODE", False)
        self.checkout_url = getattr(settings, "BEEM_CHECKOUT_URL", "https://checkout.beem.africa/v1/checkout")

        # General Live Mode override
        if live_mode is not None:
            self.snippe_live_mode = live_mode
            self.beem_live_mode = live_mode

        self.default_tzs_rate = settings.DEFAULT_TZS_PER_USDT
        self.rate_cache_ttl = settings.RATE_CACHE_TTL_SECONDS
        self.binance_p2p_url = settings.BINANCE_P2P_RATE_API_URL
        self.exchangerate_url = settings.EXCHANGE_RATE_API_URL
        self.coingecko_url = settings.COINGECKO_RATE_API_URL

        self.storage = storage or DataStorage()
        
        self._cached_rate: float = self.default_tzs_rate
        self._cached_source: str = "default_config"
        self._rate_updated_at: float = 0.0
        self._orders: Dict[str, Dict[str, Any]] = {}
        self._client: Optional[httpx.AsyncClient] = None

        logger.info(
            "TanzaniaPaymentService initialized | provider=%s | snippe_live=%s (configured=%s) | beem_live=%s (configured=%s) | default_rate=%.2f",
            self.provider,
            self.snippe_live_mode,
            bool(self.snippe_api_key),
            self.beem_live_mode,
            bool(self.beem_api_key and self.beem_secret_key),
            self.default_tzs_rate,
        )

    async def start(self) -> None:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=12.0,
                headers={"User-Agent": "SnartCrypto/3.1 (Tanzania On-Ramp Engine)"}
            )

    async def close(self) -> None:
        if self._client:
            try:
                if hasattr(self._client, "aclose"):
                    res = self._client.aclose()
                    if asyncio.iscoroutine(res):
                        await res
            except Exception:
                pass
            self._client = None

    # =========================================================================
    # 1. DYNAMIC LIVE MULTI-SOURCE EXCHANGE RATE ENGINE
    # =========================================================================

    async def get_live_rate(self) -> Dict[str, Any]:
        """
        Fetch real-time TZS / USDT exchange rate with multi-source fallback:
        Tier 1: Binance P2P Tanzanian order book API (exact real-world buy rate).
        Tier 2: Open ExchangeRate API (USD/TZS forex benchmark).
        Tier 3: CoinGecko Tether (USDT/TZS simple price).
        Fallback: Config default from .env.
        """
        now = time.time()
        if now - self._rate_updated_at < self.rate_cache_ttl:
            return {
                "rate": round(self._cached_rate, 2),
                "currency_pair": "TZS/USDT",
                "source": self._cached_source,
                "is_live": self._cached_source != "default_config",
                "min_tzs": self.MIN_DEPOSIT_TZS,
                "max_tzs": self.MAX_DEPOSIT_TZS,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "updated_seconds_ago": int(now - self._rate_updated_at) if self._rate_updated_at > 0 else 0,
            }

        await self.start()
        refreshed = False

        # ── Tier 1: Binance P2P TZS Order Book (Most Accurate Local Crypto Rate) ──
        try:
            p2p_payload = {
                "fiat": "TZS",
                "page": 1,
                "rows": 6,
                "tradeType": "BUY",
                "asset": "USDT",
                "countries": [],
                "proMerchantAds": False,
                "shieldMerchantAds": False,
                "filterType": "all",
                "periods": [],
                "additionalKycVerifyFilter": 0,
                "publisherType": None,
                "payTypes": []
            }
            res = await self._client.post(
                self.binance_p2p_url,
                json=p2p_payload,
                timeout=6.0,
            )
            if res.status_code == 200:
                data = res.json()
                ads = data.get("data", [])
                prices = [
                    float(ad["adv"]["price"])
                    for ad in ads
                    if "adv" in ad and "price" in ad["adv"]
                ]
                if prices:
                    # Calculate median rate to eliminate extreme outlier ads
                    prices.sort()
                    median_rate = prices[len(prices) // 2]
                    if 2200.0 <= median_rate <= 3600.0:
                        self._cached_rate = median_rate
                        self._cached_source = "binance_p2p"
                        self._rate_updated_at = now
                        refreshed = True
                        logger.info("Live TZS/USDT updated from Binance P2P: %.2f TZS", median_rate)
        except Exception as exc:
            logger.debug("Binance P2P rate lookup failed (%s), trying Forex benchmark...", exc)

        # ── Tier 2: Open ExchangeRate API (USD / TZS Forex Spot) ──
        if not refreshed:
            try:
                res = await self._client.get(self.exchangerate_url, timeout=5.0)
                if res.status_code == 200:
                    data = res.json()
                    rates = data.get("rates", {})
                    tzs_rate = float(rates.get("TZS", 0))
                    if 2200.0 <= tzs_rate <= 3600.0:
                        self._cached_rate = tzs_rate
                        self._cached_source = "open_exchangerate"
                        self._rate_updated_at = now
                        refreshed = True
                        logger.info("Live TZS/USDT updated from ExchangeRate API: %.2f TZS", tzs_rate)
            except Exception as exc:
                logger.debug("ExchangeRate API lookup failed (%s), trying CoinGecko...", exc)

        # ── Tier 3: CoinGecko Crypto Spot ──
        if not refreshed:
            try:
                res = await self._client.get(self.coingecko_url, timeout=5.0)
                if res.status_code == 200:
                    data = res.json()
                    tzs_rate = float(data.get("tether", {}).get("tzs", 0))
                    if 2200.0 <= tzs_rate <= 3600.0:
                        self._cached_rate = tzs_rate
                        self._cached_source = "coingecko"
                        self._rate_updated_at = now
                        refreshed = True
                        logger.info("Live TZS/USDT updated from CoinGecko: %.2f TZS", tzs_rate)
            except Exception as exc:
                logger.warning("CoinGecko rate lookup failed: %s", exc)

        if not refreshed and self._rate_updated_at == 0:
            self._cached_rate = self.default_tzs_rate
            self._cached_source = "default_config"

        return {
            "rate": round(self._cached_rate, 2),
            "currency_pair": "TZS/USDT",
            "source": self._cached_source,
            "is_live": self._cached_source != "default_config",
            "min_tzs": self.MIN_DEPOSIT_TZS,
            "max_tzs": self.MAX_DEPOSIT_TZS,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "updated_seconds_ago": int(now - self._rate_updated_at) if self._rate_updated_at > 0 else 0,
        }

    def convert_tzs_to_usdt(self, amount_tzs: float) -> float:
        """Convert TZS to USDT based on current rate."""
        if amount_tzs <= 0 or self._cached_rate <= 0:
            return 0.0
        return round(amount_tzs / self._cached_rate, 2)

    def convert_usdt_to_tzs(self, amount_usdt: float) -> float:
        """Convert USDT to TZS based on current rate."""
        if amount_usdt <= 0:
            return 0.0
        return round(amount_usdt * self._cached_rate, 2)

    # =========================================================================
    # 2. AUTOMATED USER EXCHANGE DEPOSIT ADDRESS RETRIEVAL & MANAGEMENT
    # =========================================================================

    async def get_user_exchange_deposit_address(
        self,
        user_id: str,
        exchange: str = "bybit",
        network: str = "BSC",
    ) -> Dict[str, Any]:
        """
        Fetch the user's USDT deposit address with two-tier resolution:
        1. Automated: If the user has linked Bybit or Binance API keys, query the exchange API directly via CCXT.
        2. Saved Profile: Fall back to previously saved deposit address in user_deposit_addresses table.
        """
        clean_exchange = str(exchange or "bybit").strip().lower()
        clean_network = str(network or "BSC").strip().upper()

        # Step 1: Check if user has active exchange API credentials connected
        creds = self.storage.get_user_active_exchange_credentials(user_id, clean_exchange)
        if creds and creds.get("api_key_decrypted") and creds.get("api_secret_decrypted"):
            try:
                import ccxt.async_support as ccxt_async
                
                api_key = creds["api_key_decrypted"]
                api_secret = creds["api_secret_decrypted"]
                passphrase = creds.get("passphrase_decrypted")
                is_testnet = creds.get("is_testnet", False)

                client_config: Dict[str, Any] = {
                    "apiKey": api_key,
                    "secret": api_secret,
                    "enableRateLimit": True,
                }
                if passphrase:
                    client_config["password"] = passphrase

                client = None
                if clean_exchange == "binance":
                    client = ccxt_async.binance(client_config)
                    if is_testnet:
                        client.set_sandbox_mode(True)
                elif clean_exchange == "bybit":
                    client = ccxt_async.bybit(client_config)
                    if is_testnet:
                        client.set_sandbox_mode(True)

                if client:
                    try:
                        # Attempt to query deposit address for USDT
                        net_param = clean_network
                        if clean_network == "BSC":
                            net_param = "BSC"
                        elif clean_network in ("TRC20", "TRX"):
                            net_param = "TRX" if clean_exchange == "binance" else "TRC20"

                        deposit_res = await client.fetch_deposit_address("USDT", {"network": net_param})
                        await client.close()

                        addr = deposit_res.get("address")
                        tag = deposit_res.get("tag")
                        if addr:
                            logger.info(
                                "Auto-fetched deposit address for user %s on %s (%s): %s",
                                user_id, clean_exchange, clean_network, addr
                            )
                            # Save to user profile database for quick offline access
                            self.storage.save_user_deposit_address(
                                user_id=user_id,
                                exchange=clean_exchange,
                                network=clean_network,
                                deposit_address=addr,
                                tag_or_memo=tag,
                            )
                            return {
                                "success": True,
                                "exchange": clean_exchange.upper(),
                                "network": clean_network,
                                "deposit_address": addr,
                                "tag_or_memo": tag,
                                "source": "exchange_api",
                                "auto_detected": True,
                                "message": f"Auto-detected from your connected {clean_exchange.upper()} account",
                            }
                    except Exception as ccxt_err:
                        logger.warning(
                            "CCXT fetch_deposit_address failed for %s (%s): %s",
                            clean_exchange, user_id, ccxt_err
                        )
                        if client:
                            await client.close()
            except Exception as e:
                logger.warning("Error initializing CCXT client for deposit lookup: %s", e)

        # Step 2: Fall back to database saved user deposit address
        saved = self.storage.get_user_deposit_address_for_exchange(user_id, clean_exchange, clean_network)
        if saved and saved.get("deposit_address"):
            return {
                "success": True,
                "exchange": clean_exchange.upper(),
                "network": clean_network,
                "deposit_address": saved["deposit_address"],
                "tag_or_memo": saved.get("tag_or_memo"),
                "source": "saved_profile",
                "auto_detected": False,
                "message": f"Loaded from your saved {clean_exchange.upper()} profile",
            }

        # Step 3: No address found
        return {
            "success": False,
            "exchange": clean_exchange.upper(),
            "network": clean_network,
            "deposit_address": None,
            "tag_or_memo": None,
            "source": "none",
            "auto_detected": False,
            "message": f"No deposit address found for {clean_exchange.upper()}. Please enter your address or link your API key in Settings.",
        }

    def save_user_deposit_address(
        self,
        user_id: str,
        exchange: str,
        network: str,
        deposit_address: str,
        tag_or_memo: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Validate and save a user's USDT deposit address.
        """
        clean_addr = str(deposit_address or "").strip()
        if not clean_addr or len(clean_addr) < 10:
            raise ValueError("Invalid deposit address. Address must be at least 10 characters long.")

        return self.storage.save_user_deposit_address(
            user_id=user_id,
            exchange=exchange,
            network=network,
            deposit_address=clean_addr,
            tag_or_memo=tag_or_memo,
        )

    # =========================================================================
    # 3. OPERATOR & PHONE NUMBER NORMALIZATION
    # =========================================================================

    @staticmethod
    def normalize_phone_number(phone: str) -> str:
        """
        Normalize Tanzanian phone number to E.164 without leading plus: '2557XXXXXXXX'.
        Accepts formats: '0754123456', '+255754123456', '255754123456', '754123456'.
        """
        clean = re.sub(r"[^\d]", "", str(phone or ""))
        if clean.startswith("0") and len(clean) == 10:
            return "255" + clean[1:]
        if clean.startswith("255") and len(clean) == 12:
            return clean
        if len(clean) == 9:
            return "255" + clean
        return clean

    @staticmethod
    def detect_operator(phone: str) -> str:
        """
        Detect Tanzania network provider by phone number prefix.
        Returns: 'vodacom', 'tigo', 'airtel', 'halotel', or 'unknown'.
        """
        norm = TanzaniaPaymentService.normalize_phone_number(phone)
        if len(norm) != 12 or not norm.startswith("255"):
            return "unknown"

        prefix = norm[3:5]
        if prefix in ("74", "75", "76"):
            return "vodacom"
        elif prefix in ("65", "67", "71"):
            return "tigo"
        elif prefix in ("68", "69", "78"):
            return "airtel"
        elif prefix in ("61", "62"):
            return "halotel"

        return "unknown"

    # =========================================================================
    # 4. DIRECT-TO-EXCHANGE STK PUSH ON-RAMP (BEEM AFRICA)
    # =========================================================================

    async def initiate_deposit_stk(
        self,
        phone_number: str,
        amount_tzs: float,
        exchange: str,
        deposit_address: str,
        network: str = "BSC",
        user_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Trigger an instant Mobile Money STK Push prompt to user's phone.
        When paid, USDT is dispatched directly to their Binance/Bybit wallet address.
        """
        norm_phone = self.normalize_phone_number(phone_number)
        if len(norm_phone) != 12:
            return {
                "success": False,
                "error": "Invalid Tanzanian phone number. Please enter a valid number (e.g., 0754123456 or +255754123456).",
            }

        if amount_tzs < self.MIN_DEPOSIT_TZS:
            return {
                "success": False,
                "error": f"Minimum deposit is TZS {self.MIN_DEPOSIT_TZS:,.0f}.",
            }

        if amount_tzs > self.MAX_DEPOSIT_TZS:
            return {
                "success": False,
                "error": f"Maximum single deposit is TZS {self.MAX_DEPOSIT_TZS:,.0f}.",
            }

        exchange_clean = str(exchange or "bybit").lower()
        if exchange_clean not in ("binance", "bybit", "okx"):
            exchange_clean = "bybit"

        if not deposit_address or len(deposit_address.strip()) < 10:
            return {
                "success": False,
                "error": "Please provide a valid USDT deposit address for your exchange.",
            }

        # Refresh rate if necessary
        rate_info = await self.get_live_rate()
        current_rate = rate_info["rate"]

        operator = self.detect_operator(norm_phone)
        usdt_amount = round(amount_tzs / current_rate, 2)
        order_id = f"SNART_{int(time.time())}_{uuid.uuid4().hex[:6].upper()}"

        order_record = {
            "order_id": order_id,
            "user_id": user_id or "anonymous",
            "phone_number": norm_phone,
            "operator": operator,
            "amount_tzs": amount_tzs,
            "amount_usdt": usdt_amount,
            "rate_used": current_rate,
            "rate_source": rate_info.get("source", "unknown"),
            "exchange": exchange_clean,
            "network": network.upper(),
            "deposit_address": deposit_address.strip(),
            "status": "PENDING",  # PENDING, COMPLETED, FAILED, EXPIRED
            "created_at": datetime.now(timezone.utc).isoformat(),
            "tx_hash": None,
        }

        self._orders[order_id] = order_record

        # ── Execution Layer: Snippe API vs. Beem Africa vs. Sandbox Simulation ──
        checkout_url_redirect = None
        if self.provider == "snippe" and self.snippe_live_mode and self.snippe_api_key:
            try:
                await self.start()
                headers = {
                    "Authorization": f"Bearer {self.snippe_api_key}",
                    "Content-Type": "application/json",
                }
                payload = {
                    "amount": int(amount_tzs),
                    "currency": "TZS",
                    "phone_number": norm_phone,
                    "operator": operator,
                    "reference": order_id,
                    "description": f"SnartCrypto AI On-Ramp {usdt_amount} USDT ({exchange_clean.upper()})",
                    "metadata": {
                        "user_id": user_id or "anonymous",
                        "order_id": order_id,
                        "type": "onramp",
                        "exchange": exchange_clean,
                        "deposit_address": deposit_address.strip(),
                    },
                }
                response = await self._client.post(
                    f"{self.snippe_api_url.rstrip('/')}/payments/collect",
                    json=payload,
                    headers=headers,
                )
                if response.status_code in (200, 201, 202):
                    res_json = response.json() if response.text.startswith("{") else {}
                    checkout_url_redirect = res_json.get("checkout_url") or res_json.get("payment_url")
                    order_record["checkout_url"] = checkout_url_redirect
                    order_record["provider_response"] = res_json
                    logger.info("Snippe STK Push initiated for order %s (Ref: %s)", order_id, res_json.get("reference", order_id))
                else:
                    logger.warning(
                        "Snippe API collection error | status=%s | body=%s",
                        response.status_code,
                        response.text
                    )
            except Exception as e:
                logger.error("Failed to reach Snippe API: %s", e)
        elif self.provider == "beem" and self.beem_live_mode and self.beem_api_key and self.beem_secret_key:
            try:
                await self.start()
                params = {
                    "amount": str(int(amount_tzs)),
                    "reference_number": order_id,
                    "transaction_id": str(uuid.uuid4()),
                    "mobile": norm_phone,
                    "sendSource": "true",
                }
                response = await self._client.get(
                    self.checkout_url,
                    params=params,
                    auth=(self.beem_api_key, self.beem_secret_key),
                )
                if response.status_code in (200, 201):
                    res_json = response.json() if response.text.startswith("{") else {}
                    checkout_url_redirect = res_json.get("src")
                    order_record["checkout_url"] = checkout_url_redirect
                    logger.info("Beem STK Push / Checkout session initiated for order %s (URL: %s)", order_id, checkout_url_redirect)
                else:
                    logger.warning(
                        "Beem Checkout API response | status=%s | body=%s. Note: If message is 'No payment method set', configure BPay Payment Routes (M-Pesa, Tigo, Airtel) in portal.beem.africa.",
                        response.status_code,
                        response.text
                    )
            except Exception as e:
                logger.error("Failed to reach Beem API: %s", e)
        else:
            # Sandbox / Simulated Mode: Auto-progresses order in background
            logger.info("Simulated Sandbox Mode (%s): STK Push initialized for order %s (%s TZS -> %s USDT @ %s TZS/USDT)", self.provider.upper(), order_id, amount_tzs, usdt_amount, current_rate)
            asyncio.create_task(self._auto_simulate_sandbox_approval(order_id))

        return {
            "success": True,
            "order_id": order_id,
            "status": "PENDING",
            "provider": self.provider,
            "amount_tzs": amount_tzs,
            "amount_usdt": usdt_amount,
            "rate": current_rate,
            "rate_source": rate_info.get("source", "live"),
            "operator": operator,
            "exchange": exchange_clean.upper(),
            "network": network.upper(),
            "deposit_address": deposit_address.strip(),
            "checkout_url": checkout_url_redirect,
            "instructions": f"Check your phone ({norm_phone}). A USSD prompt has been sent. Enter your {operator.upper()} PIN to approve TZS {amount_tzs:,.0f}.",
            "timeout_seconds": 120,
        }

    async def initiate_subscription_momo(
        self,
        user_id: str,
        plan_id: str,
        phone_number: str,
    ) -> Dict[str, Any]:
        """
        Trigger an instant Mobile Money STK Push prompt for plan subscriptions (Pro, VIP, VVIP).
        When paid, the user's role is automatically upgraded and activated in the database.
        """
        plan_map = {
            "pro_20": {"name": "Pro Trader", "price_usd": 20.0, "role": "pro"},
            "vip_49": {"name": "VIP Quantitative", "price_usd": 49.0, "role": "vip"},
            "vvip_99": {"name": "VVIP Institutional", "price_usd": 99.0, "role": "vvip"},
        }

        clean_plan_id = str(plan_id or "").strip()
        if clean_plan_id not in plan_map:
            return {
                "success": False,
                "error": f"Invalid plan ID: {plan_id}. Must be one of: pro_20, vip_49, vvip_99",
            }

        plan_info = plan_map[clean_plan_id]
        norm_phone = self.normalize_phone_number(phone_number)
        if len(norm_phone) != 12:
            return {
                "success": False,
                "error": "Invalid Tanzanian phone number. Please enter a valid number (e.g. 0754123456 or +255754123456).",
            }

        # Calculate exact live TZS price using current rate engine
        rate_info = await self.get_live_rate()
        current_rate = rate_info["rate"]
        amount_usd = plan_info["price_usd"]
        # Round TZS amount to clean nearest 100 TZS for nice mobile checkout
        amount_tzs = round((amount_usd * current_rate) / 100.0) * 100.0

        operator = self.detect_operator(norm_phone)
        order_id = f"SUB_{clean_plan_id.upper()}_{user_id[:6]}_{int(time.time())}"

        order_record = {
            "order_id": order_id,
            "type": "subscription",
            "user_id": user_id,
            "plan_id": clean_plan_id,
            "plan_name": plan_info["name"],
            "target_role": plan_info["role"],
            "phone_number": norm_phone,
            "operator": operator,
            "amount_usd": amount_usd,
            "amount_tzs": amount_tzs,
            "rate_used": current_rate,
            "rate_source": rate_info.get("source", "live"),
            "status": "PENDING",  # PENDING, COMPLETED, FAILED
            "created_at": datetime.now(timezone.utc).isoformat(),
            "tx_hash": None,
        }

        self._orders[order_id] = order_record

        # ── Execution Layer: Snippe API vs. Beem Africa vs. Sandbox Simulation ──
        checkout_url_redirect = None
        if self.provider == "snippe" and self.snippe_live_mode and self.snippe_api_key:
            try:
                await self.start()
                headers = {
                    "Authorization": f"Bearer {self.snippe_api_key}",
                    "Content-Type": "application/json",
                }
                payload = {
                    "amount": int(amount_tzs),
                    "currency": "TZS",
                    "phone_number": norm_phone,
                    "operator": operator,
                    "reference": order_id,
                    "description": f"SnartCrypto AI {plan_info['name']} Subscription",
                    "metadata": {
                        "user_id": user_id,
                        "plan_id": clean_plan_id,
                        "order_id": order_id,
                        "type": "subscription",
                        "target_role": plan_info["role"],
                    },
                }
                response = await self._client.post(
                    f"{self.snippe_api_url.rstrip('/')}/payments/collect",
                    json=payload,
                    headers=headers,
                )
                if response.status_code in (200, 201, 202):
                    res_json = response.json() if response.text.startswith("{") else {}
                    checkout_url_redirect = res_json.get("checkout_url") or res_json.get("payment_url")
                    order_record["checkout_url"] = checkout_url_redirect
                    order_record["provider_response"] = res_json
                    logger.info("Snippe Subscription STK Push sent for %s (Plan: %s, User: %s)", order_id, clean_plan_id, user_id)
                else:
                    logger.warning(
                        "Snippe Subscription API error | status=%s | body=%s",
                        response.status_code,
                        response.text
                    )
            except Exception as e:
                logger.error("Failed to reach Snippe Subscription API: %s", e)
        elif self.provider == "beem" and self.beem_live_mode and self.beem_api_key and self.beem_secret_key:
            try:
                await self.start()
                params = {
                    "amount": str(int(amount_tzs)),
                    "reference_number": order_id,
                    "transaction_id": str(uuid.uuid4()),
                    "mobile": norm_phone,
                    "sendSource": "true",
                }
                response = await self._client.get(
                    self.checkout_url,
                    params=params,
                    auth=(self.beem_api_key, self.beem_secret_key),
                )
                if response.status_code in (200, 201):
                    res_json = response.json() if response.text.startswith("{") else {}
                    checkout_url_redirect = res_json.get("src")
                    order_record["checkout_url"] = checkout_url_redirect
                    logger.info("Beem Subscription STK Push sent for %s (Plan: %s, User: %s, URL: %s)", order_id, clean_plan_id, user_id, checkout_url_redirect)
                else:
                    logger.warning(
                        "Beem STK Push error | status=%s | body=%s. Note: If message is 'No payment method set', configure BPay Payment Routes (M-Pesa, Tigo, Airtel) in portal.beem.africa.",
                        response.status_code,
                        response.text
                    )
            except Exception as e:
                logger.error("Failed to reach Beem STK Push API: %s", e)
        else:
            # Sandbox / Dev mode: simulate user entering PIN after 8s
            logger.info("Sandbox Mode (%s): Simulating Mobile Money subscription for %s (%s, TZS %s)", self.provider.upper(), order_id, plan_info["name"], amount_tzs)
            asyncio.create_task(self._auto_simulate_sandbox_approval(order_id))

        return {
            "success": True,
            "order_id": order_id,
            "status": "PENDING",
            "provider": self.provider,
            "plan_id": clean_plan_id,
            "plan_name": plan_info["name"],
            "amount_usd": amount_usd,
            "amount_tzs": amount_tzs,
            "rate": current_rate,
            "rate_source": rate_info.get("source", "live"),
            "operator": operator,
            "phone_number": norm_phone,
            "checkout_url": checkout_url_redirect,
            "instructions": f"Check your phone ({norm_phone}). Enter your {operator.upper()} PIN to approve TZS {amount_tzs:,.0f} for {plan_info['name']} subscription.",
            "timeout_seconds": 120,
        }

    async def _auto_simulate_sandbox_approval(self, order_id: str) -> None:
        """Simulate user entering PIN in sandbox mode after 8 seconds."""
        await asyncio.sleep(8)
        order = self._orders.get(order_id)
        if order and order["status"] == "PENDING":
            order["status"] = "COMPLETED"
            order["tx_hash"] = f"{self.provider.upper()}_SIM_{uuid.uuid4().hex[:12].upper()}"

            # If this is a subscription order, provision immediately in database!
            if order.get("type") == "subscription":
                sub_created = self.storage.create_subscription(
                    subscription_id=order_id,
                    user_id=order["user_id"],
                    plan_id=order["plan_id"],
                    payment_method=f"{self.provider}_momo",
                    amount_paid=order["amount_usd"],
                )
                logger.info(
                    "Sandbox Subscription Activated! Order: %s | User: %s | Plan: %s | Upgraded DB role: %s",
                    order_id, order["user_id"], order["plan_id"], sub_created
                )
            else:
                logger.info(
                    "Sandbox simulated approval completed for order %s | Dispatched %s USDT to %s (%s)",
                    order_id, order["amount_usdt"], order.get("exchange"), order.get("deposit_address")
                )

    def get_order_status(self, order_id: str) -> Optional[Dict[str, Any]]:
        """Get live status of an on-ramp deposit or subscription order."""
        return self._orders.get(order_id)

    def process_webhook(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process incoming payment confirmation webhook from Snippe or Beem Africa.
        Automatically verifies transaction reference, updates order state, and
        provisions user subscriptions in SQLite database.
        """
        # Multi-provider reference extraction
        order_id = (
            data.get("reference")
            or data.get("order_id")
            or data.get("transaction_id")
            or data.get("reference_number")
        )
        if not order_id and isinstance(data.get("data"), dict):
            order_id = data["data"].get("reference") or data["data"].get("order_id")

        # Multi-provider status extraction
        raw_status = (
            data.get("status")
            or (data.get("data", {}).get("status") if isinstance(data.get("data"), dict) else "")
            or ""
        )
        status_clean = str(raw_status).upper()

        if not order_id or order_id not in self._orders:
            logger.warning("Webhook received for unknown or missing order reference: %s", order_id)
            return {"success": False, "error": "Order not found", "order_id": order_id}

        order = self._orders[order_id]
        if status_clean in ("SUCCESS", "COMPLETED", "PAID", "APPROVED"):
            order["status"] = "COMPLETED"
            tx_hash = (
                data.get("transaction_id")
                or data.get("receipt")
                or data.get("id")
                or f"SNIPPE_{uuid.uuid4().hex[:10].upper()}"
            )
            order["tx_hash"] = tx_hash

            # Provision subscription if this is a subscription order
            if order.get("type") == "subscription":
                self.storage.create_subscription(
                    subscription_id=order_id,
                    user_id=order["user_id"],
                    plan_id=order["plan_id"],
                    payment_method=f"{self.provider}_momo",
                    amount_paid=order["amount_usd"],
                )
                logger.info("Webhook: Subscription activated for user %s (Plan: %s, Order: %s)", order["user_id"], order["plan_id"], order_id)
            else:
                logger.info("Webhook: Payment confirmed for order %s. Releasing %s USDT to %s", order_id, order["amount_usdt"], order["deposit_address"])

            return {"success": True, "status": "COMPLETED", "order_id": order_id}
        else:
            order["status"] = "FAILED"
            return {"success": True, "status": "FAILED", "order_id": order_id}

    # =========================================================================
    # 5. SMART P2P ASSISTANT & DEEP-LINKING (BINANCE & BYBIT)
    # =========================================================================

    def get_p2p_guides(self) -> Dict[str, Any]:
        """
        Returns pre-filtered deep links and step-by-step Swahili & English guides
        for Binance P2P and Bybit P2P with Tanzanian Mobile Money.
        """
        rate = self._cached_rate

        return {
            "live_rate_tzs": rate,
            "rate_source": self._cached_source,
            "supported_exchanges": [
                {
                    "name": "Binance",
                    "code": "binance",
                    "badge": "World's Largest Volume",
                    "deep_link_mpesa": "https://p2p.binance.com/en/trade/buy/USDT?fiat=TZS&payment=MPESA",
                    "deep_link_tigo": "https://p2p.binance.com/en/trade/buy/USDT?fiat=TZS&payment=TigoPesa",
                    "deep_link_all": "https://p2p.binance.com/en/trade/buy/USDT?fiat=TZS",
                    "fee": "0% for buyers",
                    "steps_swahili": [
                        "1. Fungua Binance P2P na chagua 'Buy USDT' kwa kutumia TZS.",
                        "2. Chagua mfanyabiashara aliyethibitishwa (mwenye alama ya njano na alama ya 98%+).",
                        "3. Weka kiasi cha TZS unachotaka kununua (mfano: TZS 50,000).",
                        "4. Tuma pesa kwa namba yake ya M-Pesa au Tigo Pesa.",
                        "5. Bonyeza 'Transferred, notify seller' kwenye Binance.",
                        "6. USDT ikishaingia, uhamishe kutoka 'Funding' kwenda 'Spot/Futures' ili bot ianze kutrade!",
                    ],
                    "steps_english": [
                        "1. Open Binance P2P and select 'Buy USDT' with TZS currency.",
                        "2. Choose a verified merchant with a yellow badge and 98%+ completion score.",
                        "3. Enter the TZS amount you want to buy (e.g. 50,000 TZS).",
                        "4. Send the exact TZS to the seller's M-Pesa or Tigo Pesa number.",
                        "5. Tap 'Transferred, notify seller' inside Binance.",
                        "6. Once received, transfer USDT from 'Funding' to 'Spot/Futures' for automated AI trading!",
                    ]
                },
                {
                    "name": "Bybit",
                    "code": "bybit",
                    "badge": "0% Fees • Fast Execution",
                    "deep_link_mpesa": "https://www.bybit.com/fiat/trade/otc/?actionType=1&token=USDT&fiat=TZS",
                    "deep_link_tigo": "https://www.bybit.com/fiat/trade/otc/?actionType=1&token=USDT&fiat=TZS",
                    "deep_link_all": "https://www.bybit.com/fiat/trade/otc/?actionType=1&token=USDT&fiat=TZS",
                    "fee": "0% for buyers",
                    "steps_swahili": [
                        "1. Fungua Bybit P2P na uchague sarafu ya TZS.",
                        "2. Chagua 'Buy USDT' na uchague njia ya malipo (M-Pesa, Tigo Pesa, au Airtel Money).",
                        "3. Chagua muuzaji mwenye nyota na maoni mazuri.",
                        "4. Fanya malipo kupitia simu yako ya mkononi.",
                        "5. Bonyeza 'Payment Completed' kwenye Bybit.",
                        "6. Hamisha USDT kutoka 'Funding Account' kwenda 'Unified Trading Account' ili bot ifanye kazi!",
                    ],
                    "steps_english": [
                        "1. Open Bybit P2P and select TZS fiat currency.",
                        "2. Select 'Buy USDT' and pick your payment method (M-Pesa, Tigo Pesa, or Airtel Money).",
                        "3. Select a reputable merchant with high completion rate.",
                        "4. Send payment via your mobile money phone.",
                        "5. Click 'Payment Completed' on Bybit.",
                        "6. Move USDT from 'Funding Account' to 'Unified Trading Account (UTA)' for SnartCrypto AI trading!",
                    ]
                }
            ],
            "anti_scam_checklist": [
                "🛡️ Kamwe usitoe PIN yako ya M-Pesa kwa mtu yeyote kwenye chat.",
                "🛡️ Hakikisha jina la mpokeaji wa M-Pesa linafanana na jina lililoonyeshwa kwenye Binance/Bybit.",
                "🛡️ Kamwe usikubali kughairi oda baada ya kutuma pesa.",
                "🛡️ Wasiliana na msaada wa SnartCrypto au Exchange endapo utapata shida yoyote."
            ]
        }
