"""
Email Service for SmartCrypto AI
Handles sending transactional emails (Verification OTP, Password Reset, Security Alerts)
Supports Free SMTP (Gmail, Brevo, SendGrid, Mailjet) and Local Console Simulation.
"""

import asyncio
import smtplib
import ssl
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Optional, Dict, Any
import logging

from src.core.config import get_settings
from src.utils.safe_logger import SafeLogger

logger = SafeLogger.get_logger("EmailService")


class EmailService:
    """Production-grade asynchronous email delivery service."""

    def __init__(self, settings=None):
        self.settings = settings or get_settings()

    @property
    def is_configured(self) -> bool:
        """Returns True if valid SMTP credentials are configured."""
        if getattr(self.settings, "SMTP_SIMULATION_MODE", False):
            return False
        host = getattr(self.settings, "SMTP_HOST", "")
        user = getattr(self.settings, "SMTP_USER", "")
        password = getattr(self.settings, "SMTP_PASSWORD", "")
        return bool(host and user and password)

    async def send_password_reset_otp(self, to_email: str, otp_code: str, expires_in_minutes: int = 15) -> bool:
        """Send a 6-digit Password Reset OTP email."""
        subject = f"🔐 Your SnartCrypto Password Reset Code: {otp_code}"
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
          <meta charset="utf-8">
          <style>
            body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; background-color: #0b0f19; color: #f8fafc; margin: 0; padding: 20px; }}
            .card {{ max-width: 500px; margin: 0 auto; background: #131b2e; border: 1px solid #1e293b; border-radius: 16px; padding: 32px; box-shadow: 0 10px 30px rgba(0,0,0,0.5); }}
            .brand {{ text-align: center; margin-bottom: 24px; }}
            .brand-name {{ font-size: 24px; font-weight: 800; color: #00e676; letter-spacing: 0.5px; }}
            .title {{ font-size: 20px; font-weight: 700; color: #ffffff; text-align: center; margin-bottom: 8px; }}
            .subtitle {{ font-size: 14px; color: #94a3b8; text-align: center; margin-bottom: 24px; }}
            .otp-box {{ background: #0b0f19; border: 2px dashed #00e676; border-radius: 12px; padding: 20px; text-align: center; margin: 24px 0; }}
            .otp-code {{ font-size: 36px; font-weight: 800; letter-spacing: 8px; color: #00e676; font-family: monospace; }}
            .warning {{ font-size: 12px; color: #f59e0b; background: rgba(245, 158, 11, 0.1); border-radius: 8px; padding: 10px; margin-top: 20px; text-align: center; }}
            .footer {{ font-size: 12px; color: #64748b; text-align: center; margin-top: 28px; }}
          </style>
        </head>
        <body>
          <div class="card">
            <div class="brand">
              <span class="brand-name">⚡ SnartCrypto</span>
            </div>
            <div class="title">Password Reset Request</div>
            <div class="subtitle">Use the verification code below to reset your password. This code will expire in <b>{expires_in_minutes} minutes</b>.</div>
            
            <div class="otp-box">
              <div class="otp-code">{otp_code}</div>
            </div>
            
            <div class="warning">
              ⚠️ If you did not request this password reset, please ignore this email. Your account remains secure.
            </div>
            
            <div class="footer">
              &copy; SnartCrypto. All rights reserved.
            </div>
          </div>
        </body>
        </html>
        """
        
        text_content = (
            f"SnartCrypto - Password Reset Request\n\n"
            f"Your 6-digit password reset code is: {otp_code}\n\n"
            f"This code will expire in {expires_in_minutes} minutes.\n\n"
            f"If you did not request this, please ignore this email."
        )
        
        return await self._dispatch_email(to_email, subject, html_content, text_content, otp_code=otp_code, purpose="PASSWORD_RESET")

    async def send_verification_otp(self, to_email: str, otp_code: str, expires_in_minutes: int = 15) -> bool:
        """Send a 6-digit Email Verification OTP email."""
        subject = f"✨ Verify Your Email for SnartCrypto: {otp_code}"
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
          <meta charset="utf-8">
          <style>
            body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; background-color: #0b0f19; color: #f8fafc; margin: 0; padding: 20px; }}
            .card {{ max-width: 500px; margin: 0 auto; background: #131b2e; border: 1px solid #1e293b; border-radius: 16px; padding: 32px; box-shadow: 0 10px 30px rgba(0,0,0,0.5); }}
            .brand {{ text-align: center; margin-bottom: 24px; }}
            .brand-name {{ font-size: 24px; font-weight: 800; color: #3b82f6; letter-spacing: 0.5px; }}
            .title {{ font-size: 20px; font-weight: 700; color: #ffffff; text-align: center; margin-bottom: 8px; }}
            .subtitle {{ font-size: 14px; color: #94a3b8; text-align: center; margin-bottom: 24px; }}
            .otp-box {{ background: #0b0f19; border: 2px dashed #3b82f6; border-radius: 12px; padding: 20px; text-align: center; margin: 24px 0; }}
            .otp-code {{ font-size: 36px; font-weight: 800; letter-spacing: 8px; color: #3b82f6; font-family: monospace; }}
            .footer {{ font-size: 12px; color: #64748b; text-align: center; margin-top: 28px; }}
          </style>
        </head>
        <body>
          <div class="card">
            <div class="brand">
              <span class="brand-name">⚡ SnartCrypto</span>
            </div>
            <div class="title">Verify Your Email Address</div>
            <div class="subtitle">Thank you for joining SnartCrypto! Enter the verification code below to verify your email address. Valid for <b>{expires_in_minutes} minutes</b>.</div>
            
            <div class="otp-box">
              <div class="otp-code">{otp_code}</div>
            </div>
            
            <div class="footer">
              &copy; SnartCrypto. All rights reserved.
            </div>
          </div>
        </body>
        </html>
        """
        
        text_content = (
            f"SnartCrypto - Email Verification\n\n"
            f"Your 6-digit verification code is: {otp_code}\n\n"
            f"This code will expire in {expires_in_minutes} minutes."
        )
        
        return await self._dispatch_email(to_email, subject, html_content, text_content, otp_code=otp_code, purpose="EMAIL_VERIFICATION")

    async def _dispatch_email(
        self,
        to_email: str,
        subject: str,
        html_content: str,
        text_content: str,
        otp_code: Optional[str] = None,
        purpose: str = "EMAIL",
    ) -> bool:
        """Dispatches email via configured SMTP, or logs to console in Simulation Mode."""
        if not self.is_configured:
            # Dev / Simulation mode
            logger.info(
                f"\n"
                f"┌─────────────────────────────────────────────────────────────┐\n"
                f"│ 📧 [EMAIL SERVICE SIMULATION] ({purpose})                 \n"
                f"│ To: {to_email:<50} \n"
                f"│ Subject: {subject:<45} \n"
                f"│ 🔑 CODE / OTP: >>> {otp_code} <<<                          \n"
                f"│ (Configure SMTP_USER & SMTP_PASSWORD in .env for live email)│\n"
                f"└─────────────────────────────────────────────────────────────┘"
            )
            return True

        # Send via SMTP in worker thread to prevent blocking asyncio loop
        return await asyncio.to_thread(self._send_smtp_sync, to_email, subject, html_content, text_content)

    def _send_smtp_sync(self, to_email: str, subject: str, html_content: str, text_content: str) -> bool:
        """Synchronous SMTP email dispatcher."""
        try:
            host = self.settings.SMTP_HOST
            port = int(self.settings.SMTP_PORT)
            user = self.settings.SMTP_USER
            password = self.settings.SMTP_PASSWORD
            from_email = self.settings.SMTP_FROM_EMAIL or user
            from_name = self.settings.SMTP_FROM_NAME or "SnartCrypto"
            use_tls = getattr(self.settings, "SMTP_USE_TLS", True)

            msg = MIMEMultipart("alternative")
            msg["Subject"] = subject
            msg["From"] = f"{from_name} <{from_email}>"
            msg["To"] = to_email

            part1 = MIMEText(text_content, "plain", "utf-8")
            part2 = MIMEText(html_content, "html", "utf-8")
            msg.attach(part1)
            msg.attach(part2)

            if port == 465:
                # SSL
                context = ssl.create_default_context()
                with smtplib.SMTP_SSL(host, port, context=context, timeout=15) as server:
                    server.login(user, password)
                    server.sendmail(from_email, [to_email], msg.as_string())
            else:
                # STARTTLS
                with smtplib.SMTP(host, port, timeout=15) as server:
                    server.ehlo()
                    if use_tls:
                        context = ssl.create_default_context()
                        server.starttls(context=context)
                        server.ehlo()
                    server.login(user, password)
                    server.sendmail(from_email, [to_email], msg.as_string())

            logger.info(f"✅ Successfully sent email to {to_email} via {host}:{port}")
            return True

        except Exception as exc:
            logger.error(f"❌ Failed to send email to {to_email} via SMTP: {exc}")
            return False
