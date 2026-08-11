"""
Emoji utilities - Safe handling of emojis in logs
"""

import platform
import sys
import re


class EmojiUtils:
    """Utility class for handling emojis safely"""
    
    # Check if emojis are supported
    @staticmethod
    def supports_emoji() -> bool:
        """Check if the current environment supports emojis"""
        is_windows = platform.system() == 'Windows'
        console_encoding = sys.stdout.encoding.lower() if sys.stdout else 'utf-8'
        return not is_windows or 'utf-8' in console_encoding
    
    # Get safe emoji or text replacement
    @staticmethod
    def safe_emoji(emoji: str, fallback: str = None) -> str:
        """Return emoji if supported, else fallback text"""
        if EmojiUtils.supports_emoji():
            return emoji
        return fallback or emoji
    
    # Remove all emojis from text
    @staticmethod
    def remove_emojis(text: str) -> str:
        """Remove all emojis from text"""
        emoji_pattern = re.compile("["
            u"\U0001F600-\U0001F64F"
            u"\U0001F300-\U0001F5FF"
            u"\U0001F680-\U0001F6FF"
            u"\U0001F1E0-\U0001F1FF"
            u"\U00002702-\U000027B0"
            u"\U000024C2-\U0001F251"
            u"\U00002500-\U00002BFF"
            u"\U00002100-\U00002149"
            "]+", flags=re.UNICODE)
        return emoji_pattern.sub('', text)
    
    # Emoji to text mapping
    EMOJI_TO_TEXT = {
        '✅': '[OK]',
        '❌': '[ERROR]',
        '⚠️': '[WARN]',
        '🚀': '[START]',
        '📊': '[DATA]',
        '💰': '[FINANCE]',
        '📈': '[UP]',
        '📉': '[DOWN]',
        '🎯': '[TARGET]',
        '🔄': '[REFRESH]',
        '💾': '[SAVE]',
        '📁': '[FOLDER]',
        '🔍': '[SEARCH]',
        '🔧': '[TOOL]',
        '🧹': '[CLEAN]',
        '🎨': '[UI]',
        '🕒': '[TIME]',
        '⏭️': '[SKIP]',
        '🏋️': '[TRAIN]',
        '🔒': '[LOCK]',
        '📥': '[DOWNLOAD]',
        '📡': '[SIGNAL]',
        '📚': '[HISTORY]',
    }
    
    @staticmethod
    def replace_emojis(text: str) -> str:
        """Replace emojis with text equivalents"""
        for emoji, replacement in EmojiUtils.EMOJI_TO_TEXT.items():
            if emoji in text:
                text = text.replace(emoji, replacement)
        return text


# Singleton
emoji_utils = EmojiUtils()