"""
Analyze what the AI discovered about the market
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import json
import tensorflow as tf


class PatternAnalyzer:
    """
    Analyze and visualize what the AI discovered
    """
    
    def __init__(self, learner):
        self.learner = learner
    
    def visualize_latent_space(self, data, labels):
        """
        Visualize the AI's learned latent space
        """
        # Get latent representations
        latent = self.learner.encoder.predict(data)
        
        # Use t-SNE to visualize
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        latent_2d = tsne.fit_transform(latent)
        
        # Plot
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Color by actual labels
        scatter1 = axes[0].scatter(
            latent_2d[:, 0], latent_2d[:, 1],
            c=labels, cmap='RdYlGn', alpha=0.6
        )
        axes[0].set_title('AI\'s Learned Representation (Colored by Actual Labels)')
        axes[0].set_xlabel('t-SNE Dimension 1')
        axes[0].set_ylabel('t-SNE Dimension 2')
        plt.colorbar(scatter1, ax=axes[0])
        
        # Color by discovered patterns
        patterns = self.learner.discover_market_patterns(data)
        scatter2 = axes[1].scatter(
            latent_2d[:, 0], latent_2d[:, 1],
            c=patterns['cluster_labels'], cmap='tab10', alpha=0.6
        )
        axes[1].set_title('AI\'s Discovered Market Patterns')
        axes[1].set_xlabel('t-SNE Dimension 1')
        axes[1].set_ylabel('t-SNE Dimension 2')
        plt.colorbar(scatter2, ax=axes[1])
        
        plt.tight_layout()
        plt.savefig('discovered_latent_space.png', dpi=300)
        plt.show()
        
        return latent_2d
    
    def analyze_pattern_characteristics(self, data, discovered_patterns):
        """
        Analyze what each discovered pattern represents
        """
        patterns = discovered_patterns['patterns']
        
        for pattern_id, pattern_info in patterns.items():
            print(f"\n{'='*60}")
            print(f"🔍 {pattern_id}")
            print('='*60)
            
            # Get samples in this pattern
            cluster_mask = discovered_patterns['cluster_labels'] == int(pattern_id.split('_')[1])
            pattern_data = data[cluster_mask]
            
            if len(pattern_data) > 0:
                # Calculate statistics
                close_prices = pattern_data[:, -1, 3]  # Close price
                volumes = pattern_data[:, -1, 4]  # Volume
                
                # Price movement
                price_change = np.diff(close_prices) / close_prices[:-1]
                
                print(f"📊 Pattern Statistics:")
                print(f"   Size: {len(pattern_data)} samples")
                print(f"   Average Close: {np.mean(close_prices):.4f}")
                print(f"   Average Volume: {np.mean(volumes):.2f}")
                print(f"   Average Price Change: {np.mean(price_change):.4%}")
                print(f"   Price Volatility: {np.std(price_change):.4%}")
                
                # What does this pattern predict?
                from sklearn.linear_model import LinearRegression
                
                # Simple regression to see what this pattern predicts
                future_returns = np.zeros(len(close_prices) - 4)
                for i in range(len(close_prices) - 4):
                    future_returns[i] = (close_prices[i+4] / close_prices[i] - 1)
                
                # Check if pattern has predictive power
                corr = np.corrcoef(price_change[:-4], future_returns)[0, 1]
                print(f"\n🎯 Predictive Power:")
                print(f"   Correlation with 4h returns: {corr:.3f}")
                print(f"   Average 4h return: {np.mean(future_returns):.4%}")
                
                # Pattern type
                if np.mean(future_returns) > 0.01:
                    print(f"   Pattern Type: 📈 BULLISH (expects upward movement)")
                elif np.mean(future_returns) < -0.01:
                    print(f"   Pattern Type: 📉 BEARISH (expects downward movement)")
                else:
                    print(f"   Pattern Type: ➡️ NEUTRAL (no clear direction)")
    
    def compare_to_human_features(self, data):
        """
        Compare AI's discovered features to human-crafted features
        """
        print("\n" + "="*60)
        print("🤖 AI vs HUMAN FEATURES COMPARISON")
        print("="*60)
        
        # Get AI's latent representation
        latent = self.learner.encoder.predict(data)
        
        # Create some human features for comparison
        closes = data[:, -1, 3]  # Close prices
        
        # Simple human features
        human_features = np.column_stack([
            np.diff(closes, prepend=closes[0]) / closes,  # Returns
            np.std(closes.reshape(-1, 20), axis=1),  # Volatility
            np.mean(closes.reshape(-1, 20), axis=1),  # Moving average
        ])
        
        print(f"\n📊 Feature Comparison:")
        print(f"   AI Features: {latent.shape[1]} dimensions")
        print(f"   Human Features: {human_features.shape[1]} dimensions")
        
        # Check if AI discovered something humans miss
        # Train simple models
        from sklearn.linear_model import LogisticRegression
        
        # Human features model
        human_model = LogisticRegression(max_iter=1000)
        human_model.fit(human_features[:-100], np.ones(len(human_features)-100))
        
        # AI features model
        ai_model = LogisticRegression(max_iter=1000)
        ai_model.fit(latent[:-100], np.ones(len(latent)-100))
        
        print(f"\n📈 Predictive Information:")
        print(f"   Human Features: {human_model.score(human_features[-100:], np.ones(100)):.2%}")
        print(f"   AI Features: {ai_model.score(latent[-100:], np.ones(100)):.2%}")
        
        return {
            'latent_dim': latent.shape[1],
            'human_dim': human_features.shape[1]
        }


def main():
    """
    Analyze discovered patterns
    """
    print("🔍 Analyzing AI's Discovered Market Patterns")
    print("="*60)
    
    # Load model
    from smartcrypto_ai_models.unconstrained_learner import UnconstrainedMarketLearner
    
    learner = UnconstrainedMarketLearner()
    learner.encoder = tf.keras.models.load_model('discovered_encoder.keras')
    learner.predictor = tf.keras.models.load_model('discovered_predictor_final.keras')
    
    # Load data
    data = np.load('prepared_data.npy')
    labels = np.load('labels.npy')
    
    # Analyze
    analyzer = PatternAnalyzer(learner)
    
    # 1. Visualize latent space
    print("\n🎨 Visualizing AI's discovered latent space...")
    analyzer.visualize_latent_space(data, labels)
    
    # 2. Discover patterns
    print("\n🔍 Discovering market patterns...")
    patterns = learner.discover_market_patterns(data)
    
    # 3. Analyze each pattern
    print("\n📊 Analyzing discovered patterns...")
    analyzer.analyze_pattern_characteristics(data, patterns)
    
    # 4. Compare to human features
    print("\n🤖 Comparing AI to human features...")
    analyzer.compare_to_human_features(data)
    
    print("\n" + "="*60)
    print("✅ Analysis complete!")
    print("="*60)


if __name__ == "__main__":
    main()