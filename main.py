import jieba
import pandas as pd
import numpy as np
from snownlp import SnowNLP
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from transformers import BertTokenizer, BertForSequenceClassification
import torch

class ChineseBrandSentimentAnalyzer:
    def __init__(self, brand_name):
        self.brand_name = brand_name
        self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
        self.bert_model = BertForSequenceClassification.from_pretrained('bert-base-chinese', num_labels=3)
        self.sentiment_dict = self._load_sentiment_dict()

    def _load_sentiment_dict(self):
        # Placeholder for loading Chinese sentiment dictionary
        # In practice, you would load from files like:
        # - HowNet Sentiment Dictionary
        # - NTUSD (Traditional Chinese)
        # - Tsinghua sentiment lexicon
        return {
            "正面词汇": 1,
            "负面词汇": -1,
            # ... more entries
        }

    def preprocess_text(self, text):
        """Basic Chinese text preprocessing"""
        # Remove special characters and normalize
        text = text.strip().lower()
        # Segment Chinese text using jieba
        words = jieba.cut(text)
        return " ".join(words)

    def rule_based_sentiment(self, text):
        """Simple rule-based sentiment analysis"""
        snow = SnowNLP(text)
        return {
            'sentiment_score': snow.sentiments,
            'keywords': snow.keywords(3),
            'summary': snow.summary(3)
        }

    def bert_sentiment(self, text):
        """BERT-based sentiment analysis"""
        inputs = self.bert_tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        outputs = self.bert_model(**inputs)
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=1)
        return probabilities.detach().numpy()

    def analyze_social_media_data(self, data):
        """Analyze social media data for brand sentiment"""
        results = defaultdict(list)

        for text in data:
            # Preprocess
            processed_text = self.preprocess_text(text)

            # Get different sentiment scores
            rule_based = self.rule_based_sentiment(text)
            bert_scores = self.bert_sentiment(text)

            # Combine results
            results['text'].append(text)
            results['rule_based_score'].append(rule_based['sentiment_score'])
            results['bert_sentiment'].append(bert_scores.argmax())
            results['keywords'].append(rule_based['keywords'])

        return pd.DataFrame(results)

    def generate_report(self, df):
        """Generate sentiment analysis report"""
        report = {
            'brand_name': self.brand_name,
            'total_mentions': len(df),
            'average_sentiment': df['rule_based_score'].mean(),
            'sentiment_distribution': {
                'positive': (df['bert_sentiment'] == 2).sum(),
                'neutral': (df['bert_sentiment'] == 1).sum(),
                'negative': (df['bert_sentiment'] == 0).sum()
            },
            'top_keywords': self._get_top_keywords(df['keywords'])
        }
        return report

    def _get_top_keywords(self, keywords_list):
        """Extract top keywords from all mentions"""
        keyword_freq = defaultdict(int)
        for keywords in keywords_list:
            for keyword in keywords:
                keyword_freq[keyword] += 1
        return dict(sorted(keyword_freq.items(), key=lambda x: x[1], reverse=True)[:10])

# Example usage
def main():
    # Sample data (in practice, this would come from your data collection pipeline)
    sample_data = [
        "这个品牌的产品质量很好，我很满意",
        "服务态度需要改善，等待时间太长了",
        "价格合理，但是送货速度有点慢"
    ]

    # Initialize analyzer
    analyzer = ChineseBrandSentimentAnalyzer("示例品牌")

    # Analyze data
    results_df = analyzer.analyze_social_media_data(sample_data)

    # Generate report
    report = analyzer.generate_report(results_df)

    return results_df, report

if __name__ == "__main__":
    results_df, report = main()
    print(results_df)
    print(report)
