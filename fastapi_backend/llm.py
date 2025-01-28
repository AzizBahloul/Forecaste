from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
import logging
from typing import Dict, Any
from functools import lru_cache
import os

class LLMAnalyzer:
    def __init__(self):
        self.logger = logging.getLogger('LLMAnalyzer')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_name = "google/flan-t5-base"
        self.max_length = 512
        self._load_model()

    def _load_model(self):
        try:
            self.tokenizer = T5Tokenizer.from_pretrained(
                self.model_name, 
                cache_dir="models/t5_cache"
            )
            self.model = T5ForConditionalGeneration.from_pretrained(
                self.model_name,
                cache_dir="models/t5_cache"
            )
            self.model.to(self.device)
            self.logger.info(f"Model loaded on {self.device}")
        except Exception as e:
            self.logger.error(f"Model loading failed: {e}")
            raise RuntimeError(f"T5 model initialization failed: {e}")

    @lru_cache(maxsize=100)
    def _generate_insights(self, 
                         predicted_units: int,
                         base_price: float,
                         total_price: float,
                         discount: float) -> str:
        prompt = f"""
        Analyze this sales prediction:
        - Predicted sales: {predicted_units} units
        - Base price: ${base_price:.2f}
        - Sale price: ${total_price:.2f}
        - Discount: {discount:.1f}%
        
        Provide a concise business analysis.
        """
        
        inputs = self.tokenizer(
            prompt, 
            return_tensors="pt",
            max_length=self.max_length,
            truncation=True
        ).to(self.device)

        outputs = self.model.generate(
            **inputs,
            max_length=200,
            num_beams=4,
            temperature=0.7,
            top_p=0.9,
            do_sample=True
        )

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def analyze_prediction(self, prediction_data: Dict[str, Any]) -> Dict[str, Any]:
        try:
            input_data = prediction_data['metadata']['input_data']
            predicted_units = prediction_data['predicted_units']
            
            # Calculate discount percentage
            discount = ((input_data['base_price'] - input_data['total_price']) 
                       / input_data['base_price'] * 100)
            
            analysis = self._generate_insights(
                predicted_units=predicted_units,
                base_price=input_data['base_price'],
                total_price=input_data['total_price'],
                discount=abs(discount)
            )

            return {
                "analysis": analysis,
                "metadata": {
                    "model": self.model_name,
                    "device": str(self.device),
                    "confidence": 0.85  # Placeholder for model confidence
                }
            }

        except Exception as e:
            self.logger.error(f"Analysis generation failed: {e}")
            return {
                "analysis": "Analysis generation failed",
                "error": str(e)
            }

# Initialize singleton instance
analyzer = LLMAnalyzer()

if __name__ == "__main__":
    # Test prediction
    test_data = {
        "predicted_units": 49,
        "metadata": {
            "input_data": {
                "base_price": 100.0,
                "total_price": 90.0,
                "is_featured_sku": 1,
                "is_display_sku": 1,
                "sku_id": 9632
            }
        }
    }
    
    result = analyzer.analyze_prediction(test_data)
    print("\nAnalysis Result:")
    print(result["analysis"])