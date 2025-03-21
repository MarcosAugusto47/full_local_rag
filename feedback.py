from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional
import json
import os
import logging

logging.basicConfig(format="%(asctime)s: %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Feedback:
    query: str
    response: str
    rating: int  # 1-5 scale
    sources: List[Dict]
    feedback_text: Optional[str]
    timestamp: datetime
    chat_history: List[Dict]

    def to_dict(self) -> Dict:
        return {
            "query": self.query,
            "response": self.response,
            "rating": self.rating,
            "sources": self.sources,
            "feedback_text": self.feedback_text,
            "timestamp": self.timestamp.isoformat(),
            "chat_history": self.chat_history
        }

class FeedbackCollector:
    def __init__(self, feedback_file: str = "feedback_data.json"):
        self.feedback_file = feedback_file
        self._load_feedback()

    def _load_feedback(self):
        """Load existing feedback from file"""
        if os.path.exists(self.feedback_file):
            try:
                with open(self.feedback_file, 'r') as f:
                    self.feedback_data = json.load(f)
            except Exception as e:
                logger.error(f"Error loading feedback data: {str(e)}")
                self.feedback_data = []
        else:
            self.feedback_data = []

    def save_feedback(self, feedback: Feedback):
        """Save new feedback to storage"""
        try:
            self.feedback_data.append(feedback.to_dict())
            with open(self.feedback_file, 'w') as f:
                json.dump(self.feedback_data, f, indent=2)
            logger.info("Feedback saved successfully")
        except Exception as e:
            logger.error(f"Error saving feedback: {str(e)}")
            raise 