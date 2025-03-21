import logging
from typing import Dict, List, Optional
from datetime import datetime

from local_rag import LocalRAG
from feedback import Feedback, FeedbackCollector

logging.basicConfig(format="%(asctime)s: %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)


class RAGChatbot:
    def __init__(self):
        self.rag = LocalRAG()
        self.chat_history: List[Dict] = []
        self.feedback_collector = FeedbackCollector()

    def chat(self, user_input: str) -> str:
        """Process user input and return response"""
        # Add user message to history
        self.chat_history.append({"role": "user", "content": user_input})

        # Get RAG response
        try:
            response = self.rag.get_rag_response(user_input, self.chat_history)
            answer = response["answer"]

            # Add assistant response to history
            self.chat_history.append({"role": "assistant", "content": answer})

            return answer, response["sources"]
        except Exception as e:
            logger.error(f"Error in chat: {str(e)}")
            return f"I encountered an error: {str(e)}", []

    def collect_feedback(self, query: str, response: str, sources: List[Dict], rating: int, feedback_text: Optional[str] = None):
        """Collect and store feedback for a chat interaction to enable RAG system improvements.
        
        This feedback collection serves multiple purposes:
        1. Quality Monitoring: Track system performance through ratings and feedback
        2. Source Evaluation: Identify reliable/unreliable knowledge sources
        3. Training Data Generation: Create datasets for model fine-tuning
        4. Conversation Analysis: Study patterns in successful/failed interactions

        Args:
            query (str): The user's original question
            response (str): The system's generated response
            sources (List[Dict]): Knowledge sources used to generate the response
            rating (int): User rating from 1-5 (1=poor, 5=excellent)
            feedback_text (Optional[str]): Additional qualitative feedback from the user

        The feedback is stored with additional metadata:
        - timestamp: When the interaction occurred
        - chat_history: Full conversation context
        
        This data can be used to:
        - Adjust source weights in the retrieval system
        - Identify and remove problematic sources
        - Generate training data for model improvements
        - Analyze and improve conversation handling
        """
        feedback = Feedback(
            query=query,
            response=response,
            rating=rating,
            sources=sources,
            feedback_text=feedback_text,
            timestamp=datetime.now(),
            chat_history=self.chat_history.copy()
        )
        self.feedback_collector.save_feedback(feedback)


def main():
    chatbot = RAGChatbot()
    print("Welcome! I'm your RAG-powered chatbot. Type 'exit' to end the conversation.")
    print("After each response, you can rate it (1-5) and provide optional feedback.")

    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() == "exit":
            break

        response, sources = chatbot.chat(user_input)
        print(f"\nAssistant: {response}")

        # Collect feedback
        try:
            rating = int(input("\nPlease rate this response (1-5): ").strip())
            if 1 <= rating <= 5:
                feedback_text = input("Any additional feedback? (Press Enter to skip): ").strip()
                chatbot.collect_feedback(user_input, response, sources, rating, feedback_text or None)
            else:
                print("Invalid rating. Skipping feedback collection.")
        except ValueError:
            print("Invalid rating. Skipping feedback collection.")


if __name__ == "__main__":
    main()