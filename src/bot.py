from pathlib import Path

from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.prompts import PromptTemplate
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
import pandas as pd

from src.config import EMBEDDING_MODEL, MODEL_NAME, TEMPERATURE, TOP_K


class CocktailBot:

    @property
    def system_prompt(self) -> None:
        return PromptTemplate(
            input_variables=["history", "input"],
            template="""You are a friendly and knowledgeable bartender who helps users discover cocktails they might enjoy.
            Use the conversation history and user preferences to provide personalized bit concise recommendations.
            If you do not know the answer, you do not lie. You provide only factual information. 
            You chat only on themes that are related to cocktails and not anything else.
            
            Previous conversation:
            {history}
            
            Human: {input}
            Assistant:""",
        )

    def __init__(self, data_path: Path, openai_api_key: str):
        self.llm = ChatOpenAI(model_name=MODEL_NAME, temperature=TEMPERATURE, openai_api_key=openai_api_key)
        self.df = pd.read_csv(data_path)
        self.cocktails_vector_store = self.create_vector_store()
        self.user_preferences_store = FAISS.from_texts([""], OpenAIEmbeddings(model=EMBEDDING_MODEL))
        self.conversation = ConversationChain(
            llm=self.llm, memory=ConversationBufferMemory(), prompt=self.system_prompt
        )

    def create_vector_store(self):
        texts = []
        for _, row in self.df.iterrows():
            text = f"{row['name']} - {row['category']} - {row['alcoholic']} - {row['ingredients']}"
            texts.append(text)

        return FAISS.from_texts(texts, OpenAIEmbeddings(model=EMBEDDING_MODEL))

    def update_user_preferences(self, message: str) -> None:
        preference_prompt = f"""
        Extract user preferences from indicated message. Focus on:
        - Preferred ingredients
        - Taste preferences (sweet, sour, bitter, etc.)
        - Allergies or restrictions
        Return as a comma-separated list or 'none' if no preferences found.
        
        Message: {message}
        """

        preferences = self.llm.predict(preference_prompt)
        if preferences.lower() != "none":
            self.user_preferences_store.add_texts([preferences])

    def search_for_recommendation(self, query: str) -> list[dict]:
        found_preferences = self.user_preferences_store.similarity_search(query, k=TOP_K)
        full_query = f"{query} {' '.join([doc.page_content for doc in found_preferences])}"

        similar_cocktails = self.cocktails_vector_store.similarity_search(full_query, k=TOP_K)

        recommendations = []
        for doc in similar_cocktails:
            cocktail_description = self.df[self.df['name'].str.contains(doc.page_content)]
            if not cocktail_description.empty:
                recommendations.append(
                    {
                        'name': cocktail_description['name'].iloc[0],
                        'ingredients': cocktail_description['ingredients'].iloc[0],
                        'instructions': cocktail_description['instructions'].iloc[0],
                    }
                )

        return recommendations

    def chat(self, message: str) -> str:
        self.update_user_preferences(message)

        if any(word in message.lower() for word in ['recommend', 'suggest', 'suggestion', 'what should', 'what can']):
            recommendations = self.search_for_recommendation(message)

            formatted_recommendations = "\n".join([f"- {r['name']}: {r['ingredients']}" for r in recommendations])

            response_prompt = f"""
            Based on the user's message and these cocktail recommendations:
            {formatted_recommendations}

            Provide a concise response that:
            1. Acknowledges their preferences
            2. Presents the recommendations naturally
            """

            response = self.llm.predict(response_prompt)
        else:
            response = self.conversation.predict(input=message)

        return response
