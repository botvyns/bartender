from pathlib import Path

from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
import pandas as pd

from src.config import EMBEDDING_MODEL, MODEL_NAME, TEMPERATURE, TOP_K


class CocktailBot:

    @property
    def template(self) -> str:
        return """You are a friendly and knowledgeable bartender who helps users discover cocktails they might enjoy.
            Use the conversation history and user preferences to provide personalized but concise recommendations.
            If you do not know the answer, you do not lie. You provide only factual information. 
            You chat only on themes that are related to drinks and not anything else.

            Previous conversation:
            {chat_history}

            Context from cocktail database:
            {context}

            Human: {question}
            Assistant:"""

    def __init__(self, data_path: Path, openai_api_key: str):
        self.llm = ChatOpenAI(model_name=MODEL_NAME, temperature=TEMPERATURE, openai_api_key=openai_api_key)
        self.qa_prompt = PromptTemplate(input_variables=["chat_history", "context", "question"], template=self.template)
        self.df = pd.read_csv(data_path)
        self.cocktails_vector_store = self.create_vector_store()
        self.user_preferences_store = FAISS.from_texts([""], OpenAIEmbeddings(model=EMBEDDING_MODEL))

        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

        self.conversation = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.cocktails_vector_store.as_retriever(search_kwargs={"k": TOP_K}),
            memory=self.memory,
            combine_docs_chain_kwargs={"prompt": self.template},
            verbose=True,
        )

    def create_vector_store(self):
        texts = self.df['ingredients'].tolist()
        metadata = self.df.apply(
            lambda row: {'name': row['name'], 'category': row['category'], 'alcoholic': row['alcoholic']}, axis=1
        ).tolist()

        return FAISS.from_texts(texts, OpenAIEmbeddings(model=EMBEDDING_MODEL), metadata=metadata)

    def update_user_preferences(self, message: str) -> None:
        preference_prompt = f"""
        Extract user preferences from indicated message. Preferences might include but not limited to: 
        - Ingredients (lemon, pear, blueberry, etc.)
        - Tastes (sweet, sour, bitter, etc.)
        - Allergies or restrictions
        Return as a comma-separated list of found preferences or 'none' if no preferences found.
        
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
            cocktail_name = doc.metadata.get('name', '')
            cocktail_description = self.df[self.df['name'] == cocktail_name]
            if not cocktail_description.empty:
                recommendations.append(
                    {
                        'name': cocktail_description['name'].iloc[0],
                        'ingredients': cocktail_description['ingredients'].iloc[0],
                    }
                )

        return recommendations

    def chat(self, message: str) -> str:
        self.update_user_preferences(message)

        if any(word in message.lower() for word in ['recommend', 'suggest', 'suggestion', 'what should', 'what can']):
            recommendations = self.search_for_recommendation(message)
            formatted_recommendations = "\n".join([f"- {r['name']}: {r['ingredients']}" for r in recommendations])

            response = self.conversation(
                {"question": f"{message}\n\nAvailable cocktails:\n{formatted_recommendations}"}
            )

            return response['answer']
        else:
            response = self.conversation({"question": message})
            return response['answer']
