from langchain.chains import ConversationalRetrievalChain
from langchain.embeddings import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.vectorstores import FAISS
from langchain_openai import ChatOpenAI

from src.config import DATA_DIR, EMBEDDING_MODEL, MODEL_NAME, TEMPERATURE, TOP_K


class VectorStoreManager:

    def __init__(self, openai_api_key: str):
        self.cocktails_vector_store = FAISS.load_local(
            str(DATA_DIR / "faiss_index"),
            OpenAIEmbeddings(model=EMBEDDING_MODEL, openai_api_key=openai_api_key),
            allow_dangerous_deserialization=True,
        )
        self.user_preferences_store = FAISS.from_texts(
            [""], OpenAIEmbeddings(model=EMBEDDING_MODEL, openai_api_key=openai_api_key)
        )

    def search_cocktails(self, query: str) -> list:
        return self.cocktails_vector_store.similarity_search(query, k=TOP_K)

    def search_user_preferences(self, query: str) -> list:
        return self.user_preferences_store.similarity_search(query, k=TOP_K)

    def add_user_preferences(self, preferences: str) -> None:
        self.user_preferences_store.add_texts([preferences])


class CocktailBot:

    @property
    def qa_template(self) -> str:
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

    @property
    def preference_template(self) -> str:
        return """
        Extract user preferences from indicated message. Preferences might include but not limited to: 
        - Ingredients (lemon, pear, blueberry, etc.)
        - Tastes (sweet, sour, bitter, etc.)
        - Allergies or restrictions
        Return as a comma-separated list of found preferences or 'none' if no preferences found.
        
        Message: {message}
        """

    def __init__(self, openai_api_key: str):
        self.vector_store_manager = VectorStoreManager(openai_api_key)
        self.llm = ChatOpenAI(model_name=MODEL_NAME, temperature=TEMPERATURE, openai_api_key=openai_api_key)
        self.qa_prompt = PromptTemplate(
            input_variables=["chat_history", "context", "question"], template=self.qa_template
        )
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        self.conversation = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.vector_store_manager.cocktails_vector_store.as_retriever(search_kwargs={"k": TOP_K}),
            memory=self.memory,
            combine_docs_chain_kwargs={"prompt": self.qa_prompt},
            verbose=True,
        )

    def update_user_preferences(self, message: str) -> None:
        preference_prompt = self.preference_template.format(message=message)
        preferences = self.llm.predict(preference_prompt)
        if preferences.lower() != "none":
            self.vector_store_manager.add_user_preferences(preferences)

    def search_for_recommendation(self, query: str) -> list[dict]:
        found_preferences = self.vector_store_manager.search_user_preferences(query)
        full_query = f"{query} {' '.join([doc.page_content for doc in found_preferences])}"
        similar_cocktails = self.vector_store_manager.search_cocktails(full_query)

        return [
            {
                'name': doc.metadata.get('name', ''),
                'ingredients': doc.metadata.get('ingredients', ''),
                'instructions': doc.metadata.get('instructions', ''),
            }
            for doc in similar_cocktails
        ]

    def chat(self, message: str) -> str:
        self.update_user_preferences(message)
        if any(word in message.lower() for word in ['recommend', 'suggest', 'suggestion', 'what should', 'what can']):
            recommendations = self.search_for_recommendation(message)
            formatted_recommendations = "\n".join(
                [f"- {r['name']}: {r['ingredients']} - {r['instructions']}" for r in recommendations]
            )

            response = self.conversation(
                {"question": f"{message}\n\nAvailable cocktails:\n{formatted_recommendations}"}
            )
            return response['answer']
        else:
            response = self.conversation({"question": message})
            return response['answer']
