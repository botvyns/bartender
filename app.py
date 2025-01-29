import os
import time

import streamlit as st

from src.bot import CocktailBot


st.set_page_config(page_title="🍸 Cocktail Recommendation Chatbot")

if 'bot' not in st.session_state:
    st.session_state.bot = CocktailBot(openai_api_key=os.getenv('OPENAI_API_KEY'))

if 'messages' not in st.session_state:
    st.session_state['messages'] = [
        {
            "role": "assistant",
            "content": "Hi! I'm your personal cocktail recommender. Tell me about your preferences or ask for recommendations!",
        }
    ]

for message in st.session_state.get('messages', []):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if query := st.chat_input("What kind of cocktail would you like?"):
    st.session_state.setdefault('messages', []).append({"role": "user", "content": query})

    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        response = st.session_state.bot.chat(query)
        recommendations = []

        if any(word in query.lower() for word in ['recommend', 'suggest', 'what should', 'what can']):
            recommendations = st.session_state.bot.search_for_recommendation(query)

        full_response = response

        message_placeholder.markdown(full_response + "▌")
        time.sleep(0.05)

        if recommendations:
            st.divider()
            for rec in recommendations:
                st.write(f"🍸 **{rec['name']}**")
                st.write(f"Ingredients: {rec['ingredients']}")
                st.write(f"Instructions: {rec['instructions']}")
                st.write("---")

        message_placeholder.markdown(full_response)

    st.session_state.setdefault('messages', []).append(
        {"role": "assistant", "content": full_response, "recommendations": recommendations}
    )
