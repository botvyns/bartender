## Usage

1. Clone this repository
2. Go to [Streamlit Community Cloud](https://streamlit.io/cloud) and sign up using your GitHub account.
3. Once logged in, click on **"New app"**
4. Select the cloned repo
5. Select the script `app.py` to run
6. In **"Advanced setting"** add your env var `OPENAI_API_KEY`
7. Click **"Deploy"**

## Time spent on the task

6 hours

## Info

`FAISS` vector database is used to create a separate vector store for cocktail dataset (using columns `name`, `category`, `alcoholic`, `ingredients`) as well as a separate vector store for user preferences. 
User preferences are extracted from user message with help of LLM. If the message contains a recommendation query, user preferences are searched for relevant details. Cocktails are retrieved from the dataset based on those details and the query. In the end, a response is generated based on found info.
For non-recommendation queries, the bot uses the ConversationChain with memory to generate a response.

## [Results](https://github.com/botvyns/bartender/tree/main/images)

Assistant actually provided cocktails with lemon (three of which are present in dataset - `Tom Collins`, `French 75`, and `Whiskey Sour`). It correcly provided 5 non-alcoholic drinks, although I couldn't find them in a dataset. It couldn't answer about my provided preferences and couldn't suggest alternative to "Hot Creamy Bush".
Overall, chatbot needs an improvement for both vectorstores. The improvements could come from experimenting with different vectors, different types of storing text and retrieval types.

**P.S.** I've spend some additional time for improvement after main development process. I've done some code refactoring and added ConversationalRetrievalChain. One of the improvements is that now the assistant remembers user preferences.
