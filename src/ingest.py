from pathlib import Path
import os

from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
import pandas as pd

from .config import DATA_DIR, COCKTAIL_DATA_PATH, EMBEDDING_MODEL


def create_index(data_path: Path, save_dir: Path, openai_api_key: str) -> None:
    df = pd.read_csv(data_path)

    texts = df['ingredients'].tolist()
    metadatas = df.apply(
        lambda row: {
            'name': row['name'],
            'category': row['category'],
            'alcoholic': row['alcoholic'],
            'ingredients': row['ingredients'],
            'instructions': row['instructions'],
        },
        axis=1,
    ).tolist()

    vector_store = FAISS.from_texts(
        texts, OpenAIEmbeddings(model=EMBEDDING_MODEL, openai_api_key=openai_api_key), metadatas=metadatas
    )

    save_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(save_dir / "cocktails.csv", index=False)
    vector_store.save_local(str(save_dir / "faiss_index"))


if __name__ == "__main__":
    create_index(
        COCKTAIL_DATA_PATH,
        DATA_DIR,
        os.getenv('OPENAI_API_KEY'),
    )
