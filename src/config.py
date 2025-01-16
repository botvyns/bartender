from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"

COCKTAIL_DATA_PATH = DATA_DIR / "cocktails.csv"

# MODEL_NAME = "gpt-3.5-turbo"
MODEL_NAME = "gpt-4o-mini"
EMBEDDING_MODEL = "text-embedding-3-small"
TEMPERATURE = 0.7
TOP_K = 3
