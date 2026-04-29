FROM python:3.11-slim

WORKDIR /app

# system deps needed for sentence-transformers + scikit-learn
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# install python deps first (layer caching — this only reruns if requirements.txt changes)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# download embedding model at BUILD time so it's baked into the image
# this means container startup is instant, no download every time
RUN python -c "\
from sentence_transformers import SentenceTransformer; \
SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2'); \
print('Embedding model downloaded successfully')"

# copy source after deps (so code changes dont invalidate the pip cache layer)
COPY . .

# Gradio default port
EXPOSE 7860

# default command runs the UI
CMD ["python", "app.py"]
