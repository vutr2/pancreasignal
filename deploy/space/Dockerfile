# Hugging Face Space (Docker SDK) for the PancreaSignal web demo.
# Copied into dist/space/ by scripts/build_space.py; edit THIS file, not the copy.
# HF builds the image on its servers (no Docker needed on your machine) and runs it as uid 1000.
#
# Python 3.14 + the lock file: the same versions that built models/urine_v1.joblib (a pickle needs them).
FROM python:3.14-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_ROOT_USER_ACTION=ignore \
    PYTHONPATH=/home/user/app/src \
    GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_SERVER_PORT=7860 \
    GRADIO_ANALYTICS_ENABLED=False

# Step 1: HF runs the container as user 1000; give it a home and an app folder
RUN useradd --create-home --uid 1000 user
WORKDIR /home/user/app

# Step 2: dependencies first (this layer is reused until the requirements change)
COPY --chown=user requirements.txt requirements-lock.txt ./
RUN pip install -r requirements.txt -c requirements-lock.txt

# Step 3: the app, owned by that user (the live RAG writes its cache under data/cache)
COPY --chown=user . .
USER user

# NVIDIA_API_KEY is NOT in the image: add it as a Space secret (Settings -> Variables and secrets);
# HF passes it to the running app as an environment variable.
EXPOSE 7860
CMD ["python", "-m", "pancreasignal.demo"]
