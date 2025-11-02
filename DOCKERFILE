# Use Ubuntu base
FROM ubuntu:22.04

# Set non-interactive frontend
ARG DEBIAN_FRONTEND=noninteractive

# Install dependencies
RUN apt-get update && apt-get install -y \
    curl git python3 python3-pip \
    && apt-get clean

# Install Ollama
RUN curl -fsSL https://ollama.com/install.sh | sh

# Create workspace
WORKDIR /app

# Copy app files
COPY . .

# Install Python packages
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Pull your LLM model
RUN ollama pull qwen2.5:1.5b

# Expose Streamlit port
EXPOSE 8501

# Start Ollama first → wait → run Streamlit
CMD bash -c "ollama serve & sleep 8 && streamlit run app.py --server.address=0.0.0.0 --server.port=8501"
