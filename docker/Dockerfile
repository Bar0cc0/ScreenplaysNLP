FROM apache/airflow:2.7.3-python3.10

USER root

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    firefox-esr \
    postgresql \
    libpq-dev \
    xvfb \
    gcc \
    wget \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install latest geckodriver (0.36.0)
RUN wget https://github.com/mozilla/geckodriver/releases/download/v0.36.0/geckodriver-v0.36.0-linux64.tar.gz && \
    tar -xvzf geckodriver-v0.36.0-linux64.tar.gz && \
    mv geckodriver /usr/local/bin/ && \
    chmod +x /usr/local/bin/geckodriver && \
    rm geckodriver-v0.36.0-linux64.tar.gz

# Set up virtual display for Firefox
RUN apt-get update && apt-get install -y --no-install-recommends \
    xvfb \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Set up airflow user
RUN echo "airflow ALL=(ALL) NOPASSWD: ALL" > /etc/sudoers.d/airflow && \
    chmod 0440 /etc/sudoers.d/airflow

# Remove incompatible provider if it's being installed
RUN pip uninstall -y apache-airflow-providers-openlineage || true

USER airflow
WORKDIR /opt/airflow/ScreenplaysNLP

# Copy files
COPY --chown=airflow:root ./ ./

# Install Python dependencies
RUN pip install --no-cache-dir -r /opt/airflow/ScreenplaysNLP/Scripts/requirements.txt

# Set environment variables
ENV DISPLAY=host.docker.internal:0.0

# Entrypoint
COPY --chown=airflow:root ./Scripts/install.sh /opt/airflow/ScreenplaysNLP/Scripts/install.sh
ENTRYPOINT [ "bash", "/opt/airflow/ScreenplaysNLP/Scripts/install.sh" ]
EXPOSE 8080