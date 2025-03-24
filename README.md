# ScreenplaysNLP

## Overview
A natural language processing tool for analyzing screenplays and extracting insights from script content, allowing writers, researchers, and film industry professionals to gain valuable insights from script analysis.

## Features
- Web scraping techniques
- Scripts parsing and alignment
- Topic extraction and non-sequential patterns discovery
- Data storage in a PostgreSQL database
- Pipeline orchestration with Apache Airflow

## Installation (Linux or Windows/WSL)
### Method 1: Run locally
- Create a virtual environment and have it activated
- From the ScreenplaysNLP/Scripts directory, execute the startup script:
  ```bash
  bash startup.sh
  ```
- Open your web browser and navigate to `http://localhost:8080`
- Search the Airflow Standalone logs in the terminal for the lines containing "username" and "password" to retrieve your credentials
- Start the dag 'ScreenplaysNLP_pipeline'

### Method 2: Run in a Docker container
(On Windows, start Docker Desktop, go to Settings > General, and ensure "Use the WSL 2 based engine" is checked. If not, enable it and restart Docker Desktop.)
- From the ScreenplaysNLP/docker directory, build the Docker image:
```bash
docker-compose build
```
- Run the containers:
 ```bash
docker-compose up -d
```
- Give the containers a minute to initialize
- Open your web browser and navigate to `http://localhost:8080`
- Your credentials are: Username = admin, Password = admin
- Start the dag 'ScreenplaysNLP_pipeline'
- Once finished, stop the services:
```bash
docker-compose down
```

### Method 3: Run scripts without the full pipeline setup (no Airflow nor PostgreSQL)
- Create a virtual environment and have it activated
- Install dependencies:
  ```bash
  pip install -r requirements.txt
  ```
- Execute scripts in this order:
  ```bash
  python CreateDataset.py
  python ModelDataset.py
  ```

## Requirements
- Linux or Windows with WSL
- Python 3.10 or higher 
- PostgreSQL database
- Apache Airflow
- Xserver (for non-Debian platforms to allow Selenium and Firefox to work). Xserver enables graphical display forwarding; you can refer to [this guide](https://wiki.archlinux.org/title/Xorg#Installation) for setup instructions.
- Docker Desktop (for Method 2, version 4.0.0 or higher recommended)

