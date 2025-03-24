#!/bin/bash

# This script installs the necessary dependencies and sets the environment for the ScreenplaysNLP project.

# Exit on error
set -e

# Get AIRFLOW_HOME from the .env file
if ! grep -q "AIRFLOW_UID=" .env 2>/dev/null; then
	export AIRFLOW_HOME=${AIRFLOW_HOME:-/home/airflow}
fi
if ! grep -q "AIRFLOW_HOME=" .env 2>/dev/null; then
	export AIRFLOW_HOME=${AIRFLOW_HOME:-/home/airflow}
fi

# Validate AIRFLOW_HOME
if [[ -z "$AIRFLOW_HOME" || "$AIRFLOW_HOME" =~ [^a-zA-Z0-9/_-] ]]; then
	echo "Error: AIRFLOW_HOME is not set correctly or contains invalid characters."
	exit 1
fi

# Check permissions
if [ ! -w "$AIRFLOW_HOME" ]; then
	echo "Warning: No write permission to $AIRFLOW_HOME - trying with sudo"
	sudo mkdir -p "$AIRFLOW_HOME"/dags "$AIRFLOW_HOME"/logs "$AIRFLOW_HOME"/config "$AIRFLOW_HOME"/ScreenplaysNLP/Scripts "$AIRFLOW_HOME"/ScreenplaysNLP/Data	
else
	mkdir -p "$AIRFLOW_HOME"/dags "$AIRFLOW_HOME"/logs "$AIRFLOW_HOME"/config "$AIRFLOW_HOME"/ScreenplaysNLP/Scripts "$AIRFLOW_HOME"/ScreenplaysNLP/Data	
fi

# Copy Rflow.py to the dags folder
if [ -f ./Rflow.py ]; then
    cp ./Rflow.py $AIRFLOW_HOME/dags
else
    echo "Error: Rflow.py not found in the current directory."
    exit 1
fi

# Copy the scripts and data folders to the ScreenplaysNLP folder
cp -R ../Scripts/* "$AIRFLOW_HOME/ScreenplaysNLP/Scripts/"
cp -R ../Data/* "$AIRFLOW_HOME/ScreenplaysNLP/Data/"

# Verify paths after copying
if [ ! -f "$AIRFLOW_HOME/dags/Rflow.py" ]; then
	echo "Error: Failed to copy Rflow.py to $AIRFLOW_HOME/dags"
	exit 1
else 
	echo "Rflow.py copied to $AIRFLOW_HOME/dags"
fi

if [ ! -d "$AIRFLOW_HOME/ScreenplaysNLP/Scripts" ]; then
	echo "Error: Failed to copy scripts to $AIRFLOW_HOME/ScreenplaysNLP/Scripts"
	exit 1
else
	echo "Scripts copied to $AIRFLOW_HOME/ScreenplaysNLP/Scripts"
fi

if [ ! -d "$AIRFLOW_HOME/ScreenplaysNLP/Data" ]; then
	echo "Error: Failed to copy data to $AIRFLOW_HOME/ScreenplaysNLP/Data"
	exit 1
else
	echo "Data copied to $AIRFLOW_HOME/ScreenplaysNLP/Data"
fi

# Check if Python is installed
if ! command -v python >/dev/null 2>&1; then
	echo "Error: Python is not installed. Please install it and try again."
	exit 1
else
	echo "Python is installed."
fi

# Check if pip is installed
if ! command -v pip >/dev/null 2>&1; then
	echo "Error: pip is not installed. Please install it and try again."
	exit 1
else
	echo "pip is installed."
fi

# Install the required Python packages
echo "Installing the required Python packages..."
pip install -r requirements.txt

# Install Firefox
if ! command -v firefox >/dev/null 2>&1; then
	echo "Firefox is not installed. Installing now..."
	sudo apt-get update && sudo apt-get install -y \
		wget \
		bzip2 \
		libgtk-3-0 \
		libgtk-3-common \
		libasound2 \
		libdbus-glib-1-2 \
		libx11-xcb1 \
		libxt6
	sudo mkdir -p /usr/lib/firefox
	wget "https://download.cdn.mozilla.net/pub/firefox/releases/117.0b5/linux-x86_64/en-US/firefox-117.0b5.tar.bz2" -O /tmp/firefox.tar.bz2
	sudo tar -xjf /tmp/firefox.tar.bz2 -C /usr/lib/
	sudo ln -sf /usr/lib/firefox/firefox /usr/bin/firefox
	rm /tmp/firefox.tar.bz2
	firefox --version
else
	echo "Firefox is installed."
fi


# Install Geckodriver (Firefox WebDriver)
if ! command -v geckodriver >/dev/null 2>&1; then
	echo "Geckodriver is not installed. Installing now..."
	GECKO_VERSION=$(curl -s https://api.github.com/repos/mozilla/geckodriver/releases/latest | grep tag_name | cut -d '"' -f 4)
	wget https://github.com/mozilla/geckodriver/releases/download/$GECKO_VERSION/geckodriver-$GECKO_VERSION-linux64.tar.gz -O /tmp/geckodriver.tar.gz
	sudo tar -xzf /tmp/geckodriver.tar.gz -C /usr/local/bin
	sudo chmod +x /usr/local/bin/geckodriver
	rm /tmp/geckodriver.tar.gz
	geckodriver --version
else
	echo "Geckodriver is installed."
fi

# Set up virtual display for headless browser
echo "Setting up virtual display..."
sudo apt-get update && sudo apt-get install -y xvfb
export DISPLAY=:99
Xvfb :99 -screen 0 1280x1024x24 > /dev/null 2>&1 &

# Check if PostgreSQL is installed
if ! command -v psql >/dev/null 2>&1; then
	echo "PostgreSQL is not installed. Installing now..."
	sudo apt-get update && sudo apt-get install -y postgresql
else
	echo "PostgreSQL is installed."
fi

# Check for PostgreSQL development headers
if ! dpkg -l | grep -q "libpq-dev"; then
	echo "Installing PostgreSQL development headers..."
	sudo apt-get update && sudo apt-get install -y libpq-dev python3-dev
fi

# Check for PostgreSQL client
if ! dpkg -l | grep -q "postgresql-client-common"; then
	echo "Installing PostgreSQL client..."
	sudo apt-get update && sudo apt-get install -y postgresql-client-common
fi

# Add PostgreSQL to the hosts file
echo "127.0.0.1 postgres" | sudo tee -a /etc/hosts


# Check if PostgreSQL service is running
if ! sudo -u root service postgresql status >/dev/null 2>&1; then
	echo "Starting PostgreSQL service..."
	sudo -u root service postgresql start
else
	echo "PostgreSQL service is already running."
fi

# Check if the PostgreSQL user exists
if ! sudo -u postgres psql -tAc "SELECT 1 FROM pg_roles WHERE rolname='postgres_default'" | grep -q 1; then
	echo "Creating the 'postgres_default' user..."
	sudo -u postgres createuser -s postgres_default
	airflow connections add 'postgres_default' \
		--conn-type 'postgres' \
		--conn-host 'localhost' \
		--conn-login 'postgres' \
		--conn-password 'postgres' \
		--conn-port 5432 \
		--conn-schema 'postgres'
else
	echo "The 'postgres_default' user already exists."
fi

# Ensure PostgreSQL accepts connections
echo "Configuring PostgreSQL..."
sudo -u postgres psql -c "ALTER USER postgres PASSWORD 'postgres';"
sudo -u postgres psql -c "GRANT ALL PRIVILEGES ON DATABASE postgres TO postgres;"

if ! sudo -u postgres psql -lqt | cut -d \| -f 1 | grep -qw postgres; then
	sudo -u postgres psql -c "CREATE DATABASE postgres WITH OWNER postgres;"
fi

# Update pg_hba.conf to allow password authentication
sudo sed -i "s/local.*all.*all.*peer/local all all md5/" /etc/postgresql/*/main/pg_hba.conf
sudo sed -i "s/host.*all.*all.*127.0.0.1\/32.*ident/host all all 127.0.0.1\/32 md5/" /etc/postgresql/*/main/pg_hba.conf
sudo service postgresql restart

# Start Airflow
echo "Starting Airflow..."
airflow standalone
