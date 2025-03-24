#!/bin/bash

set -e

# Start a virtual display for Firefox
Xvfb :99 -screen 0 1920x1080x24 > /dev/null 2>&1 &
export DISPLAY=:99

# Wait for postgres to be ready
echo "Waiting for PostgreSQL..."
sleep 5

# Initialize Airflow
airflow db migrate

# Copy DAG file
mkdir -p $AIRFLOW_HOME/dags
cp /opt/airflow/ScreenplaysNLP/Rflow.py $AIRFLOW_HOME/dags/

# Create necessary connections
if ! airflow connections get postgres_default > /dev/null 2>&1; then
    airflow connections add postgres_default \
        --conn-type postgres \
        --conn-host postgres \
        --conn-schema airflow \
        --conn-login airflow \
        --conn-password airflow \
        --conn-port 5432
else
    echo "Connection postgres_default already exists. Skipping creation."
fi
if ! airflow connections get fs_local > /dev/null 2>&1; then
	airflow connections add fs_local \
		--conn-type fs \
		--conn-extra '{"path": "/opt/airflow"}'
else
	echo "Connection fs_local already exists. Skipping creation."
fi


# Create admin user 
airflow users create \
    --username admin \
    --firstname Admin \
    --lastname User \
    --role Admin \
    --email admin@example.com \
    --password admin

# Start Airflow components separately instead of using standalone
airflow webserver -p 8080 &
airflow scheduler &

# Monitor processes
wait