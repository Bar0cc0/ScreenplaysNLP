-- Create postgres superuser
CREATE USER postgres WITH PASSWORD 'postgres' SUPERUSER;

-- Connect to 'postgres' database to safely drop 'airflow'
\c postgres;

-- Drop and recreate airflow database (if it exists)
DROP DATABASE IF EXISTS airflow;
CREATE DATABASE airflow;

-- Grant privileges
GRANT ALL PRIVILEGES ON DATABASE airflow TO airflow;