#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
author: Michael Garancher
date: 2021-09-01
Description: 
	Screenplay ETL Pipeline
	This DAG performs the following operations:
	1. Extracts screenplay data by running CreateDataset.py
	2. Transforms the data using topic modeling (ModelDataset.py)
	3. Loads the processed data into PostgreSQL
Notes:
	The pipeline does not have a schedule.
	To run the pipeline manually, execute the following command:
	$ airflow dags trigger ETL_pipeline_adhoc
"""


# Standard library
import sys, os, warnings, logging
import csv, re
from abc import ABC, abstractmethod
from typing import Any, Optional
from datetime import timedelta
import pendulum

# Airflow
from airflow.models.dag import DAG
from airflow.hooks.base import BaseHook
from airflow.operators.bash import BashOperator 
from airflow.operators.python import PythonOperator, ShortCircuitOperator
from airflow.operators.empty import EmptyOperator
from airflow.providers.postgres.hooks.postgres import PostgresHook
from airflow.providers.common.sql.operators.sql import SQLExecuteQueryOperator
from airflow.sensors.filesystem import FileSensor
from airflow.utils.task_group import TaskGroup


# Suppress warnings
sys.tracebacklimit = 0
warnings.filterwarnings("ignore")


# Project configuration
AIRFLOW_HOME = os.environ.get('AIRFLOW_HOME', '/home/airflow')
CONFIG = {
    "data_dir": os.path.join(AIRFLOW_HOME, "ScreenplaysNLP", "Data"),
    "scripts_dir": os.path.join(AIRFLOW_HOME, "ScreenplaysNLP", "Scripts"),
    "excel_filename": "bttf.xlsx",
    "table_name": "bttf"
}

# Create directories if they don't exist
for path in [CONFIG['data_dir']]:
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
        print(f"Created directory: {path}")

# Check directories exist
for path in [CONFIG['data_dir'], CONFIG['scripts_dir']]:
    if not os.path.exists(path):
        logging.warning(f"Directory not found: {path}")


# Airflow DAG configuration
DAG_ARGS:dict = {
	"owner": "Bar0cc0",
	"start_date": pendulum.today('UTC').add(days=0), 
	"retries": 1,
	"retry_delay": timedelta(minutes=5),
	"template_searchpath":"~/airflow",
	"wait_for_downstream": True,
	"catchup": False,
	"email": None,
	"email_on_failure": False
}

# Logging
logging.basicConfig(level=logging.INFO,	format='%(asctime)s - %(levelname)s - %(message)s')



class ConnectPSQL:
	_instance = None

	def __init__(self, conn_id):
		self.conn_id = conn_id or "postgres_default"
		self.conn = BaseHook.get_connection(self.conn_id)
		self.conn_type = self.conn.conn_type or "postgres"
		self.host = self.conn.host or "localhost"
		self.login = self.conn.login or "postgres"
		self.password = self.conn.password or "postgres"
		self.port = self.conn.port or 5432
		self.database = self.conn.schema or "postgres"
	
	def __new__(cls, conn_id=None):
		if cls._instance is None:
			cls._instance = super().__new__(cls)
		return cls._instance

	def validate_connection(self):
		try:
			hook = PostgresHook(postgres_conn_id=self.conn_id)
			hook.get_conn() 
			logging.info(f"Successfully validated connection to {self.conn_id}")
			return True
		except Exception as e:
			logging.error(f"Failed to connect to database: {e}")
			return False
		
	def debug(self):
		try:
			logging.info("Debugging connection...")
			hook = PostgresHook(postgres_conn_id=self)
			conn_params = hook.get_connection(self.conn_id)
			logging.info(f"Connection params: host={conn_params.host}, port={conn_params.port}")
			
			# Test with explicit localhost
			logging.info("Testing direct localhost connection...")
			import psycopg2
			direct_conn = psycopg2.connect(
				host="localhost",
				port=5432,
				user="postgres",
				password="postgres",
				database="postgres"
			)
			direct_conn.close()
			logging.info("Direct connection successful")
		except Exception as e:
			logging.error(f"Connection debugging failed: {e}")

class ITask(ABC):
	"""Interface for task creation."""
	@abstractmethod
	def create_task(self, name:str, context:Optional[ConnectPSQL]=None, config:Optional[dict]=None, **kwargs) -> Any:
		raise NotImplementedError("Method not implemented")

class BashTask(ITask):
	"""Create a Bash task to run a Python script."""
	def create_task(self, name:str, config:dict=None, **kwargs) -> Any:
		config = config or CONFIG
		script = kwargs.get("script", None)
		if not script:
			raise ValueError(f"No command provided for task {name}.")
		try:
			return BashOperator(
				task_id = name,
				bash_command = script,
				params={
					'table_name': config.get('table_name'),
					'filepath': f"{config.get('data_dir')}/{config.get('excel_filename')}"
				}
			)
		except Exception as e:
			logging.error(f"Error creating Bash task {name}: {e}")
			raise

class SQLTask(ITask):
	"""Create a SQL task that creates a table and insert data."""
	def create_task(self, name:str, context:ConnectPSQL, config:dict=None, **kwargs) -> Any:
		config = config or CONFIG
		script = kwargs.get("script", None)
		if not script:
			raise ValueError(f"No SQL script provided for task {name}.")
		if not context:
			raise ValueError("No connection context provided for SQL task.")
		try:
			test_hook = PostgresHook(postgres_conn_id=context.conn_id)
			test_hook.get_conn()
		except Exception as e:
			logging.warning(f"Connection test failed: {e}, updating hook params")
			hook_params = {
				"schema": "postgres",
				"host": "localhost",
				"login": "postgres",
				"password": "postgres",
				"port": 5432
			}
		else:
			hook_params = {
				"schema": context.database,
				"host": context.host,
				"login": context.login,
				"password": context.password,
				"port": context.port
			}
		try:
			return SQLExecuteQueryOperator(
				task_id = name,
				hook_params = hook_params,
				sql = script,
				params = {"table_name": config.get("table_name")},
			)
		except Exception as e:
			logging.error(f"Error creating SQL task {name}: {e}")
			raise

class PythonTask(ITask):
	"""Create a Python task."""
	def create_task(self, name:str, config:dict=None, **kwargs) -> Any:
		config = config or CONFIG
		py_callable = kwargs.get("script", None)
		if not py_callable:
			raise ValueError(f"No Python callable provided for task {name}.")
		try:
			return PythonOperator(
				task_id = name,
				python_callable = py_callable
			)
		except Exception as e:
			logging.error(f"Error creating Python task {name}: {e}")
			raise

class ShortCircuitTask(ITask):
	"""Create a ShortCircuit task."""
	def create_task(self, name:str, config:dict=None, **kwargs) -> Any:
		config = config or CONFIG
		py_callable = kwargs.get("script", None)
		if not py_callable:
			raise ValueError(f"No Python callable provided for task {name}.")
		try:
			return ShortCircuitOperator(
				task_id = name,
				python_callable = py_callable
			)
		except Exception as e:
			logging.error(f"Error creating ShortCircuit task {name}: {e}")
			raise

class FileSensorTask(ITask):
	"""Create a FileSensor task."""
	def create_task(self, name:str, config:dict=None, **kwargs) -> Any:
		config = config or CONFIG
		filepath = kwargs.get("filepath", None)
		if not filepath:
			raise ValueError(f"No file path provided for task {name}.")
		try:
			return FileSensor(
				task_id = name,
				filepath = filepath,
				poke_interval = 60,
				timeout = 600
			)
		except Exception as e:
			logging.error(f"Error creating FileSensor task {name}: {e}")
			raise

class TaskFactory:
	"""Factory for task creation."""
	_instance:object = None
	_task_types:dict = {
				"bash": BashTask(),
				"sql": SQLTask(),
				"python": PythonTask(),
				"short_circuit": ShortCircuitTask(),
				"file_sensor": FileSensorTask()
			}
	
	def __new__(cls, *args, **kwargs) -> object:
		if cls._instance is None:
			cls._instance = super().__new__(cls)
		return cls._instance
	
	@classmethod
	def register_task_type(cls, type_name:str, implementation:ITask) -> None:
		cls._task_types[type_name] = implementation

	@classmethod
	def create_task(cls, task_type:str, name:str, **kwargs) -> object:
		task = cls._task_types.get(task_type)
		if not task:
			raise ValueError(f"Invalid task type: {task_type}")
		if not name:
			raise ValueError(f"No task name provided for task type {task_type}.")
		return task.create_task(name, **kwargs)

	@classmethod
	def create_task_chain(cls, tasks_config:list) -> list:
		"""Create a chain of tasks from configuration."""
		task_list = []
		for task_config in tasks_config:
			
			task = cls.create_task(**task_config)
			task_list.append(task)
		return task_list
	

def process_csv_for_postgres(**context):
    # Get the file path
    excel_file = str(context['dag_run'].conf.get('data_dir', 
        os.path.join(os.environ.get('AIRFLOW_HOME'), 'ScreenplaysNLP', 'Data'))) + '/' + context['dag_run'].conf.get('excel_filename', 'bttf.xlsx')
    
    # Get csv filename
    input_file = excel_file.replace('.xlsx', '.csv')
    output_file = input_file.replace('.csv', '_fixed.csv')
    
    # Convert Excel to CSV if needed
    if not os.path.exists(input_file):
        import subprocess
        subprocess.run(["xlsx2csv", excel_file, input_file], check=True)
    
    # Process the CSV
    with open(input_file, 'r') as infile, open(output_file, 'w', newline='') as outfile:
        reader = csv.reader(infile)
        writer = csv.writer(outfile)
        
        # Get headers
        headers = next(reader)
        needed_headers = headers[1:] 
        writer.writerow(needed_headers)
        
        # Process data rows
        for row in reader:
            # Only write the 6 columns we need
            processed_row = row[1:]
            
            # Fix time format (replace comma with period and ensure valid time format)
            if len(processed_row) > 1:
                time_col = processed_row[1]
                
                # Handle invalid time values
                if time_col == "0" or time_col == "":
                    processed_row[1] = "00:00:00"
                elif ',' in time_col:
                    processed_row[1] = time_col.replace(',', '.')
                # Make sure it's in HH:MM:SS format
                elif ':' not in time_col:
                    processed_row[1] = "00:00:00"
            
            writer.writerow(processed_row)
    
    return output_file

def validate_data(filepath):
	"""Validate data in CSV file before loading."""
	with open(filepath, 'r') as f:
		reader = csv.reader(f)
		headers = next(reader)
		
		for i, row in enumerate(reader, start=2):
			# Check time format in column 1 (index 0-based)
			if len(row) > 1:
				time_value = row[1]
				if not bool(re.match(r'^\d{2}:\d{2}:\d{2}(\.\d+)?$', time_value)):
					print(f"WARNING: Invalid time format at row {i}: '{time_value}'")

def check_extract_success(**context):
	task_instance = context['ti']
	extract_tasks = task_instance.xcom_pull(task_ids=['extract.run_extract_script'])
	return all(result is not None for result in extract_tasks)

def check_load_success(**context):
	task_instance = context['ti']
	load_tasks = task_instance.xcom_pull(task_ids=['load'])
	return all(result is not None for result in load_tasks)

def alert_callback(context):
	"""Send an alert message when task fails."""
	task_id = context['task_instance'].task_id
	dag_id = context['task_instance'].dag_id
	exec_date = context['execution_date']
	error_message = context.get('exception', None)

	message = f"DAG {dag_id} failed on {exec_date} - Task: {task_id} - Error: {error_message}"
	logging.error(message)

	try:
		pass # Not implemented yet
	except Exception as e:
		logging.error(f"Failed to send alert: {e}")
	
	return message



# Connection string with fallback
args = DAG_ARGS.copy()
try:
	cs = ConnectPSQL(conn_id="postgres_default")
	if not cs.validate_connection():
		logging.warning("Connection validation failed, using localhost fallback")
		cs.host = "localhost"  
except Exception as e:
	logging.error(f"Error setting up connection: {e}")
	cs.debug()
	cs = type('ConnectionString', (), {
		'conn_id': 'postgres_default',
		'conn_type': 'postgres',
		'host': 'localhost',
		'login': 'postgres',
		'password': 'postgres',
		'port': 5432,
		'database': 'postgres'
	})
args.update(cs.__dict__)


# Chain of tasks to process data and load into PostgreSQL
load_taskchain = [
	{"task_type": "python", "name": "process_data", "script": process_csv_for_postgres},
	{"task_type": "sql", "name": "create_table", "script": """
		DROP TABLE IF EXISTS {{ params.table_name }};
		CREATE TABLE {{ params.table_name }}(
			index INTEGER PRIMARY KEY NOT NULL,
			timecode TEXT,
			part TEXT,
			srt_dialogue TEXT,
			script_dialogue TEXT,
			similarity FLOAT
		);
	""", "context": cs, "config": CONFIG},
	{"task_type": "bash", "name": "insert_values", "script": """
		xlsx2csv {{ params.filepath }} {{ params.filepath.replace('.xlsx', '.csv') }} && \
		PGPASSWORD=postgres psql -h localhost -U postgres -d postgres -c "\\COPY {{ params.table_name }} FROM '{{ params.filepath.replace('.xlsx', '_fixed.csv') }}' WITH CSV HEADER"
	""", "context": cs, "config": CONFIG},
]


# DAG definition
with DAG(
	dag_id = "ScreenplaysNLP_pipeline", 
	default_args = args,
	on_failure_callback= alert_callback,
	description="ETL pipeline for screenplay data processing and topic extraction",
	tags=["etl", "screenplay", "nlp", "postgres"],
	) as dag:
	
	# Create dataset
	extract = TaskFactory.create_task(
		task_type = "bash",
		name = "extract", 
		script = f"python3 {os.path.abspath(os.path.join(str(AIRFLOW_HOME), 'ScreenplaysNLP', 'Scripts', 'CreateDataset.py'))}"
	)
	
	# Check if extraction was successful
	extract_success = TaskFactory.create_task(
		task_type = "short_circuit",
		name = "is_extract_success",
		script = check_extract_success
	)

	# Analyse data
	transform = TaskFactory.create_task(
		task_type = "bash",
		name = "transform",
		script = f"python3 {os.path.abspath(os.path.join(str(AIRFLOW_HOME), 'ScreenplaysNLP', 'Scripts', 'ModelDataset.py'))}"
	)

	# Wait for the .xlsx file to be available
	wait_file = TaskFactory.create_task(
		task_type = "file_sensor",
		name = "wait_for_file",
		filepath = os.path.join(CONFIG["data_dir"], CONFIG["excel_filename"])
	)

	# Load into database
	load = TaskFactory.create_task_chain(load_taskchain)

	# Check load was successful
	load_success = TaskFactory.create_task(
		task_type = "short_circuit",
		name = "is_load_success",
		script = check_load_success
	)

	# Declare sequential pipeline 
	extract >> extract_success >> transform >> wait_file >> load >> load_success

