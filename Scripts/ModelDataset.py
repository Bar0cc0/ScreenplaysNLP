#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
author = Michael Garancher
date: 2023-09-01
description:

	This script extracts topics from text datasets using a combination of LDA and LSTM models.

	Screenplays - Topic Extraction
	--------------------------------------------
	1. Preprocess data
	2. Estimate best params
	3. Embeddings generation
	4. Topic inference
	5. Evaluate model's performance

	The algorithm is demonstrated using dialogues from the movie "Back to the Future" (1985) 
	but can be generalized to other datasets with similar structures.

'''

# Standard library 
import os, sys, gc
import warnings, traceback, logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

# Third-party libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

# NLP libraries
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.decomposition import LatentDirichletAllocation
import tensorflow as tf
import tensorflow_text as tf_text
from keras.models import Sequential
from keras.backend import clear_session
from keras.layers import TextVectorization, Dense, Dropout, Embedding, LSTM



# Suppress warnings
sys.tracebacklimit = 0
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Pandas/Matplotlib settings
pd.set_option('display.max_columns', None)
plt.style.use('ggplot')

# Configuration setup
ROOT: Path = Path(__file__).parents[1]
CONFIG: Dict[str, Path] = {
	"OUTPUT_DIR": ROOT / 'Data',
	"SOURCE": ROOT / 'Data/bttf.xlsx',
	"GLOVE": ROOT / 'Data/models/glove.6B.50d.txt'
}

# Constants
DEFAULT_SEQUENCE_LENGTH = 100
DEFAULT_EMBEDDING_DIM = 50
DEFAULT_MAX_FEATURES = 5000

# Configure logging
logging.basicConfig(
	level=logging.INFO,
	format='%(asctime)s - %(levelname)s - %(message)s',
	handlers=[
		logging.StreamHandler(),
		logging.FileHandler(CONFIG['OUTPUT_DIR'] / 'topic_extraction.log', mode='w')
	]
)


class ModelError(Exception):
	"""Base class for model-related exceptions"""
	pass

class DataError(Exception):
	"""Base class for data-related exceptions"""
	pass

class ConfigError(Exception):
	"""Base class for configuration-related exceptions"""
	pass

class TextProcessor:
	"""Handle data loading, cleaning, vectorization"""
	
	def __init__(self, config: Dict[str, Path]):
		self.config = config
		self.vocab = None
		
	@staticmethod
	def normalize(text: str) -> str:
		"""Normalize text by lowercasing, normalizing Unicode, and removing special characters."""
		text = tf_text.case_fold_utf8(text)  # Lowercasing
		text = tf_text.normalize_utf8(text, 'NFKD')  # Unicode normalization
		text = tf.strings.regex_replace(text, '[^a-zA-Z0-9 ]', '')  # Remove special chars
		return text
	
	def vectorize(self, data: Union[pd.Series, pd.DataFrame], 
				parameters: Optional[Dict[str, Any]] = None,
				sequence_length: int = DEFAULT_SEQUENCE_LENGTH) -> tf.Tensor:
		"""Convert text data to numerical vectors suitable for model input."""
		try:
			# Handle parameters or use defaults
			if parameters is None:
				logging.warning("No parameters provided for vectorization, using defaults")
				max_tokens = DEFAULT_MAX_FEATURES
				ngrams = 2
			else:
				try:
					max_tokens = parameters['vectorize__max_features']
					ngrams = parameters['vectorize__ngram_range'][1]
				except KeyError as e:
					logging.error(f"Missing parameter: {e}, using defaults")
					max_tokens = DEFAULT_MAX_FEATURES
					ngrams = 2
			
			# Create and adapt vectorizer
			vectorizer = TextVectorization(
				max_tokens=max_tokens,
				standardize='lower_and_strip_punctuation',
				split='whitespace',
				ngrams=ngrams,
				output_mode='int',
				output_sequence_length=sequence_length
			)
			
			# Ensure data is in proper format
			if isinstance(data, pd.DataFrame):
				if 'script_dialogue' in data.columns:
					text_data = data['script_dialogue'].astype(str)
				else:
					raise DataError("DataFrame missing 'script_dialogue' column")
			elif isinstance(data, pd.Series):
				text_data = data.astype(str)
			else:
				text_data = data
			
			# Adapt vectorizer to data
			vectorizer.adapt(text_data)
			
			# Store vocabulary for later use
			self.vocab = vectorizer.get_vocabulary()
			
			if len(self.vocab) == 0:
				logging.warning("Empty vocabulary after vectorization")
			
			# Return vectorized data
			return vectorizer(text_data)
			
		except tf.errors.ResourceExhaustedError as e:
			logging.error(f"Out of memory during vectorization: {e}")
			raise MemoryError("Out of memory during text vectorization") from e
		except Exception as e:
			logging.error(f"Error in vectorize: {type(e).__name__}: {e}")
			logging.error(traceback.format_exc())
			raise
	
	def save_results(self, y_pred: np.ndarray, original_data: pd.DataFrame, 
					topic_mapping: Optional[Dict[int, str]] = None) -> None:
		"""Save prediction results to an Excel file with topic information."""
		# Handle y_pred type: probabilities vs class indices
		if len(y_pred.shape) > 1 and y_pred.shape[1] > 1:
			predicted_topics = np.argmax(y_pred, axis=1)
			topic_probabilities = np.max(y_pred, axis=1)
		else:
			predicted_topics = y_pred.flatten()
			topic_probabilities = np.ones_like(predicted_topics)  # Dummy values
		
		# Use provided topic mapping or create a generic one
		if topic_mapping is None:
			topic_mapping = {i: f"Topic {i+1}" for i in range(len(np.unique(predicted_topics)))}
		
		# Create results DataFrame with topic names
		topic_names = [topic_mapping[int(idx)] for idx in predicted_topics]
		
		# Create a results DataFrame
		results_df = pd.DataFrame({
			'predicted_topic_id': predicted_topics,
			'predicted_topic_name': topic_names,
			'confidence': topic_probabilities
		})
		
		# Combine with original data
		df_results = pd.concat([original_data.reset_index(drop=True), results_df], axis=1)
		
		# Generate output filename
		timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
		output_path = self.config['OUTPUT_DIR'] / f'predicted_topics_{timestamp}.xlsx'
		
		# Save to Excel
		try:
			with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
				df_results.to_excel(writer, sheet_name='Topic Analysis', index=False)
				
				# Add a summary sheet with topic distributions
				topic_counts = pd.DataFrame({
					'topic_id': list(topic_mapping.keys()),
					'topic_name': list(topic_mapping.values()),
					'count': [sum(predicted_topics == i) for i in range(len(topic_mapping))]
				})
				topic_counts.to_excel(writer, sheet_name='Topic Distribution', index=False)
			
			logging.info(f"Results saved to {output_path}")
		except Exception as e:
			logging.error(f"Error saving results: {e}")
	
	@staticmethod
	def load_source(filepath: Path) -> pd.DataFrame:
		"""Load source data from Excel file."""
		try:
			data = pd.read_excel(filepath)
			if data.empty:
				raise DataError('Empty dataset from source file.')
			
			if 'script_dialogue' not in data.columns:
				raise DataError(f"Required column 'script_dialogue' not found. Available columns: {', '.join(data.columns)}")
			
			# Check for NaN values
			nan_count = data['script_dialogue'].isna().sum()
			if nan_count > 0:
				logging.warning(f"Dataset contains {nan_count} NaN values in 'script_dialogue' column")
				data = data.dropna(subset=['script_dialogue'])
				logging.info(f"Dropped {nan_count} rows with NaN values")
			
			return data
			
		except FileNotFoundError:
			raise DataError(f"Data file not found: {filepath}")
		except pd.errors.EmptyDataError:
			raise DataError(f"Data file is empty: {filepath}")
		except pd.errors.ParserError as e:
			raise DataError(f"Error parsing Excel file: {e}")
		except Exception as e:
			logging.error(f"Error loading data: {type(e).__name__}: {e}")
			logging.error(traceback.format_exc())
			raise


class ModelHandler:
	"""Handle model creation, saving, loading"""
	
	def __init__(self, config: Dict[str, Path]):
		self.config = config
	
	def create_model(self, model_type: str, params: Dict[str, Any]) -> Any:
		"""Create a model based on type and parameters."""
		if model_type == 'lda':
			# Create LDA pipeline
			return Pipeline([
				('vectorize', CountVectorizer(
					analyzer='word',
					stop_words='english',
					max_df=0.9, min_df=0.1,
					max_features=params.get('max_features', DEFAULT_MAX_FEATURES),
					ngram_range=params.get('ngram_range', (1, 1))
				)),
				('reduce_dim', LatentDirichletAllocation(
					n_components=params.get('n_components', 8),
					batch_size=150,
					learning_method='online',
					learning_offset=15,
					random_state=0,
					evaluate_every=1,
					max_iter=params.get('max_iter', 500),
					n_jobs=-1
				))
			])
		elif model_type == 'lstm':
			# Get parameters
			vocab_size = params.get('vocab_size', DEFAULT_MAX_FEATURES + 1)
			embedding_dim = params.get('embedding_dim', DEFAULT_EMBEDDING_DIM)
			input_length = params.get('input_length', DEFAULT_SEQUENCE_LENGTH)
			n_topics = params.get('n_topics', 1)
			embedding_matrix = params.get('embedding_matrix', None)
			
			# Create LSTM model
			model = Sequential([
				Embedding(
					input_dim=vocab_size,
					output_dim=embedding_dim,
					input_length=input_length,
					weights=[embedding_matrix] if embedding_matrix is not None else None,
					trainable=True,
					name='embeddings'
				),
				LSTM(64, return_sequences=True),
				LSTM(32),
				Dropout(0.3),
				Dense(20, activation='relu'),
				Dense(n_topics, activation='softmax')
			])
			
			# Compile model
			model.compile(
				loss='categorical_crossentropy',
				optimizer='adam',
				metrics=['accuracy']
			)
			
			return model
		else:
			raise ValueError(f"Unknown model type: {model_type}")
	
	def save_model(self, model: Any, model_type: str) -> Path:
		"""Save model to a file."""
		models_dir = self.config['OUTPUT_DIR'] / 'models'
		if not models_dir.exists():
			models_dir.mkdir(parents=True, exist_ok=True)
		
		if model_type == 'lda':
			filename = models_dir / 'lda_model.pkl'
			joblib.dump(model, filename)
		elif model_type == 'lstm':
			filename = models_dir / 'lstm_model.h5'
			tf.keras.models.save_model(model, filename)
		else:
			raise ValueError(f"Invalid model type: {model_type}")
		
		logging.info(f"Model saved to {filename}")
		return filename
	
	def load_model(self, model_type: str) -> Tuple[Any, bool]:
		"""Load model from a file. Returns (model, success)."""
		models_dir = self.config['OUTPUT_DIR'] / 'models'
		
		try:
			if not models_dir.exists():
				models_dir.mkdir(parents=True, exist_ok=True)
				logging.warning(f"Model directory created but no models found in {models_dir}")
				return None, False
			
			if model_type == 'lda':
				filename = models_dir / 'lda_model.pkl'
				if not filename.exists():
					logging.warning(f"LDA model file not found: {filename}")
					return None, False
				
				model = joblib.load(filename)
				
				# Validate model type
				if not hasattr(model, 'named_steps'):
					raise ModelError("Invalid LDA model: missing 'named_steps' attribute")
				
				logging.info(f"LDA model loaded from {filename}")
				return model, True
				
			elif model_type == 'lstm':
				filename = models_dir / 'lstm_model.h5'
				if not filename.exists():
					logging.warning(f"LSTM model file not found: {filename}")
					return None, False
				
				# Try with compile=False first, then with compile=True if that fails
				try:
					model = tf.keras.models.load_model(filename, compile=False)
					# Recompile with original settings
					model.compile(
						loss='categorical_crossentropy',
						optimizer='adam',
						metrics=['accuracy']
					)
				except Exception as e:
					logging.warning(f"Error loading model with compile=False: {e}, trying with compile=True")
					model = tf.keras.models.load_model(filename, compile=True)
				
				logging.info(f"LSTM model loaded from {filename}")
				return model, True
				
			else:
				raise ValueError(f"Invalid model type: {model_type}")
				
		except (IOError, OSError) as e:
			logging.error(f"File I/O error loading model: {e}")
			return None, False
		except Exception as e:
			logging.error(f"Error loading model: {type(e).__name__}: {e}")
			logging.error(traceback.format_exc())
			return None, False


class EmbeddingProvider:
	"""Handles word embeddings."""
	
	def __init__(self, filepath: Path, embedding_dim: int = DEFAULT_EMBEDDING_DIM):
		self.filepath = filepath
		self.embedding_dim = embedding_dim
		self.embedding_index = {}
		
	def load(self) -> bool:
		"""Load embeddings from file. Returns success status."""
		try:
			if not Path(self.filepath).exists():
				logging.error(f"Embedding file not found: {self.filepath}")
				return False
			
			with open(self.filepath, "r", encoding="utf-8") as f:
				for line in f:
					try:
						values = line.split()
						if len(values) < self.embedding_dim + 1:  # word + vector dimensions
							continue
						word = values[0]
						vector = np.asarray(values[1:], dtype="float32")
						self.embedding_index[word] = vector
					except ValueError as e:
						logging.warning(f"Skipping malformed embedding line: {e}")
						continue
			
			logging.info(f"Loaded {len(self.embedding_index)} word embeddings")
			return True
			
		except Exception as e:
			logging.error(f"Error loading embeddings: {e}")
			return False
	
	def create_matrix(self, word_index: List[str], max_features: int) -> Tuple[np.ndarray, int]:
		"""Create an embedding matrix for a given vocabulary."""
		logging.info('Creating embedding matrix from pre-trained embeddings')
		
		if not self.embedding_index:
			if not self.load():
				# Return empty matrix as fallback
				return np.zeros((max_features+1, self.embedding_dim)), max_features+1
		
		vocab_size = min(max_features, len(word_index)) + 1
		embedding_matrix = np.zeros((vocab_size, self.embedding_dim))
		
		found_words = 0
		for idx, word in enumerate(word_index):
			if idx < max_features:
				embedding_vector = self.embedding_index.get(word)
				if embedding_vector is not None:
					embedding_matrix[idx] = embedding_vector
					found_words += 1
		
		coverage = found_words / min(len(word_index), max_features)
		logging.info(f"Embedding coverage: {coverage:.2%} ({found_words}/{min(len(word_index), max_features)} words)")
		
		return embedding_matrix, vocab_size


class TopicModeler:
	"""LDA topic modeling"""
	
	def __init__(self, data: pd.Series, model_handler: ModelHandler):
		self.data = data
		self.model_handler = model_handler
		self.n_samples = len(data)
		self.n_features = len(data)
		self.n_top_words = int(len(data) * 0.05)  # ~5% of dataset size
		self.best_params = None
		self.model = None
		self.labels = None
	
	def estimate_best_params(self) -> Dict[str, Any]:
		"""Estimate the best parameters for the topic extraction model using grid search."""
		logging.info('Estimating best parameters for LDA')
		
		# Define pipeline
		pipeline = Pipeline([
			('vectorize', CountVectorizer(
				analyzer='word',
				stop_words='english',
				max_df=0.95, min_df=2
			)),
			('reduce_dim', LatentDirichletAllocation(
				batch_size=150,
				learning_method='online',
				learning_offset=15,
				random_state=0,
				evaluate_every=1,
				n_jobs=-1
			))
		])
		
		params_grid = {
			'vectorize__max_features': (int(self.n_features*0.75), self.n_features),
			'vectorize__ngram_range': ((1,1), (1,2)),
			'reduce_dim__n_components': ([6, 8, 10]),
			'reduce_dim__max_iter': (int(self.n_features*0.75), self.n_features)
		}
		
		grid = GridSearchCV(
			pipeline,
			params_grid,
			n_jobs=-1,
			cv=2,
			verbose=1
		)
		
		# Fit the model
		grid.fit(self.data)
		
		# Store best parameters and model
		self.best_params = grid.best_params_
		self.model = grid.best_estimator_
		
		logging.info(f'Best LDA model parameters: {self.best_params}')
		
		return self.best_params
	
	def get_priors(self, n_topics: int, embedding_matrix: Optional[np.ndarray] = None) -> Tuple[List, List]:
		"""Get prior distributions for topics from file."""
		prior_topic_words = []
		prior_doc_topics = []
		
		# Create document topic priors from embedding matrix if available
		if embedding_matrix is not None and len(embedding_matrix) > 0:
			for x in range(len(embedding_matrix)):
				topic_values = np.ones(n_topics)  # Initialize with ones
				prior_doc_topics.append([x, topic_values])
		
		return prior_topic_words, prior_doc_topics
	
	def factor(self) -> bool:
		"""Apply topic modeling to extract topics from text data."""
		try:
			if self.model is None:
				logging.info("No model found, estimating parameters")
				self.estimate_best_params()
			
			if self.model is None:
				raise ModelError("Failed to initialize model after parameter estimation")
			
			# Validate data
			if self.data is None or len(self.data) == 0:
				raise DataError("No data available for topic modeling")
			
			# Check for NaN values
			if isinstance(self.data, pd.Series) and self.data.isna().any():
				nan_count = self.data.isna().sum()
				logging.warning(f"Data contains {nan_count} NaN values, which will be treated as empty strings")
			
			logging.info('Fitting LDA model and predicting topic labels')
			
			# Fit the model
			self.model.fit(self.data)
			
			# Predict the labels
			try:
				if hasattr(self.model, 'predict'):
					# Pipeline object has predict method
					self.labels = self.model.predict(self.data)
				elif hasattr(self.model, 'transform'):
					# Direct LDA model uses transform
					self.labels = self.model.transform(self.data).argmax(axis=1)
				else:
					# If neither method exists, try to access the pipeline steps
					if hasattr(self.model, 'named_steps') and 'reduce_dim' in self.model.named_steps:
						vectorizer = self.model.named_steps['vectorize']
						lda = self.model.named_steps['reduce_dim']
						X_vectorized = vectorizer.transform(self.data)
						self.labels = lda.transform(X_vectorized).argmax(axis=1)
					else:
						raise ModelError("Cannot perform topic inference with this model")
			except Exception as e:
				logging.error(f"Error during topic inference: {e}")
				raise
			
			# Validate labels
			if self.labels is None or len(self.labels) == 0:
				raise ModelError("Model prediction returned empty labels")
			
			# Log the distribution of topics
			topic_counts = pd.Series(self.labels).value_counts()
			logging.info(f"Topic distribution: {dict(topic_counts)}")
			
			return True
			
		except Exception as e:
			logging.error(f"Error in factor(): {type(e).__name__}: {e}")
			logging.error(traceback.format_exc())
			return False
	
	def predict(self, data: pd.Series) -> np.ndarray:
		"""Predict topics for new data using the current model."""
		try:
			if self.model is None:
				raise ModelError("No model available for prediction")

			# Use the right method based on model type
			if hasattr(self.model, 'predict'):
				# Pipeline object has predict method
				return self.model.predict(data)
			elif hasattr(self.model, 'transform'):
				# Direct LDA model uses transform
				return self.model.transform(data).argmax(axis=1)
			else:
				# If neither method exists, try to access the pipeline steps
				if hasattr(self.model, 'named_steps') and 'reduce_dim' in self.model.named_steps:
					vectorizer = self.model.named_steps['vectorize']
					lda = self.model.named_steps['reduce_dim']
					X_vectorized = vectorizer.transform(data)
					return lda.transform(X_vectorized).argmax(axis=1)
				else:
					raise ModelError("Cannot perform topic inference with this model")
		except Exception as e:
			logging.error(f"Error during topic prediction: {type(e).__name__}: {e}")
			raise

	def get_topic_names(self) -> Dict[int, str]:
		"""Get descriptive names for topics based on top words."""
		if not hasattr(self.model, 'named_steps') or 'reduce_dim' not in self.model.named_steps:
			return {i: f"Topic {i+1}" for i in range(len(np.unique(self.labels)) if self.labels is not None else 0)}
		
		feature_names = self.model.named_steps['vectorize'].get_feature_names_out()
		lda_model = self.model.named_steps['reduce_dim']
		
		# Get top words for each topic to use as descriptive names
		topic_mapping = {}
		for topic_idx, topic in enumerate(lda_model.components_):
			top_features_ind = topic.argsort()[:-5-1:-1]  # Get top 5 words
			top_features = [feature_names[i] for i in top_features_ind]
			topic_mapping[topic_idx] = f"Topic {topic_idx+1}: {' '.join(top_features)}"
		
		return topic_mapping


class SequenceClassifier:
	"""LSTM sequence classification"""
	
	def __init__(self, text_processor: TextProcessor, model_handler: ModelHandler, 
				embedding_provider: EmbeddingProvider):
		self.text_processor = text_processor
		self.model_handler = model_handler
		self.embedding_provider = embedding_provider
		self.model = None
		self.embedding_matrix = None
		self.vocab_size = None
		self.val_data = None
	
	def fit(self, data: pd.Series, labels: np.ndarray, params: Dict[str, Any]) -> Any:
		"""Train LSTM model on extracted topics."""
		try:
			# Check prerequisites
			if params is None or not isinstance(params, dict):
				raise ModelError("Missing model parameters")
			
			if 'vectorize__max_features' not in params:
				raise ModelError("Invalid parameters: missing 'vectorize__max_features'")
			
			# Define parameters
			max_features = params['vectorize__max_features']
			
			# Get unique labels
			unique_labels = np.unique(labels)
			n_topics = len(unique_labels)
			
			if n_topics == 0:
				raise DataError("No topics found in labels")
			elif n_topics == 1:
				logging.warning("Only one topic found. Model might not train effectively.")
			
			logging.info(f"Training LSTM model with {n_topics} topics")
			
			# Ensure labels are zero-based and consecutive (0 to n_topics-1)
			label_mapping = {label: idx for idx, label in enumerate(unique_labels)}
			mapped_labels = np.array([label_mapping[label] for label in labels])
			
			# Split data into training and testing sets
			x_train, x_test, y_train, y_test = train_test_split(
				data.values, mapped_labels, test_size=0.3, random_state=1000
			)
			
			# Convert labels to categorical
			y_train_categorical = tf.keras.utils.to_categorical(y_train, num_classes=n_topics)
			y_test_categorical = tf.keras.utils.to_categorical(y_test, num_classes=n_topics)
			
			# Vectorize data
			X_train = self.text_processor.vectorize(x_train, params)
			X_test = self.text_processor.vectorize(x_test, params)
			
			# Get vocabulary
			vocab = self.text_processor.vocab
			
			# Create embedding matrix
			self.embedding_matrix, self.vocab_size = self.embedding_provider.create_matrix(
				word_index=vocab,
				max_features=max_features
			)
			
			# Create model
			max_len = X_train.shape[1] if len(X_train.shape) > 1 else DEFAULT_SEQUENCE_LENGTH
			
			# Ensure vocabulary size is valid
			if self.vocab_size <= 0:
				self.vocab_size = max(1000, max_features + 1)
				logging.warning(f"Invalid vocabulary size, using {self.vocab_size} instead")
			
			# Create model
			model_params = {
				'vocab_size': self.vocab_size,
				'embedding_dim': DEFAULT_EMBEDDING_DIM,
				'input_length': max_len,
				'n_topics': n_topics,
				'embedding_matrix': self.embedding_matrix
			}
			self.model = self.model_handler.create_model('lstm', model_params)
			
			# Model summary
			self.model.summary()
			
			# Convert inputs to tensors with type handling
			try:
				X_train_tensor = tf.convert_to_tensor(X_train)
				X_test_tensor = tf.convert_to_tensor(X_test)
				X_train_tensor = tf.cast(X_train_tensor, tf.int32)
				X_test_tensor = tf.cast(X_test_tensor, tf.int32)
			
			except tf.errors.InvalidArgumentError as e:
				logging.warning(f"Tensor conversion error: {e}, trying alternative approach")
				X_train_numpy = np.array(X_train.numpy() if hasattr(X_train, 'numpy') else X_train)
				X_test_numpy = np.array(X_test.numpy() if hasattr(X_test, 'numpy') else X_test)
				X_train_tensor = tf.cast(tf.convert_to_tensor(X_train_numpy), tf.int32)
				X_test_tensor = tf.cast(tf.convert_to_tensor(X_test_numpy), tf.int32)
			
			# Store validation data for later evaluation
			self.val_data = {
				'X': X_test_tensor,
				'y': y_test_categorical
			}

			# Fit model
			epochs = 10
			steps_per_epoch = 5
			batch_size = 5
			
			try:
				history = self.model.fit(
					X_train_tensor,
					y_train_categorical,
					steps_per_epoch=steps_per_epoch,
					epochs=epochs,
					verbose=True,
					validation_data=(X_test_tensor, y_test_categorical),
					batch_size=batch_size
				)
			except tf.errors.ResourceExhaustedError:
				logging.warning("Out of memory during training, reducing batch size")
				batch_size = max(1, batch_size // 2)
				history = self.model.fit(
					X_train_tensor,
					y_train_categorical,
					steps_per_epoch=steps_per_epoch,
					epochs=epochs,
					verbose=True,
					validation_data=(X_test_tensor, y_test_categorical),
					batch_size=batch_size
				)
			
			logging.info("LSTM model training completed successfully")
			return history
			
		except Exception as e:
			logging.error(f"Error in fit(): {type(e).__name__}: {e}")
			logging.error(traceback.format_exc())
			raise
		finally:
			# Clean up
			clear_session()
			gc.collect()
	
	def predict(self, data: pd.DataFrame) -> np.ndarray:
		"""Predict topics for new data using the trained LSTM model."""
		try:
			# Check if model is trained
			if self.model is None:
				raise ModelError("LSTM model not loaded or trained yet")
			
			# Extract the script_dialogue column
			if 'script_dialogue' not in data.columns:
				raise DataError("Input data missing 'script_dialogue' column")
			
			text_data = data['script_dialogue']
			
			# Check for empty data
			if len(text_data) == 0:
				logging.warning("Empty input data - nothing to predict")
				return np.array([])
			
			# Vectorize the data
			X_new = self.text_processor.vectorize(text_data)
			
			# Convert to tensor with type handling
			try:
				X_new_tensor = tf.convert_to_tensor(X_new)
				X_new_tensor = tf.cast(X_new_tensor, tf.int32)

			except tf.errors.InvalidArgumentError as e:
				logging.warning(f"Tensor conversion error: {e}, trying alternative approach")
				X_new_numpy = np.array(X_new.numpy() if hasattr(X_new, 'numpy') else X_new)
				X_new_tensor = tf.cast(tf.convert_to_tensor(X_new_numpy), tf.int32)
			
			# Predict with batching for large datasets
			if len(text_data) > 500:
				batch_size = 64
				predictions = []
				for i in range(0, len(X_new_tensor), batch_size):
					batch = X_new_tensor[i:i+batch_size]
					pred_batch = self.model.predict(batch, verbose=0)
					predictions.append(pred_batch)
				y_pred = np.vstack(predictions)
			else:
				y_pred = self.model.predict(X_new_tensor, verbose=1)
			
			return y_pred
			
		except tf.errors.ResourceExhaustedError:
			logging.error("GPU memory exceeded - trying with smaller batches")
			# Fallback with smaller batches
			batch_size = 16
			predictions = []
			for i in range(0, len(X_new_tensor), batch_size):
				batch = X_new_tensor[i:i+batch_size]
				pred_batch = self.model.predict(batch, verbose=0)
				predictions.append(pred_batch)
			return np.vstack(predictions)
		except Exception as e:
			logging.error(f"Error during prediction: {type(e).__name__}: {e}")
			logging.error(traceback.format_exc())
			raise

	def evaluate(self, data: Optional[pd.Series] = None, 
				labels: Optional[np.ndarray] = None) -> Tuple[float, float]:
		"""Evaluate the LSTM model and return metrics (loss, accuracy).
		If data and labels are not provided, uses validation data from training.
		"""
		try:
			# Check if model exists
			if self.model is None:
				raise ModelError("No model available for evaluation")
			
			# If no explicit test data is provided, check for stored validation data
			if (data is None or labels is None):
				if hasattr(self, 'val_data') and self.val_data is not None:
					X_test = self.val_data['X']
					y_test = self.val_data['y']
					logging.info("Using stored validation data for evaluation")
				else:
					logging.warning("No validation data available for evaluation")
					
					# If we have topic labels from the training data, use a subset for evaluation
					if hasattr(self, 'topic_modeler') and hasattr(self.topic_modeler, 'labels') and self.topic_modeler.labels is not None:
						# Sample a small portion for evaluation
						sample_size = min(100, len(self.topic_modeler.labels))
						indices = np.random.choice(len(self.topic_modeler.labels), sample_size, replace=False)
						return self.evaluate(
							data=self.topic_modeler.data.iloc[indices],
							labels=self.topic_modeler.labels[indices]
						)
					else:
						logging.error("Cannot evaluate model without validation data or test data")
						return float('inf'), 0.0  
			elif data is not None and labels is not None:
				logging.info("Processing provided test data for evaluation")
				
				# Get unique labels and count of topics
				unique_labels = np.unique(labels)
				n_topics = len(unique_labels)
				
				# Ensure labels are zero-based
				label_mapping = {label: idx for idx, label in enumerate(unique_labels)}
				mapped_labels = np.array([label_mapping[label] for label in labels])
				
				# Convert to categorical
				y_test_categorical = tf.keras.utils.to_categorical(mapped_labels, num_classes=n_topics)
				
				# Vectorize text data
				X_test_vectorized = self.text_processor.vectorize(data)
				
				# Convert to proper tensor types
				try:
					X_test = tf.convert_to_tensor(X_test_vectorized)
					X_test = tf.cast(X_test, tf.int32)
					y_test = y_test_categorical

				except tf.errors.InvalidArgumentError as e:
					logging.warning(f"Tensor conversion error: {e}, trying alternative approach")
					X_test_numpy = np.array(X_test_vectorized.numpy() 
										if hasattr(X_test_vectorized, 'numpy') 
										else X_test_vectorized)
					X_test = tf.cast(tf.convert_to_tensor(X_test_numpy), tf.int32)
					y_test = y_test_categorical
			else:
				raise ValueError("Must provide both data and labels for evaluation, or train with validation data")
			
			# Evaluate model
			logging.info("Evaluating LSTM model")
			metrics = self.model.evaluate(X_test, y_test, verbose=1)
			
			# Get loss and accuracy from metrics
			if isinstance(metrics, list) and len(metrics) >= 2:
				loss, accuracy = metrics[0], metrics[1]
			else:
				loss, accuracy = metrics, 0.0
				
			return loss, accuracy
			
		except Exception as e:
			logging.error(f"Error during model evaluation: {type(e).__name__}: {e}")
			logging.error(traceback.format_exc())
			return float('inf'), 0.0




def main() -> None:
	"""Main execution function for the topic extraction pipeline."""
	# Set random seed for reproducibility
	np.random.seed(19680801)
	tf.random.set_seed(19680801)
	
	try:
		logging.info('Starting topic extraction pipeline')
		
		# Create necessary directories
		output_dir = CONFIG['OUTPUT_DIR']
		if not output_dir.exists():
			output_dir.mkdir(parents=True, exist_ok=True)
			logging.info(f"Created output directory: {output_dir}")
		
		# Initialize components with dependency injection
		text_processor = TextProcessor(CONFIG)
		model_handler = ModelHandler(CONFIG)
		embedding_provider = EmbeddingProvider(CONFIG['GLOVE'])
		
		# Load source data
		try:
			df = text_processor.load_source(CONFIG['SOURCE'])
			logging.info(f"Loaded {len(df)} dialogue samples from {CONFIG['SOURCE']}")
		except DataError as e:
			logging.error(f"Failed to load source data: {e}")
			return
		
		# Extract the dialogue series for modeling
		text_series = df['script_dialogue']
		
		# Create topic modeler
		topic_modeler = TopicModeler(text_series, model_handler)
		
		# Try to load existing LDA model first
		loaded_model, success = model_handler.load_model('lda')
		if success:
			logging.info("Using existing LDA model")
			topic_modeler.model = loaded_model
			topic_modeler.best_params = loaded_model.get_params()
			
			# Get predictions using loaded model
			try:
				# Use the dedicated predict method instead
				topic_modeler.labels = topic_modeler.predict(text_series)
				topic_counts = pd.Series(topic_modeler.labels).value_counts()
				logging.info(f"Topic distribution: {dict(topic_counts)}")
			except Exception as e:
				logging.error(f"Error predicting with loaded model: {e}")
				success = False
		
		# If model loading failed or prediction failed, run LDA modeling
		if not success:
			logging.info("Running new LDA topic modeling")
			if not topic_modeler.factor():
				logging.error("Topic modeling failed, exiting")
				return
			
			# Save the model
			model_handler.save_model(topic_modeler.model, 'lda')
		
		# Get descriptive names for topics
		topic_names = topic_modeler.get_topic_names()
		logging.info(f"Identified topics: {list(topic_names.values())}")
		
		# Initialize sequence classifier with dependencies
		sequence_classifier = SequenceClassifier(
			text_processor=text_processor,
			model_handler=model_handler,
			embedding_provider=embedding_provider
		)
		
		# Try to load existing LSTM model
		lstm_model, lstm_success = model_handler.load_model('lstm')
		if lstm_success:
			logging.info("Using existing LSTM model")
			sequence_classifier.model = lstm_model
		else:
			# Train LSTM classifier
			logging.info("Training LSTM classifier")
			try:
				sequence_classifier.fit(
					data=text_series,
					labels=topic_modeler.labels,
					params=topic_modeler.best_params
				)
				
				# Save trained model
				model_handler.save_model(sequence_classifier.model, 'lstm')
			except Exception as e:
				logging.error(f"Error training LSTM model: {e}")
				logging.error(traceback.format_exc())
				return
		
		# Make predictions with the trained model
		try:
			predictions = sequence_classifier.predict(df)
			
			# Save results with topic names
			text_processor.save_results(
				y_pred=predictions,
				original_data=df,
				topic_mapping=topic_names
			)
			
			logging.info("Topic extraction pipeline completed successfully")

		except Exception as e:
			logging.error(f"Error in prediction/results step: {e}")
			logging.error(traceback.format_exc())
		
		# Report metrics
		try:
			if not hasattr(sequence_classifier, 'val_data') or sequence_classifier.val_data is None:
				loss, accuracy = sequence_classifier.evaluate(
					data=text_series.sample(min(100, len(text_series))),
					labels=topic_modeler.labels[:min(100, len(topic_modeler.labels))] if topic_modeler.labels is not None else None
				)
			else:
				loss, accuracy = sequence_classifier.evaluate()

			if accuracy > 0 or loss < float('inf'):
				logging.info(f"Model evaluation - Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")
			else:
				logging.warning("Model evaluation returned invalid metrics")

		except Exception as eval_error:
			logging.warning(f"Could not evaluate model: {eval_error}")

	
	except Exception as e:
		logging.error(f"Unexpected error in main: {e}")
		logging.error(traceback.format_exc())
	finally:
		# Clean up resources
		logging.info("Cleaning up resources")
		try:
			tf.keras.backend.clear_session()
			gc.collect()
		except Exception as e:
			logging.error(f"Error during cleanup: {e}")


if __name__ == "__main__":
	main()