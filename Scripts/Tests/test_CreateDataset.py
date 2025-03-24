import unittest
import unittest.test
import unittest.mock

from pathlib import Path

from CreateDataset import *


# Test discovery: ROOT_DIR> python -m unittest discover -s tests
# Else: ROOT_DIR> python -m unittest tests.test_CreateDataset

class TestToolbox(unittest.TestCase):
	
	def test_staging_tmpDir_create(self):
		staging_dir = Toolbox.staging_tmpDir_create()
		self.assertTrue(staging_dir.exists())
		Toolbox.staging_tmpDir_clear(staging_dir)

	def test_numbered_file(self):
		path = Toolbox.numbered_file('xlsx')
		self.assertTrue(path.name.startswith('_('))
		self.assertTrue(path.name.endswith(').xlsx'))

	def test_jaccard_similarity(self):
		# Test identical strings
		self.assertEqual(Toolbox.jaccard_similarity("hello world", "hello world"), 1.0)
		
		# Test completely different strings
		self.assertEqual(Toolbox.jaccard_similarity("hello world", "goodbye universe"), 0.0)
		
		# Test partially matching strings
		self.assertAlmostEqual(
			Toolbox.jaccard_similarity("hello world", "hello there"), 
			0.333, 
			places=3
		)
		
		# Test with empty strings
		self.assertEqual(Toolbox.jaccard_similarity("", ""), 0.0)
		
		# Test with None values
		self.assertEqual(Toolbox.jaccard_similarity(None, "hello"), 0.0)
		self.assertEqual(Toolbox.jaccard_similarity("hello", None), 0.0)
		self.assertEqual(Toolbox.jaccard_similarity(None, None), 0.0)


class TestProcessData(unittest.TestCase):
	def setUp(self):
		self.test_dir = Path(tempfile.mkdtemp())
		self.srt_content = "00:01:02\nTest dialogue line 1\n\n00:02:03\nTest dialogue line 2"
		self.script_content = "CHARACTER 1:\nTest script line 1\n\nCHARACTER 2:\nTest script line 2"
		
		# Create test files
		with open(self.test_dir / "_srt.txt", "w") as f:
			f.write(self.srt_content)
		with open(self.test_dir / "_script.txt", "w") as f:
			f.write(self.script_content)
			
		self.processor = ProcessData(self.test_dir)
	
	def tearDown(self):
		shutil.rmtree(self.test_dir)
	
	def test_init(self):
		self.assertEqual(self.processor.srt, self.test_dir / "_srt.txt")
		self.assertEqual(self.processor.script, self.test_dir / "_script.txt")
		self.assertIsNone(self.processor.result)
	
	def test_extract_srt(self):
		df = self.processor.extract_srt()
		self.assertIsInstance(df, pd.DataFrame)
		self.assertIn('timecode', df.columns)
		self.assertIn('dialogue', df.columns)
	
	def test_extract_script(self):
		df = self.processor.extract_script()
		self.assertIsInstance(df, pd.DataFrame)
		self.assertIn('part', df.columns)
		self.assertIn('dialogue', df.columns)



if __name__ == "__main__":
	unittest.main()
	