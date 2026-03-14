"""
Simple test script to verify the EXTRACT stage works correctly.
"""

from etl.extract import extract_data


def test_extract_pipeline():
	print("Running EXTRACT pipeline test...\n")

	df = extract_data()

	print("\nDataset successfully extracted!")
	print("Rows:", df.shape[0])
	print("Columns:", df.shape[1])

	print("\nColumn Names:")
	print(df.columns.tolist())

	print("\nPreview:")
	print(df.head())


if __name__ == "__main__":
	test_extract_pipeline()