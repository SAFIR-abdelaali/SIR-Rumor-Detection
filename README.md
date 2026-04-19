PFA Rumor Detection System (SIR + ML) on PHEME
==============================================

Overview
--------
This PFA project builds a rumor detection system for Twitter that combines a machine
learning classifier with an SIR (Susceptible-Infected-Recovered) simulation to estimate
spread impact. The Streamlit app provides an early warning interface for single tweets
and CSV batches using the PHEME dataset.

Key features
------------
- Classify tweets as rumor or non-rumor using exported ML artifacts.
- Simulate rumor spread with an SIR model when a rumor is detected.
- Batch analysis for CSV files with tweet metadata.
- Synthetic data generator for quick testing.

How the SIR part is used
------------------------
The app converts engagement signals (retweets, doubt score, etc.) into SIR parameters
(beta and gamma) to simulate spread over 48 hours and estimate peak impact.

Repository layout
-----------------
app.py
	Streamlit dashboard (manual analysis + CSV batch analysis + SIR simulation).
models/
	Exported model artifacts (ignored by git).
data/
	PHEME dataset and generated CSV files (ignored by git).
scripts/
	Helper scripts (synthetic data generator).
notebooks/
	Exploration notebooks.

Quick start
-----------
1. Create and activate a Python environment.
2. Install dependencies from requirements.txt:

	pip install -r requirements.txt

3. Ensure the exported model files exist in models/:
   - model_rumor_detector.pkl
   - tfidf_vectorizer.pkl
   - feature_scaler.pkl

4. Run the app:

	streamlit run app.py

Synthetic test data
-------------------
Generate a CSV with extreme examples for quick testing:

	python scripts/test_gen.py

The file is written to data/synthetic_test_data_extreme.csv.

Dataset details
---------------
The PHEME dataset includes five events:
- Charlie Hebdo: 458 rumors (22.0%) and 1,621 non-rumors (78.0%).
- Ferguson: 284 rumors (24.8%) and 859 non-rumors (75.2%).
- Germanwings Crash: 238 rumors (50.7%) and 231 non-rumors (49.3%).
- Ottawa Shooting: 470 rumors (52.8%) and 420 non-rumors (47.2%).
- Sydney Siege: 522 rumors (42.8%) and 699 non-rumors (57.2%).

Each event directory contains two subfolders, rumours and non-rumours, with tweet ID
folders. The source tweet is under source-tweet and replies are under reactions.

License
-------
The annotations are provided under a CC-BY license. Twitter retains ownership and rights of
tweet content.
