split-images:
	@python classificacao\dataSplitter\data_splitting.py
extract: 
	@python classificacao\featureExtractors\orb_FeatureExtraction.py
classify:
	@python classificacao\classifiers\svm_classifier.py