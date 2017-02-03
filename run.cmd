python 1_computeRois.py # create green squares everywhere 
python 2_cntkGenerateInputs.py # map them to labels 
python 3_runCntk.py # train model takes forever
# we don't train the svm because we have a network with a predition layer hat
python 5_visualizeResults.py # build neat squares around things
explorer ./results/