This repository contains the implementation of the LIKLE method, which combines normalizing flow-based likelihood-free inference with Karhunen–Loève expansion (KLE) for dimensionality reduction.
LIKLE is designed to estimate 2,121 parameters from sparse measurements in the Freyberg 3D hydrology problem.
The file inference.py implements pure flow-based likelihood-free inference without KLE.
The file inference_KLE.py incorporates KLE for dimensionality reduction.
The codes for normalizing flow part are built upon the BayesFlow library.
