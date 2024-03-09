# Access Point Search

Combining CMI with EVSI for anomaly detection, we use the Access Point Search Algorithm (APSA) to select the optimal set of Access Points.

## Installation
To run the code, it is recommended to install the package and related dependencies into the current Python environment.

```
pip install -r requirements.txt
```

## Usage
1. The ```dime/``` directory contains our method to dynamically select features by estimating the CMI in a discriminative fashion. In the directory, the ```cmi_estimator.py``` file is the main file for the implementation of our method. Please refer to this file for details of joint training.
2. The ```baselines/``` directory contains all the basic methods needed, such as EVSI and greedy CMI. The ```sample_dataset.ipynb``` file is the sampling process according to the Tennessee Eastman dataset.
3. The ```CMI_EVSI.ipynb``` file contains the main code of my research. Focus on the APSA part where CMIEstimator is used.

The program will run CMI-EVSI and will provide the optimal set of Access Points that can be used to detect anomalies in the system and also calculate the cost incurred in selecting the access points.

More technical details about the structure of the code can be found in the [Docs](Docs) folder.
