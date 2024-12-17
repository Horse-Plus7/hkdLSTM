# hkdLSTM

This project demonstrates how to use Long Short-Term Memory (LSTM) networks to predict future values in a time series dataset. The dataset is split into training and testing sets, and LSTM is used to build a predictive model that forecasts future values based on previous time steps.

Requirements

Make sure you have the following libraries installed to run this project:
	•	Python 3.x
	•	numpy
	•	pandas
	•	scikit-learn
	•	keras
	•	tensorflow

You can install the necessary libraries via pip by running:

pip install numpy pandas scikit-learn keras tensorflow

Project Structure

The project consists of a single script file lstm.py which contains the following main sections:
	1.	Data Preparation:
	•	The raw time series data is preprocessed by normalizing the values using MinMaxScaler.
	•	The data is split into training and testing sets, and the training data is used to create sequences of data to be input into the LSTM model.
	2.	Model Construction:
	•	An LSTM model is constructed using Keras with the following layers:
	•	LSTM layer with 50 units and relu activation.
	•	A Dense layer for output prediction.
	•	The model is compiled using the Adam optimizer and mean squared error as the loss function.
	3.	Training the Model:
	•	The model is trained for 50 epochs with a batch size of 32.
	4.	Prediction:
	•	After training, the model is used to make predictions on the test data.
	•	Results are then inverse-transformed back to the original scale of the data.

How to Use
	1.	Prepare your Data:
	•	Replace the placeholder data (train_data and test_data in the code) with your own time series data. The data should be a 2D array where each entry is a single value in the time series.
	2.	Run the Script:
	•	Execute the script to train the model and make predictions:

python lstm.py


	3.	Review the Results:
	•	After running the script, the predicted values (y_pred) and the actual test values (y_test) will be printed.
	4.	Modify Parameters:
	•	You can adjust the look_back variable to change the number of previous time steps used for predictions.
	•	The epochs and batch_size can also be modified to control the training process.

  	5.	Dataset
   	•	hkd_exchange_rate.csv is a test dataset we generated. Users can also use GET_CSV.py to generate new datasets.

Key Features
	•	Data Normalization: The data is normalized using MinMaxScaler to scale the values between 0 and 1, improving the performance of the LSTM model.
	•	Train-Test Split: The data is split into training and testing datasets. The first 80% of the data is used for training, and the remaining 20% is used for testing.
	•	LSTM Model: The LSTM model is designed to predict future values in a time series using the past 60 time steps as input.
	•	Prediction and Evaluation: The model predicts future values and compares them to the actual values in the test set.

Example Output

After running the script, you should see output similar to the following:

Train data size: (148, 60), Test data size: (53, 60)
X_train shape after reshaping: (148, 60, 1)
X_test shape after reshaping: (53, 60, 1)

Epoch 1/50
5/5 ━━━━━━━━━━━━━━━━━━━━ 1s 15ms/step - loss: 0.5357
...
Epoch 50/50
5/5 ━━━━━━━━━━━━━━━━━━━━ 0s 15ms/step - loss: 0.0081

Predicted values:  [[0.5025], [0.5074], [0.5081], ...]
Actual values:    [[0.5000], [0.5100], [0.5200], ...]

Troubleshooting
	•	Empty Test Set Error: If you receive an error about an empty test set, check that the dataset is being split correctly and that there are enough samples in both the training and testing sets.
	•	Model Training Warnings: If you encounter warnings like “Local rendezvous is aborting,” ensure that your training set is not too small and adjust the batch size or epochs accordingly.
	•	Shape Mismatch: If the data reshaping causes issues, make sure your input data is in the correct format (a 2D array with shape (n_samples, 1) for the time series).
