## Installation

Before running the code, make sure you have the following dependencies installed:

- Python >= 3.6
- NumPy
- Pandas
- OpenCV (cv2)
- PyTorch
- Blender (if needed)

You can install these dependencies using the following command:

```bash Copy code
pip install numpy pandas opencv-python torch
```
## Instructions for Data Generation and Model Training
### Generate Data:

First, generate the training and testing data by running the gen_data.py script.

Open a terminal or command prompt, navigate to the project directory, and execute the following command:

```bash Copy code
python gen_data.py
```
This will create two CSV files, train_data.csv and test_data.csv, containing the generated data.

### Train the Model:

Once the data is generated, you can proceed to train the model using the train.py script.

Run the following command in the terminal:

```bash Copy code
python train.py
```
This will train the model using the generated training data and save the trained model as model_all_7_layer.pth in the models directory.

Start Predictions:

After the training is complete, you can use the trained model for pitching tracking. It would generate a trajectory video and blender scene.  
```bash Copy code
python main.py
```

## Input
https://github.com/spen2005/pitching_location_predictor/assets/126836958/4544d3ef-6439-4f54-a5ac-99cf51d7b26f
## Result
![image](https://github.com/spen2005/pitching_location_predictor/assets/126836958/eca8d967-b134-48e1-ae3d-eb9c364e1903)




