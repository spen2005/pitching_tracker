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
cd src
python gen_data.py
```
This will create two CSV files, train_data.csv and test_data.csv, containing the generated data.

### Train the Model:

Once the data is generated, you can proceed to train the model using the train.py script.

Run the following command in the terminal:

```bash Copy code
cd src
python train.py
```
This will train the model using the generated training data and save the trained model as model_all_7_layer.pth in the models directory.

Start Predictions:

After the training is complete, you can use the trained model for pitching tracking. It would generate a trajectory video and blender scene.  
```bash Copy code
cd src
python main.py
```

## Input
https://github.com/spen2005/pitching_tracker/assets/126836958/a7dcd524-76d4-4256-a1bb-1be58ed49ff7

## My Result
https://github.com/spen2005/pitching_tracker/assets/126836958/1ae1ccd4-e4aa-4561-a6a1-8643a6d92155

<img width="645" alt="Screen Shot 2024-05-25 at 10 40 49 PM" src="https://github.com/spen2005/pitching_tracker/assets/126836958/c364be9d-c09b-4f56-bdb1-88ee1b68cf5e">

## MLB Trackman System
<img width="822" alt="Screen Shot 2024-05-25 at 10 41 29 PM" src="https://github.com/spen2005/pitching_tracker/assets/126836958/c4f31f2d-9e21-4982-882e-1f2abc2ca71a">


