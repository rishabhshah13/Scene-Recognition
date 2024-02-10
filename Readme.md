# Brainstorm
https://docs.google.com/document/d/12n1ngQg6Nj7vv5hHjfODlDoo-6Alx9YLy7klJtDVxzg/edit

# Data
download from: https://drive.google.com/file/d/112pPeJoWmyWjEvpB-AoDWpGp7UGS3QQf/view?usp=sharing

# Models
All models should be defined in the models directory. Then in main.py, import the model instatiator and add it to the `models` dictionary. 

# Main loop
Give your run a name, select a model to train, and configure under the `RUN DETAILS` comment.
Feel free to add in other metrics you want to track. 

# Sample Run
python main.py --model_base resnet18 --num_epochs  1 --batch_size  32 --learning_rate  0.001 --random_seed  42 --use_split --save_checkpoints  1