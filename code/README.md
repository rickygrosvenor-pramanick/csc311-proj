# code for Part B

## Install dependencies
Install dependencies with the following command.
```
pip install -r requirements.txt
```

## Training the model
Run the following command to train the model with the default parameters. There were the parameters to obtain the reported metrics.
```
python train.py \
    --dataset_path ../data \
    --checkpoint_path checkpoints \
    --logs_path logs \
    --epochs 200 \
    --student_embed_dim 8 \
    --question_embed_dim 16 \
    --hidden_layers 64 16 \
    --dropout_p 0.3 \
    --batch_size 32 \
    --learning_rate 0.001 \
    --device cpu
```

## Evaluating the model
Run the following command to evaluate the model. The below code uses experiment3, which was the experiment used to obtain the reported metrics.
```
python eval.py \
    --checkpoint_folder checkpoints/experiment3 \
    --eval_dataset ../data/test_data.csv \
    --student_meta ../data/student_meta.csv
```

## Visualizing loss and accuracy
Run the following command to visualize accuracy and loss curves as training progresses.
```
tensorboard --logdir=logs
```