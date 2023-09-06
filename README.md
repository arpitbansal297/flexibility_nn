# Flexibility Neural Networks (flexibility_nn)

## ImageNet Training
To train a neural network model on the ImageNet dataset, run the following command:

```bash
python main.py \
  --batch_size=128 \
  --optimizer=adamw \
  --lr=0.0002 \
  --total_steps=1000000 \
  --architecture=resnet9 \
  --start_num_example=115000 \
  --end_num_example=250000 \
  --dataset=imagenet \
  --data_dir=/state/partition1/imagenet/imagenet \
  --num_to_print=10000 \
  --workers=20
```

### Parameters:
- `--batch_size`: Size of the training batch.
- `--optimizer`: Optimization algorithm to use (`adamw` in this case).
- `--lr`: Learning rate for the optimizer.
- `--total_steps`: Total number of training steps.
- `--architecture`: Neural network architecture to use (`resnet9` in this case).
- `--start_num_example`: Minimum number of samples to train with.
- `--end_num_example`: Maximum number of samples to train with.
- `--dataset`: The dataset to use (`imagenet` in this case).
- `--data_dir`: Directory containing the dataset.
- `--num_to_print`: Number of training examples to print.
- `--workers`: Number of worker threads for data loading.

### Optional Flags:
- `random_labels`: If you want to train with random labels, add the `--random_labels` flag to the command.
- `random_input`: If you want to train with random (Gaussian) input, add the `--random_input` flag to the command.

Note that the paths and parameters are set as examples and should be configured according to your specific requirements.
