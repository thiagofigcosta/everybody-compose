# Everybody Compose: Deep Beats To Music 
Authors: Conghao (Tom) Shen, Violet Yao, Yixin Liu

## Abstract

This project presents a deep learning approach to generate monophonic melodies based on input beats, allowing even amateurs to create their own music compositions. Three effective methods - LSTM with Full Attention, LSTM with Local Attention, and Transformer with Relative Position Representation - are proposed for this novel task, providing great variation, harmony, and structure in the generated music. This project allows anyone to compose their own music by tapping their keyboards or ``recoloring'' beat sequences from existing works.

## Getting Started

To get started, clone this repository and install the required packages:
```sh
git clone https://github.com/tsunrise/everybody-compose.git
cd everybody-compose
pip install -r requirements.txt
```

Install ffmpeg:
- Ubuntu: `sudo apt install ffmpeg`
- Mac: `brew install ffmpeg`
- Windows: [follow instructions](http://blog.gregzaal.com/how-to-install-ffmpeg-on-windows/)

You may encouter dependency issues during training on `protobuf`. If so, try reinstall `tensorboard` by running:
```sh
pip install --upgrade tensorboard
```

And also upgrade `pip` by running:

```sh
pip install --upgrade pip
```

This issue is due to an conflicting requirements of `note_seq` and `tensorboard`.

We have also provided a [Colab Notebook](https://colab.research.google.com/drive/1oVn-lZI1K23EC9py6UibDOL7swQGp4v9?usp=sharing#scrollTo=kp6HIjuYvoye) for your reference.

## Training
The preprocessed dataset will automatically be downloaded before training. To train a model, run the `train.py` script with the `-m` or `--model_name` argument followed by a string specifying the name of the model to use. The available model names are:

- `lstm_attn`: LSTM with Local Attention
- `vanilla_rnn`: Decoder Only Vanilla RNN
- `attention_rnn`: LSTM with Full Attention
- `transformer`: Transformer RPR

You have to also specify which genre the model will be trained with the `-g` or `--genre` argument. The available genres are:

- `all_midis`: All genres
- `classical`: Classical songs
- `folk`: Folk songs
- `pop_rock`: Classical Pop and Rock songs
- `soul_reggae`: Soul and Reggae Songs 

You can also use the `-nf` or `--n_files` argument followed by an integer to specify the number of files to use for training (the default value of -1 means that all available files will be used).

To specify the number of epochs to train the model for, use the `-n` or `--n_epochs` argument followed by an integer. The default value is 100.

To specify the device to use for training, use the `-d` or `--device` argument followed by a string. The default value is cuda if a CUDA-enabled GPU is available, or cpu if not.

To specify the frequency at which to save snapshots of the trained model, use the `-s` or `--snapshots_freq` argument followed by an integer. This specifies the number of epochs between each saved snapshot. The default value is 200. The snapshots will be saved in the `project_data/snapshots` directory. The default value is 200.

To specify a checkpoint to load the model from, use the `-c` or `--checkpoint` argument followed by a string specifying the path to the checkpoint file. The default value is None, which means that no checkpoint will be loaded.

Here are some examples of how to use these arguments:

```sh
# Train the LSTM with Local Attention model using all available files of Classical songs, for 100 epochs, on the default device, saving snapshots every 200 epochs, and not using a checkpoint
python train.py -m lstm_attn -g classical

# Train the LSTM with Local Attention model using 10 files of Folk songs, for 400 epochs, on the CPU, saving snapshots every 100 epochs, and starting from the checkpoint
python train.py -m lstm_attn -g folk -nf 10 -n 400 -d cpu -s 100 -c ./project_data/snapshots/my_checkpoint.pth

# Train the Transformer RPR model using all available files, for 500 epochs, on the default device, saving snapshots every 50 epochs, and not using a checkpoint
python train.py -m transformer -n 500 -s 50
```

## Generating Melodies from Beats

To generate a predicted notes sequence and save it as a MIDI file, run the `predict_stream.py` script with the `-m` or `--model_name` argument followed by a string specifying the name of the model to use. The available model names are:

- `lstm_attn`: LSTM with Local Attention
- `vanilla_rnn`: Decoder Only Vanilla RNN
- `attention_rnn`: LSTM with Full Attention
- `transformer`: Transformer RPR

Use the `-c` or `--checkpoint_path` argument followed by a string 
specifying the path to the checkpoint file to use for the model.

The generated MIDI file will be saved using the filename specified by the `-o` or `--midi_filename` argument (the default value is `output.mid`).

To specify the device to use for generating the predicted sequence, use the `-d` or `--device` argument followed by a string. The default value is `cuda` if a CUDA-enabled GPU is available, or `cpu` if not.

To specify the source of the input beats, use the `-s` or `--source` argument followed by a string. The default value is `interactive`, which means that the user will be prompted to input the beats using the keyboard. Other possible values are:

- A file path, e.g. `beat_sequence.npy`, to load the recorded beats from a file. Recorded beats can be generated using the `create_beats.py` script.
- `dataset` to use a random sample from the dataset as the beats.

To specify the profile to use for generating the predicted sequence, use the `-t` or `--profile` argument followed by a string. The available values are `beta`, which uses stochastic search, or `beam`, which uses hybrid beam search. The heuristic parameters for these profiles can be customized in the config.toml file by adjusting the corresponding sections in `[sampling.beta]` and `[sampling.beam]`. The default value is default, which uses the settings specified in the `config.toml` file.

Here are some examples of how to use these arguments:

```sh
# Generate a predicted sequence using the best performin LSTM with Local Attention model of classical songs, from beats by the user using the keyboard, using the checkpoint at ./project_data/snapshots/my_checkpoint.pth, on the default device, and using the beta profile with default settings
python predict_stream.py -m lstm_attn -c ./project_data/snapshots/lstm_attn_all_classical_best.pth -t beta
```

# Running AiCaThi experiments:

1. Train models with `classical` genre
```shell
python3 train.py -m lstm_attn -g classical -n 100
python3 train.py -m vanilla_rnn -g classical -n 200
python3 train.py -m attention_rnn -g classical -n 150
python3 train.py -m transformer -g classical -n 200

python3 gan_training.py  -gm gan_gen -dm gan_disc -g classical -n 400 # WiP
```

2. Evaluate metrics for all genres trained with `classical` genre
```shell
python3 train.py -m lstm_attn -g classical -c aicathi_models/lstm_attn_all_classical_best.pth --test_only
python3 train.py -m lstm_attn -g folk -c aicathi_models/lstm_attn_all_classical_best.pth --test_only
python3 train.py -m lstm_attn -g pop_rock -c aicathi_models/lstm_attn_all_classical_best.pth --test_only
python3 train.py -m lstm_attn -g soul_reggae -c aicathi_models/lstm_attn_all_classical_best.pth --test_only


python3 train.py -m vanilla_rnn -g classical -c aicathi_models/vanilla_rnn_all_classical_best.pth --test_only
python3 train.py -m vanilla_rnn -g folk -c aicathi_models/vanilla_rnn_all_classical_best.pth --test_only
python3 train.py -m vanilla_rnn -g pop_rock -c aicathi_models/vanilla_rnn_all_classical_best.pth --test_only
python3 train.py -m vanilla_rnn -g soul_reggae -c aicathi_models/vanilla_rnn_all_classical_best.pth --test_only


python3 train.py -m attention_rnn -g classical -c aicathi_models/attention_rnn_all_classical_best.pth --test_only
python3 train.py -m attention_rnn -g folk -c aicathi_models/attention_rnn_all_classical_best.pth --test_only
python3 train.py -m attention_rnn -g pop_rock -c aicathi_models/attention_rnn_all_classical_best.pth --test_only
python3 train.py -m attention_rnn -g soul_reggae -c aicathi_models/attention_rnn_all_classical_best.pth --test_only


python3 train.py -m transformer -g classical -c aicathi_models/transformer_all_classical_best.pth --test_only
python3 train.py -m transformer -g folk -c aicathi_models/transformer_all_classical_best.pth --test_only
python3 train.py -m transformer -g pop_rock -c aicathi_models/transformer_all_classical_best.pth --test_only
python3 train.py -m transformer -g soul_reggae -c aicathi_models/transformer_all_classical_best.pth --test_only



python3 gan_training.py  -gm gan_gen -dm gan_disc  -g classical -gc aicathi_models/gan-G-gan_gen_all_classical_best.pth -dc aicathi_models/gan-D-gan_disc_all_classical_best.pth --test_only
python3 gan_training.py  -gm gan_gen -dm gan_disc  -g folk -gc aicathi_models/gan-G-gan_gen_all_classical_best.pth -dc aicathi_models/gan-D-gan_disc_all_classical_best.pth --test_only
python3 gan_training.py  -gm gan_gen -dm gan_disc  -g pop_rock -gc aicathi_models/gan-G-gan_gen_all_classical_best.pth -dc aicathi_models/gan-D-gan_disc_all_classical_best.pth --test_only
python3 gan_training.py  -gm gan_gen -dm gan_disc  -g soul_reggae -gc aicathi_models/gan-G-gan_gen_all_classical_best.pth -dc aicathi_models/gan-D-gan_disc_all_classical_best.pth --test_only
```

3. Train models with all genres
```shell
python3 train.py -m lstm_attn -g all_midis -n 100
python3 train.py -m vanilla_rnn -g all_midis -n 200
python3 train.py -m attention_rnn -g all_midis -n 150
python3 train.py -m transformer -g all_midis -n 200

python3 gan_training.py  -gm gan_gen -dm gan_disc -g all_midis -n 400
```

4. Evaluate metrics for all genres trained with all genres
```shell
python3 train.py -m lstm_attn -g classical -c aicathi_models/lstm_attn_all_all_midis_best.pth --test_only
python3 train.py -m lstm_attn -g folk -c aicathi_models/lstm_attn_all_all_midis_best.pth --test_only
python3 train.py -m lstm_attn -g pop_rock -c aicathi_models/lstm_attn_all_all_midis_best.pth --test_only
python3 train.py -m lstm_attn -g soul_reggae -c aicathi_models/lstm_attn_all_all_midis_best.pth --test_only


python3 train.py -m vanilla_rnn -g classical -c aicathi_models/vanilla_rnn_all_all_midis_best.pth --test_only
python3 train.py -m vanilla_rnn -g folk -c aicathi_models/vanilla_rnn_all_all_midis_best.pth --test_only
python3 train.py -m vanilla_rnn -g pop_rock -c aicathi_models/vanilla_rnn_all_all_midis_best.pth --test_only
python3 train.py -m vanilla_rnn -g soul_reggae -c aicathi_models/vanilla_rnn_all_all_midis_best.pth --test_only


python3 train.py -m attention_rnn -g classical -c aicathi_models/attention_rnn_all_all_midis_best.pth --test_only
python3 train.py -m attention_rnn -g folk -c aicathi_models/attention_rnn_all_all_midis_best.pth --test_only
python3 train.py -m attention_rnn -g pop_rock -c aicathi_models/attention_rnn_all_all_midis_best.pth --test_only
python3 train.py -m attention_rnn -g soul_reggae -c aicathi_models/attention_rnn_all_all_midis_best.pth --test_only


python3 train.py -m transformer -g classical -c aicathi_models/transformer_all_all_midis_best.pth --test_only
python3 train.py -m transformer -g folk -c aicathi_models/transformer_all_all_midis_best.pth --test_only
python3 train.py -m transformer -g pop_rock -c aicathi_models/transformer_all_all_midis_best.pth --test_only
python3 train.py -m transformer -g soul_reggae -c aicathi_models/transformer_all_all_midis_best.pth --test_only



python3 gan_training.py  -gm gan_gen -dm gan_disc  -g classical -gc aicathi_models/gan-G-gan_gen_all_all_midis_best.pth  -dc aicathi_models/gan-D-gan_disc_all_all_midis_best.pth --test_only
python3 gan_training.py  -gm gan_gen -dm gan_disc  -g folk -gc aicathi_models/gan-G-gan_gen_all_all_midis_best.pth  -dc aicathi_models/gan-D-gan_disc_all_all_midis_best.pth --test_only
python3 gan_training.py  -gm gan_gen -dm gan_disc  -g pop_rock -gc aicathi_models/gan-G-gan_gen_all_all_midis_best.pth  -dc aicathi_models/gan-D-gan_disc_all_all_midis_best.pth --test_only
python3 gan_training.py  -gm gan_gen -dm gan_disc  -g soul_reggae -gc aicathi_models/gan-G-gan_gen_all_all_midis_best.pth  -dc aicathi_models/gan-D-gan_disc_all_all_midis_best.pth --test_only
```


5. Fine tunning and evaluate
```bash
python3 train.py -m lstm_attn -g classical -n 100 -c aicathi_models/lstm_attn_all_all_midis_best.pth 
python3 train.py -m lstm_attn -g folk -n 100 -c aicathi_models/lstm_attn_all_all_midis_best.pth 
python3 train.py -m lstm_attn -g pop_rock -n 100 -c aicathi_models/lstm_attn_all_all_midis_best.pth 
python3 train.py -m lstm_attn -g soul_reggae -n 100 -c aicathi_models/lstm_attn_all_all_midis_best.pth 

python3 train.py -m vanilla_rnn -g classical -n 200 -c aicathi_models/vanilla_rnn_all_all_midis_best.pth
python3 train.py -m vanilla_rnn -g folk -n 200 -c aicathi_models/vanilla_rnn_all_all_midis_best.pth
python3 train.py -m vanilla_rnn -g pop_rock -n 200 -c aicathi_models/vanilla_rnn_all_all_midis_best.pth
python3 train.py -m vanilla_rnn -g soul_reggae -n 200 -c aicathi_models/vanilla_rnn_all_all_midis_best.pth

python3 train.py -m attention_rnn -g classical -n 150 -c aicathi_models/attention_rnn_all_all_midis_best.pth
python3 train.py -m attention_rnn -g folk -n 150 -c aicathi_models/attention_rnn_all_all_midis_best.pth
python3 train.py -m attention_rnn -g pop_rock -n 150 -c aicathi_models/attention_rnn_all_all_midis_best.pth
python3 train.py -m attention_rnn -g soul_reggae -n 150 -c aicathi_models/attention_rnn_all_all_midis_best.pth

python3 train.py -m transformer -g classical -n 200 -c aicathi_models/transformer_all_all_midis_best.pth
python3 train.py -m transformer -g folk -n 200 -c aicathi_models/transformer_all_all_midis_best.pth
python3 train.py -m transformer -g pop_rock -n 200 -c aicathi_models/transformer_all_all_midis_best.pth
python3 train.py -m transformer -g soul_reggae -n 200 -c aicathi_models/transformer_all_all_midis_best.pth

python3 gan_training.py  -gm gan_gen -dm gan_disc -g classical -n 400 -gc aicathi_models/gan-G-gan_gen_all_all_midis_best.pth  -dc aicathi_models/gan-D-gan_disc_all_all_midis_best.pth
python3 gan_training.py  -gm gan_gen -dm gan_disc -g folk -n 400 -gc aicathi_models/gan-G-gan_gen_all_all_midis_best.pth  -dc aicathi_models/gan-D-gan_disc_all_all_midis_best.pth
python3 gan_training.py  -gm gan_gen -dm gan_disc -g pop_rock -n 400 -gc aicathi_models/gan-G-gan_gen_all_all_midis_best.pth  -dc aicathi_models/gan-D-gan_disc_all_all_midis_best.pth
python3 gan_training.py  -gm gan_gen -dm gan_disc -g soul_reggae -n 400 -gc aicathi_models/gan-G-gan_gen_all_all_midis_best.pth  -dc aicathi_models/gan-D-gan_disc_all_all_midis_best.pth
```