# AiCaThi: Using Deep Neural Networks to Generate Music from Beats
Authors: Aian Shay, Thiago Figueiredo, João Canavarro

## Abstract

This project introduces AiCaThi, an approach to AI-driven music generation, focusing on the translation of beats into diverse musical genres using
deep neural networks. We explore several neural network architectures, including Vanilla RNN and LSTM, LSTM with Local and Full Attention, GRU, and Transformer models trained on a comprehensive dataset covering multiple music genres. Our methodology involves both genre-specific training and fine-tuning, aiming to evaluate the effectiveness of each model in music generation. The results demonstrate that attention-enhanced LSTMs consistently outperform other models, highlighting the importance of attention mechanisms in capturing the intricacies of musical composition. This research contributes to the field of AI music generation by providing insights into the capabilities of different neural network architectures and their application in creating music across various genres.

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

- `lstm_local_attn`: LSTM with Local Attention
- `vanilla_rnn`: Decoder Only Vanilla RNN
- `vanilla_lstm`: Basic LSTM
- `gru`: Gated Recurrent Unit
- `lstm_full_attn`: LSTM with Full Attention
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
python train.py -m lstm_local_attn -g classical

# Train the LSTM with Local Attention model using 10 files of Folk songs, for 400 epochs, on the CPU, saving snapshots every 100 epochs, and starting from the checkpoint
python train.py -m lstm_local_attn -g folk -nf 10 -n 400 -d cpu -s 100 -c ./project_data/snapshots/my_checkpoint.pth

# Train the Transformer RPR model using all available files, for 500 epochs, on the default device, saving snapshots every 50 epochs, and not using a checkpoint
python train.py -m transformer -n 500 -s 50
```

## Generating Melodies from Beats

To generate a predicted notes sequence and save it as a MIDI file, run the `predict_stream.py` script with the `-m` or `--model_name` argument followed by a string specifying the name of the model to use. The available model names are:

- `lstm_local_attn`: LSTM with Local Attention
- `vanilla_rnn`: Decoder Only Vanilla RNN
- `vanilla_lstm`: Basic LSTM
- `gru`: Gated Recurrent Unit
- `lstm_full_attn`: LSTM with Full Attention
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
python predict_stream.py -m lstm_local_attn -c ./project_data/snapshots/lstm_local_attn_all_classical_best.pth -t beta
```

# Running AiCaThi experiments:

## Preparing environment

```shell
pip3 install virtualenv
python3 -m virtualenv -p python3 venv
source venv/bin/activate
pip install -r requirements.txt

# sudo apt install ffmpeg
# brew install ffmpeg
```

## Commands to run experiments

1. Train models with all genres
```shell
python3 train.py -m lstm_local_attn -g all_midis -n 100
python3 train.py -m vanilla_rnn -g all_midis -n 200
python3 train.py -m lstm_full_attn -g all_midis -n 150
python3 train.py -m transformer -g all_midis -n 200
python3 train.py -m gru -g all_midis -n 200
python3 train.py -m vanilla_lstm -g all_midis -n 200
python3 gan_training.py  -gm gan_gen -dm gan_disc -g all_midis -n 400
```

2. Evaluate metrics for all genres trained with all genres
```shell
python3 train.py -m lstm_local_attn -g classical -c aicathi_models/lstm_local_attn_all_all_midis_best.pth --test_only
python3 train.py -m lstm_local_attn -g folk -c aicathi_models/lstm_local_attn_all_all_midis_best.pth --test_only
python3 train.py -m lstm_local_attn -g pop_rock -c aicathi_models/lstm_local_attn_all_all_midis_best.pth --test_only
python3 train.py -m lstm_local_attn -g soul_reggae -c aicathi_models/lstm_local_attn_all_all_midis_best.pth --test_only

python3 train.py -m vanilla_rnn -g classical -c aicathi_models/vanilla_rnn_all_all_midis_best.pth --test_only
python3 train.py -m vanilla_rnn -g folk -c aicathi_models/vanilla_rnn_all_all_midis_best.pth --test_only
python3 train.py -m vanilla_rnn -g pop_rock -c aicathi_models/vanilla_rnn_all_all_midis_best.pth --test_only
python3 train.py -m vanilla_rnn -g soul_reggae -c aicathi_models/vanilla_rnn_all_all_midis_best.pth --test_only

python3 train.py -m lstm_full_attn -g classical -c aicathi_models/lstm_full_attn_all_all_midis_best.pth --test_only
python3 train.py -m lstm_full_attn -g folk -c aicathi_models/lstm_full_attn_all_all_midis_best.pth --test_only
python3 train.py -m lstm_full_attn -g pop_rock -c aicathi_models/lstm_full_attn_all_all_midis_best.pth --test_only
python3 train.py -m lstm_full_attn -g soul_reggae -c aicathi_models/lstm_full_attn_all_all_midis_best.pth --test_only

python3 train.py -m transformer -g classical -c aicathi_models/transformer_all_all_midis_best.pth --test_only
python3 train.py -m transformer -g folk -c aicathi_models/transformer_all_all_midis_best.pth --test_only
python3 train.py -m transformer -g pop_rock -c aicathi_models/transformer_all_all_midis_best.pth --test_only
python3 train.py -m transformer -g soul_reggae -c aicathi_models/transformer_all_all_midis_best.pth --test_only

python3 train.py -m gru -g classical -c aicathi_models/gru_all_all_midis_best.pth --test_only
python3 train.py -m gru -g folk -c aicathi_models/gru_all_all_midis_best.pth --test_only
python3 train.py -m gru -g pop_rock -c aicathi_models/gru_all_all_midis_best.pth --test_only
python3 train.py -m gru -g soul_reggae -c aicathi_models/gru_all_all_midis_best.pth --test_only

python3 train.py -m vanilla_lstm -g classical -c aicathi_models/vanilla_lstm_all_all_midis_best.pth --test_only
python3 train.py -m vanilla_lstm -g folk -c aicathi_models/vanilla_lstm_all_all_midis_best.pth --test_only
python3 train.py -m vanilla_lstm -g pop_rock -c aicathi_models/vanilla_lstm_all_all_midis_best.pth --test_only
python3 train.py -m vanilla_lstm -g soul_reggae -c aicathi_models/vanilla_lstm_all_all_midis_best.pth --test_only

python3 gan_training.py  -gm gan_gen -dm gan_disc  -g classical -gc aicathi_models/gan-G-gan_gen_all_all_midis_best.pth  -dc aicathi_models/gan-D-gan_disc_all_all_midis_best.pth --test_only
python3 gan_training.py  -gm gan_gen -dm gan_disc  -g folk -gc aicathi_models/gan-G-gan_gen_all_all_midis_best.pth  -dc aicathi_models/gan-D-gan_disc_all_all_midis_best.pth --test_only
python3 gan_training.py  -gm gan_gen -dm gan_disc  -g pop_rock -gc aicathi_models/gan-G-gan_gen_all_all_midis_best.pth  -dc aicathi_models/gan-D-gan_disc_all_all_midis_best.pth --test_only
python3 gan_training.py  -gm gan_gen -dm gan_disc  -g soul_reggae -gc aicathi_models/gan-G-gan_gen_all_all_midis_best.pth  -dc aicathi_models/gan-D-gan_disc_all_all_midis_best.pth --test_only
```


3. Fine-tuning

    3.1 Train base model

```shell
python3 train.py -m lstm_local_attn -g midis_except_classical -n 100
python3 train.py -m lstm_local_attn -g midis_except_folk -n 100
python3 train.py -m lstm_local_attn -g midis_except_pop_rock -n 100
python3 train.py -m lstm_local_attn -g midis_except_soul_reggae -n 100

python3 train.py -m vanilla_rnn -g midis_except_classical -n 200
python3 train.py -m vanilla_rnn -g midis_except_folk -n 200
python3 train.py -m vanilla_rnn -g midis_except_pop_rock -n 200
python3 train.py -m vanilla_rnn -g midis_except_soul_reggae -n 200

python3 train.py -m lstm_full_attn -g midis_except_classical -n 150
python3 train.py -m lstm_full_attn -g midis_except_folk -n 150
python3 train.py -m lstm_full_attn -g midis_except_pop_rock -n 150
python3 train.py -m lstm_full_attn -g midis_except_soul_reggae -n 150

python3 train.py -m transformer -g midis_except_classical -n 200
python3 train.py -m transformer -g midis_except_folk -n 200
python3 train.py -m transformer -g midis_except_pop_rock -n 200
python3 train.py -m transformer -g midis_except_soul_reggae -n 200

python3 train.py -m gru -g midis_except_classical -n 200
python3 train.py -m gru -g midis_except_folk -n 200
python3 train.py -m gru -g midis_except_pop_rock -n 200
python3 train.py -m gru -g midis_except_soul_reggae -n 200

python3 train.py -m vanilla_lstm -g midis_except_classical -n 200
python3 train.py -m vanilla_lstm -g midis_except_folk -n 200
python3 train.py -m vanilla_lstm -g midis_except_pop_rock -n 200
python3 train.py -m vanilla_lstm -g midis_except_soul_reggae -n 200
```

    3.2 Evaluate base model
```shell
python3 train.py -m lstm_local_attn -g classical -c aicathi_models/lstm_local_attn_all_midis_except_classical_best.pth --test_only
python3 train.py -m lstm_local_attn -g folk -c aicathi_models/lstm_local_attn_all_midis_except_folk_best.pth --test_only
python3 train.py -m lstm_local_attn -g pop_rock -c aicathi_models/lstm_local_attn_all_midis_except_pop_rock_best.pth --test_only
python3 train.py -m lstm_local_attn -g classical -c aicathi_models/lstm_local_attn_all_midis_except_soul_reggae_best.pth --test_only

python3 train.py -m vanilla_rnn -g classical -c aicathi_models/vanilla_rnn_all_midis_except_classical_best.pth --test_only
python3 train.py -m vanilla_rnn -g folk -c aicathi_models/vanilla_rnn_all_midis_except_folk_best.pth --test_only
python3 train.py -m vanilla_rnn -g pop_rock -c aicathi_models/vanilla_rnn_all_midis_except_pop_rock_best.pth --test_only
python3 train.py -m vanilla_rnn -g soul_reggae -c aicathi_models/vanilla_rnn_all_midis_except_soul_reggae_best.pth --test_only

python3 train.py -m lstm_full_attn -g classical -c aicathi_models/lstm_full_attn_all_midis_except_classical_best.pth --test_only
python3 train.py -m lstm_full_attn -g folk -c aicathi_models/lstm_full_attn_all_midis_except_folk_best.pth --test_only
python3 train.py -m lstm_full_attn -g pop_rock -c aicathi_models/lstm_full_attn_all_midis_except_pop_rock_best.pth --test_only
python3 train.py -m lstm_full_attn -g soul_reggae -c aicathi_models/lstm_full_attn_all_midis_except_soul_reggae_best.pth --test_only

python3 train.py -m transformer -g classical -c aicathi_models/transformer_all_midis_except_classical_best.pth --test_only
python3 train.py -m transformer -g folk -c aicathi_models/transformer_all_midis_except_folk_best.pth --test_only
python3 train.py -m transformer -g pop_rock -c aicathi_models/transformer_all_midis_except_pop_rock_best.pth --test_only
python3 train.py -m transformer -g soul_reggae -c aicathi_models/transformer_all_midis_except_soul_reggae_best.pth --test_only

python3 train.py -m gru -g classical -c aicathi_models/gru_all_midis_except_classical_best.pth --test_only
python3 train.py -m gru -g folk -c aicathi_models/gru_all_midis_except_folk_best.pth --test_only
python3 train.py -m gru -g pop_rock -c aicathi_models/gru_all_midis_except_pop_rock_best.pth --test_only
python3 train.py -m gru -g soul_reggae -c aicathi_models/gru_all_midis_except_soul_reggae_best.pth --test_only

python3 train.py -m vanilla_lstm -g classical -c aicathi_models/vanilla_lstm_all_midis_except_classical_best.pth --test_only
python3 train.py -m vanilla_lstm -g folk -c aicathi_models/vanilla_lstm_all_midis_except_folk_best.pth --test_only
python3 train.py -m vanilla_lstm -g pop_rock -c aicathi_models/vanilla_lstm_all_midis_except_pop_rock_best.pth --test_only
python3 train.py -m vanilla_lstm -g soul_reggae -c aicathi_models/vanilla_lstm_all_midis_except_soul_reggae_best.pth --test_only
```

    3.3 Fine-tune and retrieve results
```bash
python3 train.py -m lstm_local_attn -g classical -n 50 -c aicathi_models/lstm_local_attn_all_midis_except_classical_best.pth 
python3 train.py -m lstm_local_attn -g folk -n 50 -c aicathi_models/lstm_local_attn_all_midis_except_folk_best.pth 
python3 train.py -m lstm_local_attn -g pop_rock -n 50 -c aicathi_models/lstm_local_attn_all_midis_except_pop_rock_best.pth 
python3 train.py -m lstm_local_attn -g soul_reggae -n 50 -c aicathi_models/lstm_local_attn_all_midis_except_soul_reggae_best.pth 

python3 train.py -m vanilla_rnn -g classical -n 100 -c aicathi_models/vanilla_rnn_all_midis_except_classical_best.pth
python3 train.py -m vanilla_rnn -g folk -n 100 -c aicathi_models/vanilla_rnn_all_midis_except_folk_best.pth
python3 train.py -m vanilla_rnn -g pop_rock -n 100 -c aicathi_models/vanilla_rnn_all_midis_except_pop_rock_best.pth
python3 train.py -m vanilla_rnn -g soul_reggae -n 100 -c aicathi_models/vanilla_rnn_all_midis_except_soul_reggae_best.pth

python3 train.py -m lstm_full_attn -g classical -n 75 -c aicathi_models/lstm_full_attn_all_midis_except_classical_best.pth
python3 train.py -m lstm_full_attn -g folk -n 75 -c aicathi_models/lstm_full_attn_all_midis_except_folk_best.pth
python3 train.py -m lstm_full_attn -g pop_rock -n 75 -c aicathi_models/lstm_full_attn_all_midis_except_pop_rock_best.pth
python3 train.py -m lstm_full_attn -g soul_reggae -n 75 -c aicathi_models/lstm_full_attn_all_midis_except_soul_reggae_best.pth

python3 train.py -m transformer -g classical -n 100 -c aicathi_models/transformer_all_midis_except_classical_best.pth
python3 train.py -m transformer -g folk -n 100 -c aicathi_models/transformer_all_midis_except_folk_best.pth
python3 train.py -m transformer -g pop_rock -n 100 -c aicathi_models/transformer_all_midis_except_pop_rock_best.pth
python3 train.py -m transformer -g soul_reggae -n 100 -c aicathi_models/transformer_all_midis_except_soul_reggae_best.pth

python3 train.py -m gru -g classical -n 100 -c aicathi_models/gru_all_midis_except_classical_best.pth
python3 train.py -m gru -g folk -n 100 -c aicathi_models/gru_all_midis_except_folk_best.pth
python3 train.py -m gru -g pop_rock -n 100 -c aicathi_models/gru_all_midis_except_pop_rock_best.pth
python3 train.py -m gru -g soul_reggae -n 100 -c aicathi_models/gru_all_midis_except_soul_reggae_best.pth

python3 train.py -m vanilla_lstm -g classical -n 100 -c aicathi_models/vanilla_lstm_all_midis_except_classical_best.pth
python3 train.py -m vanilla_lstm -g folk -n 100 -c aicathi_models/vanilla_lstm_all_midis_except_folk_best.pth
python3 train.py -m vanilla_lstm -g pop_rock -n 100 -c aicathi_models/vanilla_lstm_all_midis_except_pop_rock_best.pth
python3 train.py -m vanilla_lstm -g soul_reggae -n 100 -c aicathi_models/vanilla_lstm_all_midis_except_soul_reggae_best.pth
```