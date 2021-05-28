# Beat Saber Map Generator

Beat Saber map generator created using PyTorch. Using a deep learning LSTM, this program produces a beatmap for a given song that can be played on BeatSaber.This model was trained on a dataset of 10000 beatmaps over the span of 24 Hours. This generator takes into account past combinations of beats as well as audio analysis with LibROSA. A baseline model for this generator was created using Markov Chains to map the beats to strings and treat them as sentences. 

## Example Gameplay:

Original (Left) vs Generated (Right)
https://user-images.githubusercontent.com/40185967/120013662-017bba80-bfaf-11eb-864a-63923361d44d.mp4


