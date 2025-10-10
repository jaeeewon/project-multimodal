```bash
git clone https://github.com/jaeeewon/LibriMix.git ../LibriMix_repo
cd ../LibriMix_repo
pip install -r requirements.txt
bash generate_librimix.sh '..'
# ~.sh -> '..' + /LibriSpeech
# automatically downloads wham_noise (17G) and generate LibriMix
```