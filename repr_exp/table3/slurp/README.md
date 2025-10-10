# not being used;;

# download audio (src)[https://github.com/pswietojanski/slurp/blob/master/scripts/download_audio.sh]
```bash
mkdir -p ../slurp

wget -O slurp_real.tar.gz https://zenodo.org/record/4274930/files/slurp_real.tar.gz
wget -O slurp_synth.tar.gz https://zenodo.org/record/4274930/files/slurp_synth.tar.gz


tar -zxvf slurp_real.tar.gz -C ../slurp
tar -zxvf slurp_synth.tar.gz -C ../slurp
```