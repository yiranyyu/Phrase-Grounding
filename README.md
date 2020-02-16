# Phrase-Grounding

## Train

```sh
python ./main.py train @configs/cfg-s1204-L1-H2-dp0.4-b256-lr0.00005-wp0.1-abs.args --seed 0 --bs 64 --split train,test --grad-acc-steps 2
```
