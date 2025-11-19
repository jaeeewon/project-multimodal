# changes in qformer window size and stride for salmonn-7b

## SKR-BASE-B5

- second_per_window: 0.333333
- second_stride: 0.333333

```bash
python -m util.sakura_ld_datasets --model_name salmonn-7b --exp_id SKR-BASE-B5 --type source --pos original --target_len 30
python -m evaluation.sakura_exp --device cuda:0 cuda:1 cuda:2 cuda:3 --model_name salmonn-7b --exp_ids SKR-BASE-B5 --batch_size 5 --skip_confirm --inference_only
python -m evaluation.sakura_exp --device cuda:0 cuda:1 cuda:2 cuda:3 --model_name salmonn-7b --exp_ids SKR-BASE-B5 --batch_size 5 --skip_confirm --save_exp
```

## SKR-WIN-SIZE-X2-B5

- second_per_window: 0.666666
- second_stride: 0.333333

```bash
python -m util.sakura_ld_datasets --model_name salmonn-7b --exp_id SKR-WIN-SIZE-X2-B5 --type source --pos original --target_len 30
python -m evaluation.sakura_exp --device cuda:0 cuda:1 --model_name salmonn-7b --exp_ids SKR-WIN-SIZE-X2-B5 --batch_size 5 --skip_confirm --inference_only
python -m evaluation.sakura_exp --device cuda:0 cuda:1 --model_name salmonn-7b --exp_ids SKR-WIN-SIZE-X2-B5 --batch_size 5 --skip_confirm --save_exp
```

## SKR-WIN-SIZE-X3.3-B5 | 1 0.5

- second_per_window: 1
- second_stride: 0.5

```bash
python -m util.sakura_ld_datasets --model_name salmonn-7b --exp_id SKR-WIN-SIZE-X3.3-B5 --type source --pos original --target_len 30
python -m evaluation.sakura_exp --device cuda:0 cuda:1 --model_name salmonn-7b --exp_ids SKR-WIN-SIZE-X3.3-B5 --batch_size 5 --skip_confirm --inference_only
python -m evaluation.sakura_exp --device cuda:0 cuda:1 --model_name salmonn-7b --exp_ids SKR-WIN-SIZE-X3.3-B5 --batch_size 5 --skip_confirm --save_exp
```

## SKR-WIN-SIZE-0.5-B5 0.5 0.5

- second_per_window: 0.5
- second_stride: 0.5

```bash
python -m util.sakura_ld_datasets --model_name salmonn-7b --exp_id SKR-WIN-SIZE-0.5-B5 --type source --pos original --target_len 30
python -m evaluation.sakura_exp --device cuda:0 cuda:1 --model_name salmonn-7b --exp_ids SKR-WIN-SIZE-0.5-B5 --batch_size 5 --skip_confirm --inference_only
python -m evaluation.sakura_exp --device cuda:0 cuda:1 --model_name salmonn-7b --exp_ids SKR-WIN-SIZE-0.5-B5 --batch_size 5 --skip_confirm --save_exp
```

## results

| experiment                     | base                           |   run_num | category        |   correct |   incorrect |   total |   accuracy |   accuracy_delta |
|:-------------------------------|:-------------------------------|----------:|:----------------|----------:|------------:|--------:|-----------:|-----------------:|
| SKR-BASE-B5                    | SKR-BASE-B5                    |         0 | animal-multi    |       175 |         325 |     500 |       35   |              0.4 |
| SKR-BASE-B5                    | SKR-BASE-B5                    |         0 | animal-single   |       342 |         158 |     500 |       68.4 |              0.2 |
| SKR-BASE-B5                    | SKR-BASE-B5                    |         0 | emotion-multi   |       141 |         359 |     500 |       28.2 |             -0.2 |
| SKR-BASE-B5                    | SKR-BASE-B5                    |         0 | emotion-single  |       101 |         399 |     500 |       20.2 |              0.2 |
| SKR-BASE-B5                    | SKR-BASE-B5                    |         0 | gender-multi    |       242 |         258 |     500 |       48.4 |             -0.4 |
| SKR-BASE-B5                    | SKR-BASE-B5                    |         0 | gender-single   |       301 |         199 |     500 |       60.2 |              0.2 |
| SKR-BASE-B5                    | SKR-BASE-B5                    |         0 | language-multi  |       148 |         352 |     500 |       29.6 |             -0.2 |
| SKR-BASE-B5                    | SKR-BASE-B5                    |         0 | language-single |       106 |         394 |     500 |       21.2 |              0.6 |
| SKR-WIN-SIZE-X2-B5             | SKR-WIN-SIZE-X2-B5             |         0 | animal-multi    |       172 |         328 |     500 |       34.4 |             -0.2 |
| SKR-WIN-SIZE-X2-B5             | SKR-WIN-SIZE-X2-B5             |         0 | animal-single   |       337 |         163 |     500 |       67.4 |             -0.8 |
| SKR-WIN-SIZE-X2-B5             | SKR-WIN-SIZE-X2-B5             |         0 | emotion-multi   |       146 |         354 |     500 |       29.2 |              0.8 |
| SKR-WIN-SIZE-X2-B5             | SKR-WIN-SIZE-X2-B5             |         0 | emotion-single  |        98 |         402 |     500 |       19.6 |             -0.4 |
| SKR-WIN-SIZE-X2-B5             | SKR-WIN-SIZE-X2-B5             |         0 | gender-multi    |       244 |         256 |     500 |       48.8 |              0   |
| SKR-WIN-SIZE-X2-B5             | SKR-WIN-SIZE-X2-B5             |         0 | gender-single   |       310 |         190 |     500 |       62   |              2   |
| SKR-WIN-SIZE-X2-B5             | SKR-WIN-SIZE-X2-B5             |         0 | language-multi  |       146 |         354 |     500 |       29.2 |             -0.6 |
| SKR-WIN-SIZE-X2-B5             | SKR-WIN-SIZE-X2-B5             |         0 | language-single |       110 |         390 |     500 |       22   |              1.4 |
| SKR-WIN-SIZE-X3.3-B5           | SKR-WIN-SIZE-X3.3-B5           |         0 | animal-multi    |       164 |         336 |     500 |       32.8 |             -1.8 |
| SKR-WIN-SIZE-X3.3-B5           | SKR-WIN-SIZE-X3.3-B5           |         0 | animal-single   |       289 |         211 |     500 |       57.8 |            -10.4 |
| SKR-WIN-SIZE-X3.3-B5           | SKR-WIN-SIZE-X3.3-B5           |         0 | emotion-multi   |       147 |         353 |     500 |       29.4 |              1   |
| SKR-WIN-SIZE-X3.3-B5           | SKR-WIN-SIZE-X3.3-B5           |         0 | emotion-single  |        97 |         403 |     500 |       19.4 |             -0.6 |
| SKR-WIN-SIZE-X3.3-B5           | SKR-WIN-SIZE-X3.3-B5           |         0 | gender-multi    |       237 |         263 |     500 |       47.4 |             -1.4 |
| SKR-WIN-SIZE-X3.3-B5           | SKR-WIN-SIZE-X3.3-B5           |         0 | gender-single   |       254 |         246 |     500 |       50.8 |             -9.2 |
| SKR-WIN-SIZE-X3.3-B5           | SKR-WIN-SIZE-X3.3-B5           |         0 | language-multi  |       145 |         355 |     500 |       29   |             -0.8 |
| SKR-WIN-SIZE-X3.3-B5           | SKR-WIN-SIZE-X3.3-B5           |         0 | language-single |       134 |         366 |     500 |       26.8 |              6.2 |
| SKR-WIN-SIZE-0.5-B5            | SKR-WIN-SIZE-0.5-B5            |         0 | animal-multi    |       167 |         333 |     500 |       33.4 |             -1.2 |
| SKR-WIN-SIZE-0.5-B5            | SKR-WIN-SIZE-0.5-B5            |         0 | animal-single   |       293 |         207 |     500 |       58.6 |             -9.6 |
| SKR-WIN-SIZE-0.5-B5            | SKR-WIN-SIZE-0.5-B5            |         0 | emotion-multi   |       147 |         353 |     500 |       29.4 |              1   |
| SKR-WIN-SIZE-0.5-B5            | SKR-WIN-SIZE-0.5-B5            |         0 | emotion-single  |        95 |         405 |     500 |       19   |             -1   |
| SKR-WIN-SIZE-0.5-B5            | SKR-WIN-SIZE-0.5-B5            |         0 | gender-multi    |       240 |         260 |     500 |       48   |             -0.8 |
| SKR-WIN-SIZE-0.5-B5            | SKR-WIN-SIZE-0.5-B5            |         0 | gender-single   |       256 |         244 |     500 |       51.2 |             -8.8 |
| SKR-WIN-SIZE-0.5-B5            | SKR-WIN-SIZE-0.5-B5            |         0 | language-multi  |       149 |         351 |     500 |       29.8 |              0   |
| SKR-WIN-SIZE-0.5-B5            | SKR-WIN-SIZE-0.5-B5            |         0 | language-single |       123 |         377 |     500 |       24.6 |              4   |
