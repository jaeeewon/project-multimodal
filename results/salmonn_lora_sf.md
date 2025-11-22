# changes in lora scaling factor for salmonn-7b

## SKR-LoRA-SF-3-B5

```bash
python -m util.sakura_ld_datasets --model_name salmonn-7b --exp_id SKR-LoRA-SF-3-B5 --type source --pos original --target_len 30
python -m evaluation.sakura_exp --device cuda:0 cuda:1 --model_name salmonn-7b --exp_ids SKR-LoRA-SF-3-B5 --batch_size 5 --skip_confirm --inference_only
python -m evaluation.sakura_exp --device cuda:0 cuda:1 --model_name salmonn-7b --exp_ids SKR-LoRA-SF-3-B5 --batch_size 5 --skip_confirm --save_exp
```

## SKR-LoRA-SF-2-B5

```bash
python -m util.sakura_ld_datasets --model_name salmonn-7b --exp_id SKR-LoRA-SF-2-B5 --type source --pos original --target_len 30
python -m evaluation.sakura_exp --device cuda:0 cuda:1 --model_name salmonn-7b --exp_ids SKR-LoRA-SF-2-B5 --batch_size 5 --skip_confirm --inference_only
python -m evaluation.sakura_exp --device cuda:0 cuda:1 --model_name salmonn-7b --exp_ids SKR-LoRA-SF-2-B5 --batch_size 5 --skip_confirm --save_exp
```

## SKR-LoRA-SF-1-B5

```bash
python -m util.sakura_ld_datasets --model_name salmonn-7b --exp_id SKR-LoRA-SF-1-B5 --type source --pos original --target_len 30
python -m evaluation.sakura_exp --device cuda:0 cuda:1 --model_name salmonn-7b --exp_ids SKR-LoRA-SF-1-B5 --batch_size 5 --skip_confirm --inference_only
python -m evaluation.sakura_exp --device cuda:0 cuda:1 --model_name salmonn-7b --exp_ids SKR-LoRA-SF-1-B5 --batch_size 5 --skip_confirm --save_exp
```

## SKR-LoRA-SF-0-B5

```bash
python -m util.sakura_ld_datasets --model_name salmonn-7b --exp_id SKR-LoRA-SF-0-B5 --type source --pos original --target_len 30
python -m evaluation.sakura_exp --device cuda:0 cuda:1 --model_name salmonn-7b --exp_ids SKR-LoRA-SF-0-B5 --batch_size 5 --skip_confirm --inference_only
python -m evaluation.sakura_exp --device cuda:0 cuda:1 --model_name salmonn-7b --exp_ids SKR-LoRA-SF-0-B5 --batch_size 5 --skip_confirm --save_exp
```

## results

| experiment                     | base                           |   run_num | category        |   correct |   incorrect |   total |   accuracy |   accuracy_delta |
|:-------------------------------|:-------------------------------|----------:|:----------------|----------:|------------:|--------:|-----------:|-----------------:|
| SKR-LoRA-SF-0-B5               | SKR-LoRA-SF-0-B5               |         0 | animal-multi    |       192 |         308 |     500 |       38.4 |              3.8 |
| SKR-LoRA-SF-0-B5               | SKR-LoRA-SF-0-B5               |         0 | animal-single   |       243 |         257 |     500 |       48.6 |            -19.6 |
| SKR-LoRA-SF-0-B5               | SKR-LoRA-SF-0-B5               |         0 | emotion-multi   |       129 |         371 |     500 |       25.8 |             -2.6 |
| SKR-LoRA-SF-0-B5               | SKR-LoRA-SF-0-B5               |         0 | emotion-single  |       109 |         391 |     500 |       21.8 |              1.8 |
| SKR-LoRA-SF-0-B5               | SKR-LoRA-SF-0-B5               |         0 | gender-multi    |       189 |         311 |     500 |       37.8 |            -11   |
| SKR-LoRA-SF-0-B5               | SKR-LoRA-SF-0-B5               |         0 | gender-single   |       183 |         317 |     500 |       36.6 |            -23.4 |
| SKR-LoRA-SF-0-B5               | SKR-LoRA-SF-0-B5               |         0 | language-multi  |       171 |         329 |     500 |       34.2 |              4.4 |
| SKR-LoRA-SF-0-B5               | SKR-LoRA-SF-0-B5               |         0 | language-single |       151 |         349 |     500 |       30.2 |              9.6 |
| SKR-LoRA-SF-1-B5               | SKR-LoRA-SF-1-B5               |         0 | animal-multi    |       190 |         310 |     500 |       38   |              3.4 |
| SKR-LoRA-SF-1-B5               | SKR-LoRA-SF-1-B5               |         0 | animal-single   |       275 |         225 |     500 |       55   |            -13.2 |
| SKR-LoRA-SF-1-B5               | SKR-LoRA-SF-1-B5               |         0 | emotion-multi   |       128 |         372 |     500 |       25.6 |             -2.8 |
| SKR-LoRA-SF-1-B5               | SKR-LoRA-SF-1-B5               |         0 | emotion-single  |       121 |         379 |     500 |       24.2 |              4.2 |
| SKR-LoRA-SF-1-B5               | SKR-LoRA-SF-1-B5               |         0 | gender-multi    |       189 |         311 |     500 |       37.8 |            -11   |
| SKR-LoRA-SF-1-B5               | SKR-LoRA-SF-1-B5               |         0 | gender-single   |       193 |         307 |     500 |       38.6 |            -21.4 |
| SKR-LoRA-SF-1-B5               | SKR-LoRA-SF-1-B5               |         0 | language-multi  |       165 |         335 |     500 |       33   |              3.2 |
| SKR-LoRA-SF-1-B5               | SKR-LoRA-SF-1-B5               |         0 | language-single |       154 |         346 |     500 |       30.8 |             10.2 |
| SKR-LoRA-SF-2-B5               | SKR-LoRA-SF-2-B5               |         0 | animal-multi    |       187 |         313 |     500 |       37.4 |              2.8 |
| SKR-LoRA-SF-2-B5               | SKR-LoRA-SF-2-B5               |         0 | animal-single   |       308 |         192 |     500 |       61.6 |             -6.6 |
| SKR-LoRA-SF-2-B5               | SKR-LoRA-SF-2-B5               |         0 | emotion-multi   |       144 |         356 |     500 |       28.8 |              0.4 |
| SKR-LoRA-SF-2-B5               | SKR-LoRA-SF-2-B5               |         0 | emotion-single  |       114 |         386 |     500 |       22.8 |              2.8 |
| SKR-LoRA-SF-2-B5               | SKR-LoRA-SF-2-B5               |         0 | gender-multi    |       221 |         279 |     500 |       44.2 |             -4.6 |
| SKR-LoRA-SF-2-B5               | SKR-LoRA-SF-2-B5               |         0 | gender-single   |       227 |         273 |     500 |       45.4 |            -14.6 |
| SKR-LoRA-SF-2-B5               | SKR-LoRA-SF-2-B5               |         0 | language-multi  |       174 |         326 |     500 |       34.8 |              5   |
| SKR-LoRA-SF-2-B5               | SKR-LoRA-SF-2-B5               |         0 | language-single |       163 |         337 |     500 |       32.6 |             12   |
| SKR-LoRA-SF-3-B5               | SKR-LoRA-SF-3-B5               |         0 | animal-multi    |       203 |         297 |     500 |       40.6 |              6   |
| SKR-LoRA-SF-3-B5               | SKR-LoRA-SF-3-B5               |         0 | animal-single   |       345 |         155 |     500 |       69   |              0.8 |
| SKR-LoRA-SF-3-B5               | SKR-LoRA-SF-3-B5               |         0 | emotion-multi   |       142 |         358 |     500 |       28.4 |              0   |
| SKR-LoRA-SF-3-B5               | SKR-LoRA-SF-3-B5               |         0 | emotion-single  |        93 |         407 |     500 |       18.6 |             -1.4 |
| SKR-LoRA-SF-3-B5               | SKR-LoRA-SF-3-B5               |         0 | gender-multi    |       263 |         237 |     500 |       52.6 |              3.8 |
| SKR-LoRA-SF-3-B5               | SKR-LoRA-SF-3-B5               |         0 | gender-single   |       250 |         250 |     500 |       50   |            -10   |
| SKR-LoRA-SF-3-B5               | SKR-LoRA-SF-3-B5               |         0 | language-multi  |       157 |         343 |     500 |       31.4 |              1.6 |
| SKR-LoRA-SF-3-B5               | SKR-LoRA-SF-3-B5               |         0 | language-single |       166 |         334 |     500 |       33.2 |             12.6 |