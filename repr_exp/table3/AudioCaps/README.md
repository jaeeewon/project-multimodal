# prepare metadata

- download dataset from huggingface and call under
- original dataset has 4875 rows, but we only collected 4411 audios

```python
audiocaps[target_set] = audiocaps[target_set].select_columns(["youtube_id", "caption"])
audiocaps[target_set].to_csv(meta_path)
```