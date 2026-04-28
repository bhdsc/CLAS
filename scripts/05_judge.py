import __import__
print(__file__)

import os
import pandas as pd
from glob import glob

def get_datasets(concept, splits=['train', 'valid', 'test'], exclude='-judge.csv'):
    splits = ['test']

    paths = glob(f'{__import__.path_coefficients}/**/**/{concept}_0_*.csv', recursive=True)
    paths = sorted(set(paths))

    paths = [path for path in paths if any(split in path for split in splits)]
    paths = [path for path in paths if not os.path.isfile(path.replace('.csv', '-judge.csv'))]
    paths = [path for path in paths if exclude not in path]

    datasets = [{'path': path, 'data': pd.read_csv(path)} for path in paths]

    return datasets

def judge_datasets():
    from _judge import score_judges

    judges = score_judges
    print(list(judges))

    for judge_id, judge in judges.items():
        judge = judge()
        datasets = get_datasets(concept=judge_id)

        for dataset in datasets:
            path = dataset['path'].replace('.csv', '-judge.csv')
            df = dataset['data']
            assert "model_output" in df.columns
            df["judge_score"] = judge(*judge.prepare(df))
            df["judge_accuracy"] = judge.score_to_accuracy()
            df["judge_instructions_template"] = judge.instructions_template
            df["judge_prompt_template"] = judge.prompt_template
            df.to_csv(path, index=False)
            print(path)
            print(df["judge_accuracy"].value_counts(ascending=True))

    return judges

def main():
    judge_datasets()

if __name__ == "__main__":
    main()