import os
import pandas as pd


def save_csv(df: pd.DataFrame, filename: str, verbose: bool = True, existed: str = 'append') -> None:
    if existed == 'overwrite':
        pass
    elif existed == 'append':
        if os.path.exists(filename):
            df = pd.concat([pd.read_csv(filename, index_col=0), df])
    elif existed == 'keep_both':
        base, ext = os.path.splitext(filename)
        cnt = 1
        while os.path.exists(filename):
            filename = f"{base}-{cnt}{ext}"
            cnt += 1
    elif existed == 'raise' and os.path.exists(filename):
        raise FileExistsError(f"{filename} already exists.")
    else:
        raise ValueError(f"Unknown value for 'existed': {existed}")

    os.makedirs(os.path.dirname(filename), exist_ok=True)
    df.to_csv(filename)
    print(f"{filename} saved.")
    if verbose:
        print(df)
    return df
