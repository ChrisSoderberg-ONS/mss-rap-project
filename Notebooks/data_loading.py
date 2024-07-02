from ucimlrepo import fetch_ucirepo
import pandas as pd

def load_data():
    abalones = fetch_ucirepo(id=1)
    df = pd.DataFrame(abalones.data.features)
    df['Rings'] = abalones.data.targets
    return df
