import re

def classical_clean(text):
    text=text.lower()
    text=text.strip()
    text=re.sub(r"\s+"," ",text)
    return text

def preprocess_classical(df,col1,col2):
    df[col1]=df[col1].apply(classical_clean)
    df[col2]=df[col2].apply(classical_clean)
    return df

def transformer_clean(text):
    text=text.strip()
    text=re.sub(r"\s+"," ",text)
    return text

def preprocess_transformer(df,col1,col2):
    df[col1]=df[col1].apply(
        transformer_clean
    )
    df[col2]=df[col2].apply(
        transformer_clean
    )
    return df

