
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import OneHotEncoder, FunctionTransformer, StandardScaler, PowerTransformer
import joblib

def encode_sex_column(df):
  df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
  return df

sex_transformer = FunctionTransformer(encode_sex_column, validate=False)
cat_transformer = OneHotEncoder(drop="first")
num_transformer = Pipeline([
    ("power_trans", PowerTransformer(method='yeo-johnson')),
    ("std_scaler", StandardScaler())
])

preprocessor = ColumnTransformer([
    ("embarked", cat_transformer, ["Embarked"]),
    ("sex", sex_transformer, ["Sex"]),
    ("num_features", num_transformer, ["Fare", "Age"]),

])

pipeline = Pipeline([
    ("preprocessing", preprocessor)
    ])

df_transformed = pipeline.fit_transform(df_for_pipeline)
joblib.dump(pipeline, "models/preprocessing_pipeline.pkl")
