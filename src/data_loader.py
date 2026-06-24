import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_data(filepath):
      wine_df = pd.read_csv(filepath)
      print("Dataset Shape:", wine_df.shape)
      print(wine_df.head())
      print(wine_df.info())
      return wine_df


def preprocess(wine_df):
      # Separate into dependent and independent variables
      x = wine_df.iloc[:, :-1].values
      y = wine_df.iloc[:, -1].values

    # Split into train and test sets
      x_train, x_test, y_train, y_test = train_test_split(
          x, y, test_size=0.2, random_state=42
      )
      print("x_train shape:", x_train.shape)

    # Apply standard scaling
      sc = StandardScaler()
      x_train = sc.fit_transform(x_train)
      x_test = sc.transform(x_test)

    return x_train, x_test, y_train, y_test
