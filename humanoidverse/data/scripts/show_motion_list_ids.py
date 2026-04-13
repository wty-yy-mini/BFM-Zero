import joblib
import argparse

PATH_PKL = "/root/Coding/BFM-Zero/humanoidverse/data/dailylife_data_v1_bfmzero.pkl"

if __name__ == "__main__":
    data = joblib.load(PATH_PKL)
    for i, key in enumerate(data.keys()):
        print(f"{i}: {key}")
