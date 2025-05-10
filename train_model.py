import requests

def train_model():
    # Gọi API retrain_model
    response = requests.post('http://localhost:8000/api/v1/interactions/retrain_model/')
    
    if response.status_code == 200:
        print("Model trained successfully!")
        print("Model saved in recommender/model/")
    else:
        print("Error training model:", response.text)

if __name__ == "__main__":
    train_model() 