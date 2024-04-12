from fastapi import FastAPI
from pydantic import BaseModel
from kobert_model.predictor import KoBERTPredictor

app = FastAPI()


class Review(BaseModel):
    review_sentence: str


@app.post("/predict")
async def predict_result(review: Review):
    review_sentence = review.review_sentence
    predictor = KoBERTPredictor()
    emotion_probability = predictor.predict(review_sentence)
    processed_data = {"result" : emotion_probability}
    return processed_data
