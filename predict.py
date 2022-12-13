import bentoml
from bentoml.io import JSON
from pydantic import BaseModel

class PenaltyAim(BaseModel):
    team: str
    opposition: str
    minute_taken: int
    scoreline: str
    venue: str
    previous_penalty: int
    natural_foot: str
    position_style: str
    competition: str
    gk: str


model_ref = bentoml.xgboost.get("cristiano_penalty_aim_model:latest")
dv = model_ref.custom_objects["DictVectorizer"]


model_runner = model_ref.to_runner()


svc = bentoml.Service("cristiano_penalty_aim", runners=[model_runner])


@svc.api(input=JSON(pydantic_model=PenaltyAim), output=JSON())
async def classify(cristiano_penalty_aim):
    application_data = cristiano_penalty_aim.dict()
    vector = dv.transform(application_data)
    prediction = await model_runner.predict.async_run(vector)
    print(prediction)

    result = prediction[0]

    if result > 0.55 and result<1.45:
        return {"Aim": "left"}
    elif result > 1.44 and result<2.45:
        return {"Aim": "middle"}
    else :
        return {"Aim": "right"}
    