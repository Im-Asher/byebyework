import uvicorn

from fastapi import FastAPI

app = FastAPI()



@app.get("/") # 根路由
def root():
    return {"ByeByeWork(筹)"}


if __name__=="__main__":
    uvicorn.run(app='main:app',host='0.0.0.0')