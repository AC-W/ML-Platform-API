# fastAPI imports
import uvicorn
from fastapi import FastAPI ,File , UploadFile
from fastapi.middleware.cors import CORSMiddleware

# Machine Learning imports
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from io import BytesIO
from module.loadstore import FileManager as fm

# load labels
file_manager = fm()
car_labels = file_manager.loadData('./Labels/Car_model_labels')

transform = transforms.Compose([
    transforms.Resize((256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Grayscale(3),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
])

app = FastAPI()

origins = [
    "*",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

car_net = None

@app.get('/')
def index():
    return {'message': 'Hello, World'}

 # Load the car model recognition model
@app.get('/load_car_model')
def load_car_model():
    global car_net
    if (car_net):
        return {"msg":"car_model loaded"}
    else:
        car_net = torchvision.models.resnet50(pretrained=False)
        num_ftrs = car_net.fc.in_features
        car_net.fc = nn.Linear(num_ftrs, 1513)
        state = torch.hub.load_state_dict_from_url("https://github.com/AC-W/ML-Platform-Server/raw/main/Models/CarRecognitionModel",map_location=torch.device('cpu'))
        car_net.load_state_dict(state)
        return {"msg":"car_model loaded"}


# predict car model
@app.post('/predict_car_model')
async def predict_car_image(file: UploadFile = File(...)):
    if (car_net):
        img = Image.open(BytesIO(await file.read()))
        show_rank_num = 5
        img = transform(img)
        with torch.no_grad():
            car_net.eval()
            all = car_net(img.unsqueeze(0))
            p = torch.nn.functional.softmax(all[0], dim=0)
            _, top = torch.topk(all,show_rank_num)
            top = top.tolist()
            name = []
            prob = []
            for i in range(len(top[0])):
                index = top[0][i]
                name.append(f'{car_labels.get(index)}')
                prob.append(f'{p[index]*100:.2f}%')
        return {"models":name,"probs":prob}
    else:
        return {"msg":'model not loaded'}

if __name__ == '__main__':
    uvicorn.run(app)