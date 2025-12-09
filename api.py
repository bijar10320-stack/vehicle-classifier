import torch
import uvicorn
from fastapi import FastAPI, UploadFile, File
from torchvision import transforms
from PIL import Image
from model import VehicleCNN  # CNN model

app = FastAPI()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

classes = ["Bus", "Car", "Truck", "motorcycle"]

model = VehicleCNN(input_feature=3, output_feature=len(classes), hidden_unit=10)
model.load_state_dict(torch.load("vehicle_cnn.pth", map_location=device))
model.to(device)
model.eval()


transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])


def predict_image(image: Image.Image):
    img_tensor = transform(image).unsqueeze(0).to(device)

    with torch.inference_mode():
        pred = model(img_tensor)
        class_id = pred.argmax(dim=1).item()

    return classes[class_id]


@app.get("/")
def home():
    return {"message": "Vehicle Classifier API is running!"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    img = Image.open(file.file).convert("RGB")
    prediction = predict_image(img)
    return {"predicted_class": prediction}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
