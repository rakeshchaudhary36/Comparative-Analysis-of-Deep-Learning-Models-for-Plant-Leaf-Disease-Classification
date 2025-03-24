import streamlit as st
from streamlit_option_menu import option_menu
from PIL import Image
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from torch import nn
from efficientnet_pytorch import EfficientNet
import base64 

 

def get_base64(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_background(image_file):
    bin_str = get_base64(image_file)
    page_bg_img = f'''
    <style>
    [data-testid="stAppViewContainer"] > .main {{
      background-image: url("data:image/jpeg;base64,{bin_str}");
      background-size: cover;
      background-position: top;
      background-repeat: no-repeat;
      background-attachment: fixed;
    }}
    
    [data-testid="stSidebar"] > div:first-child {{
      background-image: url("data:image/jpeg;base64,{bin_str}");
      background-position: center; 
      background-repeat: no-repeat;
      background-attachment: fixed;
    }}

    [data-testid="stHeader"] {{
      background: rgba(0,0,0,0);
    }}

    [data-testid="stToolbar"] {{
      right: 2rem;
    }}
    </style>
    '''
    st.markdown(page_bg_img, unsafe_allow_html=True)

set_background('back.jpg')

# CSS part for coloring text
text_color_css = '''
<style>
h1 {
    color: black; 
    background: rgba(0,0,0,0);
    text-align: left; 
    margin-top: 10px; 
}

h2 {
    color: black; 
    font-size: 30px;
}

h3 {
    color: black; 
    font-size: 20px;
}

h4 {
    color: black; 
    font-size: 20px;
}

p, ul {
    color: black; 
    font-size: 18px;
}
li{
    color: black; 
    font-size: 25px;
}  
body {
    color: #E0F505; 
    font-size: 20px;
}
</style>
'''


st.markdown(text_color_css, unsafe_allow_html=True)



st.cache_data.clear()
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        self._to_linear = None
        self.convs = self._create_conv_layers()
        self._get_flattened_size()
        self.classifier = nn.Sequential(
            nn.Linear(self._to_linear, 4096),
            nn.ReLU(),
            nn.Linear(4096, num_classes)  
        )
    def _create_conv_layers(self):
        layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        return layers
    
    def _get_flattened_size(self):
        # Create a dummy tensor to calculate the output size after the convolutions
        with torch.no_grad():
            dummy_input = torch.ones(1, 3, 128, 128)  # Batch size of 1, 3 channels, 128x128 image
            dummy_output = self.features(dummy_input)
            self._to_linear = dummy_output.numel()  # Number of elements in the flattened tensor

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.classifier(x)
        return x


# Defining model architectures
def get_model(name, num_classes=61):
    if name == "ResNet":
        model = models.resnet50(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, num_classes)  # Update the final layer
    elif name == "VGG16":
        model = models.vgg16(pretrained=False)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)  # Update the final layer
    elif name == "VGG19": 
        model = models.vgg19(pretrained=False)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)  # Update the final layer
    elif name == "AlexNet":
        model = models.alexnet(pretrained=False)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)  # Update the final layer
    elif name == "DenseNet":
        model = models.densenet121(pretrained=False)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)  # Update the final layer
    elif name == "EfficientNet":
        model = EfficientNet.from_pretrained('efficientnet-b0')
        model._fc = nn.Linear(model._fc.in_features, num_classes)
    elif name == "CNN":
        model = SimpleCNN(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model name: {name}")
    return model

# model loading
def load_model(name, model_path):
    model = get_model(name)
    state_dict = torch.load(model_path)


    if name == "ResNet":
        aux_keys = [k for k in state_dict.keys() if "AuxLogits" in k]
        for k in aux_keys:
            del state_dict[k]
    
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model

models_paths = {
    "ResNet": "resnet50.pth",
    "VGG16": "vgg16.pth",
    "VGG19": "vgg19.pth",
    "AlexNet": "alexnet.pth",
    "DenseNet": "densenet.pth",
    "EfficientNet": "efficientnet.pth",
    "CNN": "basic_cnn1.pth"
}

models = {name: load_model(name, path) for name, path in models_paths.items()}

# Load class names
class_names = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Cauliflower_Bacterial spot rot', 'Cauliflower_Black Rot', 'Cauliflower_Downy Mildew', 'Cauliflower_Healthy', 
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy',
    'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
    'Guava_Healthy', 'Guava_Phytopthora', 'Guava_Red rust', 'Guava_Scab', 'Guava_Styler and Root',
    'Mango_Anthracnose', 'Mango_Die Back', 'Mango_Gall Midge', 'Mango_Healthy', 'Mango_Sooty Mould',
    'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
    'Pumpkin_Bacterial Leaf Spot', 'Pumpkin_Healthy', 'Pumpkin_Mosaic Disease', 'Pumpkin_Powdery_Mildew',
    'Rice_Bacterial_blight', 'Rice___Brown_Spot', 'Rice___Healthy', 'Rice___Leaf_Blast',
    'Soybean_Bacterial Pustule', 'Soybean_Frogeye Leaf Spot', 'Soybean_Sudden Death Syndrome', 'Soybean_Target Leaf Spot', 'Soybean_Yellow Mosaic', 'Soybean___healthy',
    'Strawberry_Anthracnose', 'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Strawberry_leaf_spot', 'Streawberry_ang_leaf_spot', 
    'Sugarcane_Healthy', 'Sugarcane_Red Rot', 'Sugarcane_Rust', 'Sugarcane_Yellow',
    'Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Septoria_leaf_spot', 'Tomato___healthy',
    'Wheat___Brown_Rust', 'Wheat___Healthy', 'Wheat___Yellow_Rust',
]
disease_info = {
    'Apple___Apple_scab': {
        'description': "Apple scab is a fungal disease that affects apple trees.",
        'symptoms': "Dark, sunken lesions on apples and leaves, with a rough texture.",
        'prevention_measures': "Use resistant varieties, apply fungicides, and remove fallen leaves.",
        'medicines': "Fungicides such as Captan or Fungicide containing Mancozeb can be used."
    },
    'Apple___Black_rot': {
        'description': "Black rot is a fungal disease that affects apples.",
        'symptoms': "Dark, sunken lesions on fruit and leaves, leading to decay.",
        'prevention_measures': "Remove infected fruit and leaves, use fungicides.",
        'medicines': "Fungicides such as Thiophanate-methyl can be used."
    },
    'Apple___Cedar_apple_rust': {
        'description': "Cedar apple rust is a fungal disease that affects apple trees.",
        'symptoms': "Orange, gelatinous growths on leaves and fruit.",
        'prevention_measures': "Use resistant varieties, remove cedar trees near apple trees.",
        'medicines': "Fungicides such as Chlorothalonil can be used."
    },
    'Apple___healthy': {
        'description': "Healthy apple trees with no visible disease symptoms.",
        'symptoms': "No symptoms, healthy growth.",
        'prevention_measures': "Regular care, proper watering, and fertilization.",
        'medicines': "No treatment needed."
    },
    'Cauliflower_Bacterial spot rot': {
        'description': "Bacterial spot rot is a bacterial disease that affects cauliflower.",
        'symptoms': "Water-soaked spots on leaves and curds, leading to rot.",
        'prevention_measures': "Avoid overhead watering, use resistant varieties.",
        'medicines': "Copper-based bactericides can be used."
    },
    'Cauliflower_Black Rot': {
        'description': "Black rot is a bacterial disease that affects cauliflower.",
        'symptoms': "Dark, V-shaped lesions on leaf margins, leading to leaf death.",
        'prevention_measures': "Practice crop rotation, use resistant varieties.",
        'medicines': "Copper-based bactericides can be used."
    },
    'Cauliflower_Downy Mildew': {
        'description': "Downy mildew is a fungal disease that affects cauliflower.",
        'symptoms': "Yellowing and curling of leaves, with gray, downy fungal growth.",
        'prevention_measures': "Ensure good air circulation, avoid overhead watering.",
        'medicines': "Fungicides such as Mancozeb or Chlorothalonil can be used."
    },
    'Cauliflower_Healthy': {
        'description': "Healthy cauliflower plants with no visible disease symptoms.",
        'symptoms': "No symptoms, healthy growth.",
        'prevention_measures': "Regular care, proper watering, and fertilization.",
        'medicines': "No treatment needed."
    },
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot': {
        'description': "Cercospora leaf spot and gray leaf spot are fungal diseases that affects corn.",
        'symptoms': "Gray or brown lesions with yellow halos on leaves.",
        'prevention_measures': "Use resistant varieties, avoid overhead watering.",
        'medicines': "Fungicides such as Chlorothalonil can be used."
    },
    'Corn_(maize)___Common_rust_': {
        'description': "Common rust is a fungal disease that affects corn.",
        'symptoms': "Rusty, reddish-brown pustules on leaves and stems.",
        'prevention_measures': "Use resistant varieties, apply fungicides if necessary.",
        'medicines': "Fungicides such as Propiconazole can be used."
    },
    'Corn_(maize)___Northern_Leaf_Blight': {
        'description': "Northern leaf blight is a fungal disease that affects corn.",
        'symptoms': "Large, grayish-green lesions on leaves with dark borders.",
        'prevention_measures': "Use resistant varieties, practice crop rotation.",
        'medicines': "Fungicides such as Tebuconazole can be used."
    },
    'Corn_(maize)___healthy': {
        'description': "Healthy corn plants with no visible disease symptoms.",
        'symptoms': "No symptoms, healthy growth.",
        'prevention_measures': "Regular care, proper watering, and fertilization.",
        'medicines': "No treatment needed."
    },
    'Grape___Black_rot': {
        'description': "Black rot is a fungal disease that affects grapes.",
        'symptoms': "Dark, sunken lesions on grapes and leaves.",
        'prevention_measures': "Remove infected fruit and leaves, use fungicides.",
        'medicines': "Fungicides such as Captan or Copper-based sprays can be used."
    },
    'Grape___Esca_(Black_Measles)': {
        'description': "Esca (Black Measles) is a fungal disease that affects grapes.",
        'symptoms': "Dark streaks and lesions on leaves, wood, and fruit.",
        'prevention_measures': "Remove and destroy infected plant material.",
        'medicines': "No specific treatment; management focuses on sanitation and pruning."
    },
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)': {
        'description': "Leaf blight (Isariopsis Leaf Spot) is a fungal disease that affects grapes.",
        'symptoms': "Round, brown lesions with a yellow halo on leaves.",
        'prevention_measures': "Ensure good air circulation, avoid overhead watering.",
        'medicines': "Fungicides such as Mancozeb or Chlorothalonil can be used."
    },
    'Grape___healthy': {
        'description': "Healthy grapevines with no visible disease symptoms.",
        'symptoms': "No symptoms, healthy growth.",
        'prevention_measures': "Regular care, proper watering, and fertilization.",
        'medicines': "No treatment needed."
    },
    'Guava_Healthy': {
        'description': "Healthy guava plants with no visible disease symptoms.",
        'symptoms': "No symptoms, healthy growth.",
        'prevention_measures': "Regular care, proper watering, and fertilization.",
        'medicines': "No treatment needed."
    },
    'Guava_Phytopthora': {
        'description': "Phytophthora is a fungal disease that affects guava plants.",
        'symptoms': "Water-soaked lesions on leaves and fruit, leading to rot.",
        'prevention_measures': "Ensure good drainage, avoid overhead watering.",
        'medicines': "Fungicides such as Metalaxyl can be used."
    },
    'Guava_Red rust': {
        'description': "Red rust is a fungal disease that affects guava plants.",
        'symptoms': "Red, rust-like lesions on leaves and stems.",
        'prevention_measures': "Use resistant varieties, apply fungicides.",
        'medicines': "Fungicides such as Chlorothalonil can be used."
    },
    'Guava_Scab': {
        'description': "Scab is a fungal disease that affects guava plants.",
        'symptoms': "Dark, scabby lesions on leaves and fruit.",
        'prevention_measures': "Remove infected plant material, use fungicides.",
        'medicines': "Fungicides such as Captan or Mancozeb can be used."
    },
    'Guava_Styler and Root': {
        'description': "Styler and Root rot is a fungal disease that affects guava plants.",
        'symptoms': "Rotting of roots and stems, with stunted growth.",
        'prevention_measures': "Ensure good drainage, avoid waterlogging.",
        'medicines': "Fungicides such as Metalaxyl or Copper-based sprays can be used."
    },
    'Mango_Anthracnose': {
        'description': "Anthracnose is a fungal disease that affects mango trees.",
        'symptoms': "Dark, sunken lesions on fruit, leaves, and stems.",
        'prevention_measures': "Remove and destroy infected plant material, apply fungicides.",
        'medicines': "Fungicides such as Mancozeb or Copper-based sprays can be used."
    },
    'Mango_Die Back': {
        'description': "Dieback is a fungal disease that affects mango trees.",
        'symptoms': "Wilting and dieback of branches and shoots.",
        'prevention_measures': "Remove infected branches, improve air circulation.",
        'medicines': "Fungicides such as Copper-based sprays or systemic fungicides can be used."
    },
    'Mango_Gall Midge': {
        'description': "Gall midge is an insect pest that affects mango trees.",
        'symptoms': "Deformed leaves and fruit galls.",
        'prevention_measures': "Use insecticides, remove infected plant material.",
        'medicines': "Insecticides such as Imidacloprid can be used."
    },
    'Mango_Healthy': {
        'description': "Healthy mango trees with no visible disease symptoms.",
        'symptoms': "No symptoms, healthy growth.",
        'prevention_measures': "Regular care, proper watering, and fertilization.",
        'medicines': "No treatment needed."
    },
    'Mango_Sooty Mould': {
        'description': "Sooty mould is a fungal disease that affects mango trees.",
        'symptoms': "Black, sooty mold on leaves and fruit, usually secondary to honeydew from insects.",
        'prevention_measures': "Control insect pests that produce honeydew.",
        'medicines': "Fungicides are not effective; control insect pests."
    },
    'Potato___Early_blight': {
        'description': "Early blight is a fungal disease that affects potatoes.",
        'symptoms': "Dark, concentric lesions on leaves, leading to premature leaf drop.",
        'prevention_measures': "Use resistant varieties, apply fungicides.",
        'medicines': "Fungicides such as Chlorothalonil or Mancozeb can be used."
    },
    'Potato___Late_blight': {
        'description': "Late blight is a fungal disease that affects potatoes.",
        'symptoms': "Large, irregular, water-soaked lesions on leaves and tubers.",
        'prevention_measures': "Use resistant varieties, apply fungicides, avoid overhead watering.",
        'medicines': "Fungicides such as Metalaxyl or Mefenoxam can be used."
    },
    'Potato___healthy': {
        'description': "Healthy potato plants with no visible disease symptoms.",
        'symptoms': "No symptoms, healthy growth.",
        'prevention_measures': "Regular care, proper watering, and fertilization.",
        'medicines': "No treatment needed."
    },
    'Pumpkin_Bacterial Leaf Spot': {
        'description': "Bacterial leaf spot is a bacterial disease that affects pumpkins.",
        'symptoms': "Water-soaked spots on leaves that turn brown and dry out.",
        'prevention_measures': "Avoid overhead watering, use resistant varieties.",
        'medicines': "Copper-based bactericides can be used."
    },
    'Pumpkin_Healthy': {
        'description': "Healthy pumpkin plants with no visible disease symptoms.",
        'symptoms': "No symptoms, healthy growth.",
        'prevention_measures': "Regular care, proper watering, and fertilization.",
        'medicines': "No treatment needed."
    },
    'Pumpkin_Mosaic Disease': {
        'description': "Mosaic disease is a viral disease that affects pumpkins.",
        'symptoms': "Mosaic patterns of light and dark green on leaves.",
        'prevention_measures': "Control insect vectors, use resistant varieties.",
        'medicines': "No effective treatment; management focuses on vector control."
    },
    'Pumpkin_Powdery_Mildew': {
        'description': "Powdery mildew is a fungal disease that affects pumpkins.",
        'symptoms': "White, powdery fungal growth on leaves and stems.",
        'prevention_measures': "Ensure good air circulation, use resistant varieties.",
        'medicines': "Fungicides such as Sulfur or Potassium bicarbonate can be used."
    },
    'Rice_Bacterial_blight': {
        'description': "Bacterial blight is a bacterial disease that affects rice.",
        'symptoms': "Water-soaked lesions on leaves that turn yellow and dry out.",
        'prevention_measures': "Use resistant varieties, avoid overhead watering.",
        'medicines': "Copper-based bactericides can be used."
    },
    'Rice___Brown_Spot': {
        'description': "Brown spot is a fungal disease that affects rice.",
        'symptoms': "Brown, circular lesions with a yellow halo on leaves.",
        'prevention_measures': "Use resistant varieties, apply fungicides.",
        'medicines': "Fungicides such as Mancozeb can be used."
    },
    'Rice___Healthy': {
        'description': "Healthy rice plants with no visible disease symptoms.",
        'symptoms': "No symptoms, healthy growth.",
        'prevention_measures': "Regular care, proper watering, and fertilization.",
        'medicines': "No treatment needed."
    },
    'Rice___Leaf_Blast': {
        'description': "Leaf blast is a fungal disease that affects rice.",
        'symptoms': "Elliptical lesions with gray centers and dark borders on leaves.",
        'prevention_measures': "Use resistant varieties, apply fungicides.",
        'medicines': "Fungicides such as Pyricularia or Carbendazim can be used."
    },
    'Soybean_Bacterial Pustule': {
        'description': "Bacterial pustule is a bacterial disease that affects soybeans.",
        'symptoms': "Raised pustules on leaves and stems, which eventually turn yellow.",
        'prevention_measures': "Use resistant varieties, avoid overhead watering.",
        'medicines': "Copper-based bactericides can be used."
    },
    'Soybean_Frogeye Leaf Spot': {
        'description': "Frogeye leaf spot is a fungal disease that affects soybeans.",
        'symptoms': "Round, gray spots with dark borders on leaves.",
        'prevention_measures': "Use resistant varieties, apply fungicides.",
        'medicines': "Fungicides such as Mancozeb can be used."
    },
    'Soybean_Sudden Death Syndrome': {
        'description': "Sudden Death Syndrome is a fungal disease that affects soybeans.",
        'symptoms': "Wilting and yellowing of leaves, with a sudden death of plants.",
        'prevention_measures': "Use resistant varieties, avoid excessive soil moisture.",
        'medicines': "No effective treatment; management focuses on resistance and soil management."
    },
    'Soybean_Target Leaf Spot': {
        'description': "Target leaf spot is a fungal disease that affects soybeans.",
        'symptoms': "Round, dark spots with concentric rings on leaves.",
        'prevention_measures': "Use resistant varieties, apply fungicides.",
        'medicines': "Fungicides such as Chlorothalonil can be used."
    },
    'Soybean_Yellow Mosaic': {
        'description': "Yellow mosaic is a viral disease that affects soybeans.",
        'symptoms': "Yellow, mosaic-like patterns on leaves.",
        'prevention_measures': "Control insect vectors, use resistant varieties.",
        'medicines': "No effective treatment; management focuses on vector control."
    },
    'Soybean___healthy': {
        'description': "Healthy soybean plants with no visible disease symptoms.",
        'symptoms': "No symptoms, healthy growth.",
        'prevention_measures': "Regular care, proper watering, and fertilization.",
        'medicines': "No treatment needed."
    },
    'Strawberry_Anthracnose': {
        'description': "Anthracnose is a fungal disease that affects strawberries.",
        'symptoms': "Dark, sunken lesions on fruit and leaves.",
        'prevention_measures': "Remove infected fruit and leaves, use fungicides.",
        'medicines': "Fungicides such as Captan or Chlorothalonil can be used."
    },
    'Strawberry___Leaf_scorch': {
        'description': "Leaf scorch is a disease that affects strawberries.",
        'symptoms': "Brown, scorched edges on leaves.",
        'prevention_measures': "Avoid overhead watering, ensure proper drainage.",
        'medicines': "Fungicides are not effective; management focuses on cultural practices."
    },
    'Strawberry___healthy': {
        'description': "Healthy strawberry plants with no visible disease symptoms.",
        'symptoms': "No symptoms, healthy growth.",
        'prevention_measures': "Regular care, proper watering, and fertilization.",
        'medicines': "No treatment needed."
    },
    'Strawberry_leaf_spot': {
        'description': "Leaf spot is a fungal disease that affects strawberries.",
        'symptoms': "Round, dark spots on leaves.",
        'prevention_measures': "Remove infected leaves, apply fungicides.",
        'medicines': "Fungicides such as Mancozeb can be used."
    },
    'Streawberry_ang_leaf_spot': {
        'description': "Angular leaf spot is a fungal disease that affects strawberries.",
        'symptoms': "Angular lesions on leaves with a yellow halo.",
        'prevention_measures': "Ensure good air circulation, avoid overhead watering.",
        'medicines': "Fungicides such as Copper-based sprays can be used."
    },
    'Sugarcane_Healthy': {
        'description': "Healthy sugarcane plants with no visible disease symptoms.",
        'symptoms': "No symptoms, healthy growth.",
        'prevention_measures': "Regular care, proper watering, and fertilization.",
        'medicines': "No treatment needed."
    },
    'Sugarcane_Red Rot': {
        'description': "Red rot is a fungal disease that affects sugarcane.",
        'symptoms': "Red, water-soaked lesions on stalks.",
        'prevention_measures': "Use resistant varieties, avoid waterlogging.",
        'medicines': "Fungicides such as Carbendazim can be used."
    },
    'Sugarcane_Rust': {
        'description': "Rust is a fungal disease that affects sugarcane.",
        'symptoms': "Rusty pustules on leaves and stems.",
        'prevention_measures': "Use resistant varieties, apply fungicides.",
        'medicines': "Fungicides such as Triadimefon can be used."
    },
    'Sugarcane_Yellow': {
        'description': "Yellow disease is a condition that affects sugarcane.",
        'symptoms': "Yellowing of leaves and stunted growth.",
        'prevention_measures': "Use resistant varieties, avoid nutrient deficiencies.",
        'medicines': "No specific treatment; management focuses on proper nutrition."
    },
    'Tomato_Yellow_Leaf_Curl_Virus': {
        'description': "Yellow Leaf Curl Virus is a viral disease that affects tomatoes.",
        'symptoms': "Curling and yellowing of leaves, stunted growth.",
        'prevention_measures': "Control insect vectors, use resistant varieties.",
        'medicines': "No effective treatment; management focuses on vector control."
    },
    'Tomato___Bacterial_spot': {
        'description': "Bacterial spot is a bacterial disease that affects tomatoes.",
        'symptoms': "Water-soaked spots on leaves and fruit.",
        'prevention_measures': "Use resistant varieties, avoid overhead watering.",
        'medicines': "Copper-based bactericides can be used."
    },
    'Tomato___Early_blight': {
        'description': "Early blight is a fungal disease that affects tomatoes.",
        'symptoms': "Dark, concentric lesions on leaves and fruit.",
        'prevention_measures': "Use resistant varieties, apply fungicides.",
        'medicines': "Fungicides such as Chlorothalonil can be used."
    },
    'Tomato___Late_blight': {
        'description': "Late blight is a fungal disease that affects tomatoes.",
        'symptoms': "Large, irregular, water-soaked lesions on leaves and fruit.",
        'prevention_measures': "Use resistant varieties, apply fungicides.",
        'medicines': "Fungicides such as Metalaxyl or Mefenoxam can be used."
    },
    'Tomato___Septoria_leaf_spot': {
        'description': "Septoria leaf spot is a fungal disease that affects tomatoes.",
        'symptoms': "Small, dark spots with white centers on leaves.",
        'prevention_measures': "Use resistant varieties, apply fungicides.",
        'medicines': "Fungicides such as Chlorothalonil can be used."
    },
    'Tomato___healthy': {
        'description': "Healthy tomato plants with no visible disease symptoms.",
        'symptoms': "No symptoms, healthy growth.",
        'prevention_measures': "Regular care, proper watering, and fertilization.",
        'medicines': "No treatment needed."
    },
    'Wheat___Brown_Rust': {
        'description': "Brown rust is a fungal disease that affects wheat.",
        'symptoms': "Brown, powdery pustules on leaves.",
        'prevention_measures': "Use resistant varieties, apply fungicides.",
        'medicines': "Fungicides such as Propiconazole can be used."
    },
    'Wheat___Healthy': {
        'description': "Healthy wheat plants with no visible disease symptoms.",
        'symptoms': "No symptoms, healthy growth.",
        'prevention_measures': "Regular care, proper watering, and fertilization.", 
        'medicines': "No treatment needed."
    },
    'Wheat___Yellow_Rust': {
        'description': "Yellow rust is a fungal disease that affects wheat.",
        'symptoms': "Yellow, powdery pustules on leaves.",
        'prevention_measures': "Use resistant varieties, apply fungicides.",
        'medicines': "Use Fungicides such as Tebuconazole."
    }
}






# Threshold for confidence 
CONFIDENCE_THRESHOLD = 0.9


# List of plants 
plants_list = [
    "Corn", "Tomato", "Apple", "Potato", "Strawberry",
    "Soybean", "Wheat", "Rice", "Mango", "Sugarcane",
    "Cauliflower", "Pumpkin", "Grape", "Guava"
]



# Defining transforms for image preprocessing
preprocess = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def predict(image, models):
    inputs = preprocess(image).unsqueeze(0)
    results = {}
    
    for name, model in models.items():
        model.eval()  
        with torch.no_grad():
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            confidence = torch.nn.functional.softmax(outputs, dim=1).max().item()
            results[name] = (class_names[preds[0]], confidence)
    
    # To select model with the highest confidence
    best_model = max(results, key=lambda x: results[x][1])
    return best_model, results[best_model]




def is_leaf_image(image):
    return True


with st.sidebar:
    selected = option_menu(
    menu_title="Menu",
    options=["Home", "Prediction", "About Us", "Information", "Contact/Help" ],
    icons=["house-fill", "search-heart-fill", "people-fill", "exclude", "telephone-fill"],
    styles={
    "container": {"padding": "0!important", "background_img": "1.jpeg", "opacity": 1.0},
    "icon": {"color": "orange", "font-size": "15px"},
    "nav-link": {
            "font-size": "12px",
            "text-align": "left",
            "margin": "1px",
            "--hover-color": "#475360",
            },
                "nav-link-selected": {
                "background": "rgba(0, 0, 0, 0) !important", 
                "color": "#ffffff",  
    }
    }
    )




if selected == "Home":
    st.markdown(f"<h1 class='custom-header'>Comparative Analysis of Deep Learning Models for Plant Leaf Disease Classification</h1>", unsafe_allow_html=True)
    image_path = "4.jpeg"
    st.image(image_path, use_column_width=True)
    st.markdown(""" 
    WELCOME TO OUR SYSTEM! 🌿🔍
            
    Use the sidebar to navigate through the features.

    Our mission is to help in identifying plant diseases efficiently. Upload an image of a plant, and our system will analyze it to detect any signs of diseases. Together, let's protect our plants and ensure a healthier harvest!

    ### How It Works
    - **Upload Image:** Go to the **Prediction** page and upload an image of a plant with suspected diseases.
    - **Analysis:** Our system will process the image using advanced deep learning models to identify potential diseases.
    - **Results:** View the results and recommendations for further action.

    ### Why Choose Us?
    - **Accuracy:** Our system utilizes state-of-the-art deep learning techniques for accurate disease detection.
    - **User-Friendly:** Simple and intuitive interface for seamless user experience.
    - **Fast and Efficient:** Receive results in seconds, allowing for quick decision-making.

    ### Get Started
    Click on the **Prediction** page in the sidebar to upload an image and experience the power of our Plant Leaf Disease Detection System!

    ### About Us
    Learn more about the project, our team, and our goals on the **About Us** page.
    """)

if selected == "Prediction":
    st.header("Upload Image")

  # Instructions for using the system
    st.markdown("""
    <div style="padding: 10px; border-radius: 5px; font-size: 14px; font-style: italic;">
        <h4><b><u>Instructions for Using Our System</u></b></h4>
        <ul>
            <li><b>Please ensure that the images you upload are of the leaves of the following plants:</b></li>
            <ul><b> 
                <li>Corn</li>
                <li>Tomato</li>
                <li>Apple</li>
                <li>Potato</li>
                <li>Strawberry</li>
                <li>Soybean</li>
                <li>Wheat</li>
                <li>Rice</li>
                <li>Mango</li>
                <li>Sugarcane</li>
                <li>Cauliflower</li>
                <li>Pumpkin</li>
                <li>Grape</li>
                <li>Guava</li>
           </b></ul>
            <li><b>This system is specifically designed to classify diseases for these plants only.</li>
            <li><b>For accurate results, make sure that the leaf is clearly visible in the image, free from any obstructions, and well-lit.</li>
            <li><b>Uploading images of leaves from plants not listed above may lead to incorrect or failed detection.</li>
        </ul>
        <p><b>Thank you for using this system!</b></p>
    </div>
    """, unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "png", "jpeg"], key="upload_image")

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption='Uploaded Image', use_column_width=True)
        
        if is_leaf_image(image):
            model_name, (predicted_class, confidence) = predict(image, models)
            if confidence >= CONFIDENCE_THRESHOLD:
                st.snow()
                st.write(f"Based on the {model_name}, this leaf is classified as '{predicted_class}' with a confidence of {confidence*100:.2f}%.")
               
                if predicted_class in disease_info:
                    description = disease_info[predicted_class]['description']
                    symptoms = disease_info[predicted_class]['symptoms']
                    prevention = disease_info[predicted_class]['prevention_measures']
                    medicines = disease_info[predicted_class]['medicines']
    
                    st.subheader("Disease Information")
                    st.write(f"**Description:** {description}")
                    st.write(f"**Symptoms:** {symptoms}")
                    st.write(f"**Prevention Measures:** {prevention}")
                    st.write(f"**Medicines:** {medicines}")
                else:
                    st.write("No additional information available for this class.")

                st.session_state['prediction'] = predicted_class
                st.session_state.prediction_done = True

                if st.session_state.prediction_done:
                    st.subheader("Select Models for Comparison")
                    selected_models = st.multiselect(
                        "Choose models to compare predictions:",
                        options=list(models.keys())
                    )
                     
                    if selected_models:
                        st.write("Comparing predictions with selected models...")
                        comparison_results = {}
                        for model_name in selected_models:
                            model, (predicted_class, confidence) = predict(image, {model_name: models[model_name]})
                            comparison_results[model_name] = (predicted_class, confidence)
                        
                        for model_name, (predicted_class, confidence) in comparison_results.items():
                            st.write(f"Model: {model_name}")
                            st.write(f"Prediction: {predicted_class} with confidence: {confidence*100:.2f}%")
                
            else:
                st.write("The confidence is too low. The image may not be a leaf.")
        else:
            st.write("The image does not appear to be a leaf.")



if selected == "About Us":
    st.markdown(f"<h1 class='custom-header' style='color:black;'>About Us</h1>", unsafe_allow_html=True)
    st.markdown("""
<p style='font-size:20px; color:black;'>
We are a team of five passionate computer engineering students from Khwopa Engineering College, currently in our 8th semester, dedicated to developing a comparative analysis of deep learning model for plant leaf disease classification system. Our team consists of:
</p>

<ul style='font-size:20px; color:black;'>
  <li style='font-size:20px;'><strong>Aashish Pandey:</strong> 760302</li>
  <li style='font-size:20px;'><strong>Ishan Bista:</strong> 760317</li>
  <li style='font-size:20px;'><strong>Nijal Kachhepati:</strong> 760323</li>
  <li style='font-size:20px;'><strong>Oman Neupane:</strong> 760324</li>
  <li style='font-size:20px;'><strong>Rakesh Kumar Chaudhary:</strong> 760330</li>
</ul>

<p style='font-size:20px; color:black;'>
Driven by a desire to contribute to the agricultural sector, we embarked on this project to create a user-friendly and effective tool for early classification of plant diseases. Our system leverages the power of deep learning to analyze images of leaves and identify potential diseases.
</p>

<h3 style='color:black'>Our Mission:</h3>

<p style='font-size:20px; color:black'>
Our mission is to empower farmers and agricultural professionals with an innovative and accessible technology that aids in early disease classification. By enabling prompt identification of plant diseases, our system aims to:
</p>

<ul style='font-size:20px; color:black'>
  <li style='font-size:20px;'>Increase crop yield and quality by minimizing damage caused by undetected diseases.</li>
  <li style='font-size:20px;'>Enhance agricultural sustainability by promoting preventative measures and optimizing resource use.</li>
</ul>

<h3 style='color:black;'>Our Approach:</h3>

<p style='font-size:20px; color:black'>
Our system is designed to be user-friendly and adaptable to various environments. We are committed to:
</p>

<ul style='font-size:20px; color:black'>
  <li style='font-size:20px;'>Developing a user interface that is intuitive and easy to navigate for users of all technical backgrounds.</li>
  <li style='font-size:20px;'>Training the system on a comprehensive dataset of plant leaf images encompassing a wide range of diseases to ensure accurate classification.</li>
  <li style='font-size:20px;'>Continuing to refine and improve the system's capabilities through ongoing research and development.</li>
</ul>

<h3 style='color:black;'>Our Vision:</h3>

<p style='font-size:18px; color:black'>
We envision a future where this system becomes an essential tool for farmers worldwide, contributing to a more productive, sustainable, and resilient agricultural sector.
</p>

<h3 style='color:black;'>About Dataset</h3>

<p style='font-size:18px; color:black'>
This dataset is recreated using offline augmentation and adding new set of images to the original dataset. The original dataset can be found on this GitHub repo.
This dataset consists of about 100K RGB images of healthy and diseased crop leaves categorized into 61 different classes. The total dataset is divided into 80/20 ratio of training and validation set preserving the directory structure.
A new directory containing 305 test images is created later for prediction purposes.
</p>

<h4 style='font-size:25px; color:black'>Content</h4>
<ul style='font-size:16px; color:black'>
  <li style='font-size:20px;'>Train (80327 images)</li>
  <li style='font-size:20px;'>Test (305 images)</li>
  <li style='font-size:20px;'>Validation (24364 images)</li>
</ul>
""", unsafe_allow_html=True)


if selected == "Information":
    st.markdown(f"<h1 class='custom-header'>Disease Information</h1>", unsafe_allow_html=True)
    st.write("Here you can find detailed information about various plants, including their diseases, symptoms, causes, and preventive measures.")
    st.write("*Please note that this information is general. For specific guidance, consult a local agricultural extension office or plant pathologist.*")
      # Apple Leaf Diseases Information
    st.subheader("#Apple Leaf Diseases")
    
    # Apple Scab
    st.write("### 1. Apple Scab")
    st.write("**Symptoms:**")
    st.write("- Olive-green to black velvety lesions on leaves, fruit, and young shoots.")
    st.write("- Infected leaves may become twisted or puckered.")
    st.write("- Severe infections can cause early leaf drop and reduced fruit quality.")
    st.write("**Preventive Measures:**")
    st.write("- **Resistant Varieties:** Plant apple varieties that are resistant to apple scab.")
    st.write("- **Sanitation:** Remove and destroy fallen leaves and fruit to reduce the source of inoculum.")
    st.write("- **Pruning:** Prune trees to improve air circulation and reduce humidity within the canopy.")
    st.write("**Medicines:**")
    st.write("Apply fungicides such as captan, mancozeb, or myclobutanil during the growing season, especially during periods of wet weather. Follow recommended application schedules and intervals.")
    
    # Powdery Mildew
    st.write("### 2. Powdery Mildew")
    st.write("**Symptoms:**")
    st.write("- White, powdery fungal growth on leaves, shoots, and buds.")
    st.write("- Infected leaves may become distorted and curled.")
    st.write("- Severe infections can reduce photosynthesis, leading to stunted growth and poor fruit quality.")
    st.write("**Preventive Measures:**")
    st.write("- **Resistant Varieties:** Choose apple varieties that are resistant to powdery mildew.")
    st.write("- **Sanitation:** Remove and destroy infected plant parts to reduce the spread of the disease.")
    st.write("- **Pruning:** Prune trees to enhance air circulation and reduce humidity within the canopy.")
    st.write("**Medicines:**")
    st.write("Apply fungicides such as sulfur, potassium bicarbonate, or myclobutanil. Use these fungicides at the recommended rates and intervals, particularly during periods of high humidity and warm temperatures.")
    
    # Cedar Apple Rust
    st.write("### 3. Cedar Apple Rust")
    st.write("**Symptoms:**")
    st.write("- Bright yellow-orange spots on the upper surface of leaves.")
    st.write("- Tubular structures (aecia) form on the underside of leaves.")
    st.write("- Severe infections can cause leaf drop, reducing the tree's vigor and fruit production.")
    st.write("**Preventive Measures:**")
    st.write("- **Resistant Varieties:** Plant resistant apple varieties.")
    st.write("- **Remove Alternate Hosts:** Remove nearby juniper or cedar trees, which are the alternate hosts for the rust fungus.")
    st.write("- **Pruning:** Prune trees to improve air circulation and reduce humidity around the foliage.")
    st.write("**Medicines:**")
    st.write("Apply fungicides such as myclobutanil or propiconazole. Begin applications in early spring before infection occurs and continue at regular intervals, especially during wet conditions.")
    
    # Apple Black Rot
    st.write("### 4. Apple Black Rot")
    st.write("**Symptoms:**")
    st.write("- **Leaf Spots:** Initial symptoms include small, purple spots on the leaves. These spots gradually enlarge and turn brown, often with a purple border.")
    st.write("- **Leaf Blight:** Infected leaves may develop extensive blight, causing them to die and fall prematurely.")
    st.write("- **Fruit Rot:** In advanced stages, the disease can also affect fruits, causing black, sunken lesions, often with concentric rings.")
    st.write("- **Twig Canker:** Black rot can cause cankers on twigs and branches, leading to dieback and further infection spread.")
    st.write("**Preventive Measures:**")
    st.write("- **Sanitation:** Remove and destroy fallen leaves, mummified fruits, and pruned branches to reduce sources of inoculum. Regularly clean up orchard debris and maintain good sanitation practices.")
    st.write("- **Pruning:** Prune infected twigs and branches during the dormant season to remove cankers and improve air circulation. Avoid leaving stubs and make clean cuts to promote rapid healing.")
    st.write("- **Resistant Varieties:** Plant apple varieties that are resistant or less susceptible to black rot. Consult local extension services for recommendations on resistant varieties suitable for your region.")
    st.write("- **Water Management:** Avoid overhead irrigation, as it can create favorable conditions for spore dispersal and infection. Water at the base of trees to keep foliage dry and reduce disease pressure.")
    st.write("**Treatment:**")
    st.write("- **Fungicides:** Apply protective fungicides during key periods of the growing season, particularly during wet and humid conditions. Commonly used fungicides include captan, myclobutanil, and thiophanate-methyl. Follow label instructions for application rates and timing.")
    st.write("- **Biological Control:** Utilize beneficial microorganisms, such as Bacillus subtilis, which can help suppress fungal pathogens and reduce infection. Encourage natural predators of the fungus by maintaining a healthy orchard ecosystem.")
    st.write("- **Nutrient Management:** Maintain balanced tree nutrition to promote healthy growth and resilience against diseases. Avoid excessive nitrogen fertilization, as it can lead to lush growth that is more susceptible to infection.")
    st.write("- **Cultural Practices:** Implement crop rotation and avoid planting apple trees in areas with a history of black rot. Use disease-free planting material and certified rootstocks to reduce the risk of introducing the pathogen.")
    
    
    # Tomato Leaf Diseases
    st.subheader("#Tomato Leaf Diseases")
    
    st.write("### 1. Early Blight")
    st.write("**Symptoms:**")
    st.write("- Dark, concentric ring spots on older leaves.")
    st.write("- Yellowing of leaves starting from the bottom of the plant.")
    st.write("- Leaf drop leading to reduced yield and sunscald on fruits.")
    
    st.write("**Preventive Measures:**")
    st.write("- **Cultural Practices:** Rotate crops and avoid planting tomatoes in the same spot year after year. Mulch around plants to reduce soil splashing onto leaves.")
    st.write("- **Resistant Varieties:** Plant disease-resistant varieties if available.")
    st.write("- **Sanitation:** Remove and destroy infected plant debris to reduce the source of infection.")
    
    st.write("**Medicines:**")
    st.write("Use fungicides like chlorothalonil, mancozeb, or copper-based products. Apply these at the recommended rates and intervals, especially during warm, humid conditions.")
    
    st.write("### 2. Late Blight")
    st.write("**Symptoms:**")
    st.write("- Large, water-soaked lesions on leaves and stems, often with a fuzzy white growth on the underside of leaves.")
    st.write("- Rapid collapse and blackening of infected plant parts.")
    st.write("- Rotting of fruits, especially during wet weather.")
    
    st.write("**Preventive Measures:**")
    st.write("- **Cultural Practices:** Avoid overhead watering to keep foliage dry. Ensure good air circulation by spacing plants adequately.")
    st.write("- **Resistant Varieties:** Plant late blight-resistant tomato varieties.")
    st.write("- **Sanitation:** Remove and destroy infected plant debris to reduce the source of infection.")
    
    st.write("**Medicines:**")
    st.write("Use systemic fungicides like metalaxyl or mefenoxam, and contact fungicides like copper-based products. Apply according to the manufacturer's instructions, particularly during periods of high humidity.")
    
    st.write("### 3. Tomato Mosaic Virus (TMV)")
    st.write("**Symptoms:**")
    st.write("- Mosaic-like mottling and discoloration of leaves, with light and dark green areas.")
    st.write("- Leaf distortion and reduced plant growth.")
    st.write("- Poor fruit development and quality.")
    
    st.write("**Preventive Measures:**")
    st.write("- **Resistant Varieties:** Use TMV-resistant tomato varieties.")
    st.write("- **Sanitation:** Disinfect tools and equipment to prevent virus spread.")
    st.write("- **Control:** Remove and destroy infected plants to prevent further spread.")
    
    st.write("**Medicines:**")
    st.write("There are no chemical treatments for viral diseases. Prevention through resistance and sanitation is key.")
    
    st.write("### 4. Leaf Mold")
    st.write("**Symptoms:**")
    st.write("- Velvety, greenish-gray mold on the underside of leaves.")
    st.write("- Yellowing and necrosis of affected leaves.")
    st.write("- Reduced fruit quality and yield.")
    
    st.write("**Preventive Measures:**")
    st.write("- **Cultural Practices:** Avoid overhead watering and provide adequate spacing between plants to improve air circulation.")
    st.write("- **Resistant Varieties:** Use varieties resistant to leaf mold.")
    st.write("- **Sanitation:** Remove and destroy infected leaves and plant debris.")
    
    st.write("**Medicines:**")
    st.write("Use fungicides like azoxystrobin or pyraclostrobin. Apply these products according to the manufacturer's guidelines.")
    
    # Grape Leaf Diseases
    st.subheader("#Grape Leaf Diseases")
    st.write("### 1. Powdery Mildew")
    st.write("**Symptoms:**")
    st.write("- White to gray powdery growth on leaves, shoots, and berries.")
    st.write("- Distorted and stunted growth of young shoots and leaves.")
    st.write("- Reduced fruit quality and yield.")
    
    st.write("**Preventive Measures:**")
    st.write("- **Cultural Practices:** Ensure good air circulation by pruning and training vines properly. Avoid overhead irrigation to reduce leaf wetness.")
    st.write("- **Resistant Varieties:** Plant resistant grape varieties if available.")
    st.write("- **Sanitation:** Remove and destroy infected plant debris to reduce the source of infection.")
    
    st.write("**Medicines:**")
    st.write("Use sulfur-based fungicides or systemic fungicides like myclobutanil, triadimefon, or tebuconazole. Follow the recommended application rates and intervals to ensure effectiveness and reduce the risk of resistance development.")
    
    st.write("### 2. Downy Mildew")
    st.write("**Symptoms:**")
    st.write("- Yellowish, oily spots on the upper surface of leaves, which turn brown and necrotic.")
    st.write("- White, fluffy growth on the underside of leaves, especially in humid conditions.")
    st.write("- Infected berries may become brown, shriveled, and drop prematurely.")
    
    st.write("**Preventive Measures:**")
    st.write("- **Cultural Practices:** Improve air circulation by pruning and training vines. Avoid overhead irrigation and use drip irrigation to keep foliage dry.")
    st.write("- **Resistant Varieties:** Choose resistant or tolerant grape varieties if available.")
    st.write("- **Sanitation:** Remove and destroy infected plant debris to reduce the source of infection.")
    
    st.write("**Medicines:**")
    st.write("Use copper-based fungicides or systemic fungicides like metalaxyl, fosetyl-Al, or dimethomorph. Apply these fungicides at the recommended rates and intervals, particularly during periods of high humidity and wet conditions to protect the foliage and fruit.")
    
    st.write("### 3. Black Rot")
    st.write("**Symptoms:**")
    st.write("- Small, dark brown spots on leaves with a black border.")
    st.write("- Berries develop black, shriveled lesions.")
    
    st.write("**Preventive Measures:**")
    st.write("- **Resistant Varieties:** Use black rot-resistant grape varieties.")
    st.write("- **Crop Rotation:** Avoid planting grapes in the same location consecutively.")
    st.write("- **Sanitation:** Remove and destroy infected plant debris.")
    
    st.write("**Medicines:**")
    st.write("Use fungicides like myclobutanil or captan. Prune and destroy infected parts to reduce the spread.")
    
    st.write("### 4. Leaf Blight")
    st.write("**Symptoms:**")
    st.write("- **Lesions:** Small, water-soaked lesions on leaves that enlarge and turn brown or black.")
    st.write("- **Leaf Drop:** Severely infected leaves may fall off prematurely.")
    st.write("- **Discoloration:** Leaves may show yellowing around the lesions.")
    st.write("- **Vine Health:** Severe infections can weaken the vine, reducing yield and quality.")
    
    st.write("**Preventive Measures:**")
    st.write("- **Resistant Varieties:** Plant grape varieties known for their resistance to leaf blight.")
    st.write("- **Good Sanitation:** Remove and destroy fallen leaves and pruned parts to reduce inoculum sources.")
    st.write("- **Proper Spacing:** Ensure adequate spacing between plants to enhance air circulation and reduce humidity around the vines.")
    st.write("- **Irrigation Management:** Avoid overhead irrigation to prevent water from sitting on the leaves, which promotes fungal growth.")
    st.write("- **Regular Monitoring:** Inspect vines regularly for early signs of infection to manage outbreaks promptly.")
    
    st.write("**Treatment:**")
    st.write("- **Fungicides:** Use protectant fungicides such as mancozeb, chlorothalonil, or copper-based compounds as a preventive measure. Apply systemic fungicides like tebuconazole or myclobutanil for treating existing infections.")
    st.write("- **Biological Control:** Use biofungicides containing Bacillus subtilis or Trichoderma spp. as a preventive measure to inhibit pathogen growth.")
    st.write("- **Pruning and Removal:** Prune and remove infected leaves and shoots to prevent the spread. Disinfect pruning tools to avoid transferring the pathogen to healthy plants.")
    st.write("- **Integrated Pest Management (IPM):** Implement an IPM strategy that includes crop rotation, use of resistant varieties, and proper vineyard management to reduce disease pressure.")
    
    st.write("### 5. Grape Esca (Black Measles)")
    st.write("**Symptoms:**")
    st.write("- **Leaf Symptoms:** Leaves develop interveinal chlorosis (yellowing) with brown or black streaks and spots, creating a 'tiger stripe' appearance. Affected areas may become necrotic, leading to dead tissue and leaf drop.")
    st.write("- **Cane and Trunk Symptoms:** Cross-sections of affected canes and trunks show dark brown or black streaks in the wood, often accompanied by white rot. Affected vines exhibit dieback of shoots and canes.")
    st.write("- **Berry Symptoms:** Grapes may shrivel and become raisin-like, reducing fruit quality and yield. Berries can develop black spots or streaks, affecting their appearance and marketability.")
    st.write("- **Vine Decline:** Infected vines often show reduced vigor, stunted growth, and poor overall health. Severe infections can lead to the death of entire vines, especially in older plants.")
    
    st.write("**Preventive Measures:**")
    st.write("- **Pruning:** Remove and destroy infected canes and wood during dormant pruning to reduce the spread of the pathogen. Regularly disinfect pruning tools to prevent the spread of the disease between vines.")
    st.write("- **Irrigation Management:** Ensure proper irrigation management to avoid water stress, which can predispose vines to infection. Enhance soil drainage to prevent waterlogging and reduce disease incidence.")
    st.write("- **Vineyard Sanitation:** Regularly remove plant debris and fallen leaves from the vineyard floor to minimize sources of infection. Manage weeds to reduce competition and improve air circulation around the vines.")
    st.write("- **Plant Resistance:** Choose grape varieties that are less susceptible to esca for new plantings. Use rootstocks that are resistant to soil-borne pathogens.")
    
    st.write("**Treatment:**")
    st.write("- **Fungicides:** Apply fungicides as a preventive measure during the growing season, particularly before wet periods. Use fungicides containing active ingredients such as tebuconazole, boscalid, or pyraclostrobin. Ensure timely applications according to the product label and local agricultural extension recommendations.")
    st.write("- **Biocontrol Agents:** Use biocontrol agents like Trichoderma spp. to reduce pathogen load in the soil and on vine surfaces. Promote beneficial soil microorganisms to improve vine health and resilience.")
    st.write("- **Cultural Practices:** Implement proper canopy management to improve air circulation and reduce humidity around the vines. Provide balanced nutrition to maintain vine health and reduce susceptibility to infections.")
    st.write("- **Heat Treatment:** Use hot water treatment of nursery stock to reduce the presence of pathogens before planting. In severe cases, controlled burning of infected wood can help reduce pathogen load in the vineyard.")

    # Corn Leaf Diseases
    st.subheader("#Corn Leaf Diseases")
    
    st.write("### 1. Northern Corn Leaf Blight")
    st.write("**Symptoms:**")
    st.write("- Elongated, gray-green lesions on leaves that eventually turn brown.")
    st.write("- Lesions often have a grayish or tan center with dark borders.")
    st.write("- Reduced photosynthesis and premature leaf death.")
    
    st.write("**Preventive Measures:**")
    st.write("- **Crop Rotation:** Avoid planting corn in the same location consecutively.")
    st.write("- **Resistant Varieties:** Plant corn varieties resistant to northern corn leaf blight.")
    st.write("- **Sanitation:** Remove and destroy infected plant debris to reduce the source of infection.")
    
    st.write("**Medicines:**")
    st.write("Use fungicides like propiconazole or tebuconazole. Apply these products according to the manufacturer's instructions, particularly during periods of high humidity.")
    
    st.write("### 2. Southern Corn Leaf Blight")
    st.write("**Symptoms:**")
    st.write("- Small, round to oval lesions on leaves with a yellow halo.")
    st.write("- Lesions turn brown or grayish with a darker border.")
    st.write("- Severe infections lead to leaf death and reduced yield.")
    
    st.write("**Preventive Measures:**")
    st.write("- **Crop Rotation:** Avoid planting corn in the same field every year.")
    st.write("- **Resistant Varieties:** Choose corn varieties that are resistant to southern corn leaf blight.")
    st.write("- **Sanitation:** Remove and destroy infected plant debris.")
    
    st.write("**Medicines:**")
    st.write("Use fungicides like chlorothalonil or azoxystrobin. Follow the application rates and timing suggested by the manufacturer.")
    
    st.write("### 3. Gray Leaf Spot")
    st.write("**Symptoms:**")
    st.write("- Small, rectangular lesions with gray centers and brown borders on leaves.")
    st.write("- Lesions can coalesce, leading to large areas of dead tissue.")
    st.write("- Reduced photosynthesis and premature leaf drop.")
    
    st.write("**Preventive Measures:**")
    st.write("- **Crop Rotation:** Avoid planting corn in the same location for several years.")
    st.write("- **Resistant Varieties:** Plant varieties that are less susceptible to gray leaf spot.")
    st.write("- **Sanitation:** Remove and destroy infected plant debris.")
    
    st.write("**Medicines:**")
    st.write("Use fungicides such as tebuconazole or pyraclostrobin. Apply according to the manufacturer's recommendations.")
    
    st.write("### 4. Common Rust")
    st.write("**Symptoms:**")
    st.write("- Small, round, reddish-brown pustules on leaves and stems.")
    st.write("- Pustules break open, releasing rusty spores.")
    st.write("- Severe infections can lead to reduced yield and poor grain quality.")
    
    st.write("**Preventive Measures:**")
    st.write("- **Crop Rotation:** Rotate crops and avoid planting corn in the same area each year.")
    st.write("- **Resistant Varieties:** Use corn varieties with resistance to common rust.")
    st.write("- **Sanitation:** Remove and destroy infected plant debris.")
    
    st.write("**Medicines:**")
    st.write("Use fungicides such as chlorothalonil or azoxystrobin. Follow application guidelines to control the disease effectively.")


    # Potato Leaf Diseases Information
    st.subheader("#Potato Leaf Diseases")

    # Late Blight
    st.write("### 1. Late Blight")
    st.write("**Symptoms:**")
    st.write("- **Leaves:** Dark, water-soaked lesions with a greenish-black color. The lesions often have a fuzzy, white appearance on the underside in humid conditions.")
    st.write("- **Stems:** Dark, sunken lesions that can cause the stem to collapse.")
    st.write("- **Tubers:** Dark, rotted areas with a damp appearance.")
    st.write("**Preventive Measures:**")
    st.write("- **Use Resistant Varieties:** Opt for potato varieties that are resistant to late blight.")
    st.write("- **Proper Spacing:** Ensure adequate spacing between plants to improve air circulation and reduce humidity.")
    st.write("- **Regular Inspection:** Monitor plants frequently and remove infected parts immediately.")
    st.write("- **Avoid Overhead Irrigation:** Use drip irrigation to minimize leaf wetness.")
    st.write("**Medicines/Treatments:**")
    st.write("- **Fungicides:** Apply systemic fungicides such as metalaxyl-mancozeb or copper-based fungicides like Bordeaux mixture.")
    st.write("- **Organic Treatments:** Neem oil or potassium bicarbonate can be used as alternatives.")

    # Early Blight
    st.write("### 2. Early Blight")
    st.write("**Symptoms:**")
    st.write("- **Leaves:** Dark brown to black lesions with concentric rings, often starting on the lower, older leaves. The lesions are typically surrounded by a yellow halo.")
    st.write("- **Stem and Fruits:** Can also be affected with lesions that lead to plant dieback.")
    st.write("**Preventive Measures:**")
    st.write("- **Resistant Varieties:** Choose potato varieties that are less susceptible to early blight.")
    st.write("- **Crop Rotation:** Rotate potatoes with non-solanaceous crops to reduce pathogen load.")
    st.write("- **Remove Infected Material:** Regularly remove and dispose of infected leaves and plant debris.")
    st.write("- **Good Field Hygiene:** Avoid working in fields when they are wet to minimize disease spread.")
    st.write("**Medicines/Treatments:**")
    st.write("- **Fungicides:** Apply fungicides such as chlorothalonil or mancozeb for effective control.")
    st.write("- **Preventive Sprays:** Start fungicide applications early in the growing season to prevent outbreaks.")

    # Powdery Scab
    st.write("### 3. Powdery Scab")
    st.write("**Symptoms:**")
    st.write("- **Tubers:** Powdery, rough lesions on the tuber surface, which can be covered with a powdery substance.")
    st.write("- **Roots:** Can affect roots, causing them to become distorted or swollen.")
    st.write("**Preventive Measures:**")
    st.write("- **Use Clean Seed Tubers:** Plant disease-free seed tubers and ensure they are treated before planting.")
    st.write("- **Soil Management:** Avoid planting in fields with a history of powdery scab and practice good soil management.")
    st.write("- **Proper Watering:** Avoid over-watering and maintain good soil drainage.")
    st.write("**Medicines/Treatments:**")
    st.write("- **Soil Treatments:** Apply soil fumigants or fungicides that are effective against the pathogen. Consult with local agricultural extension services for specific recommendations.")

      # Adding Strawberry Leaf Diseases Information
    st.subheader("#Strawberry Leaf Diseases")

    # Powdery Mildew
    st.write("### 1. Powdery Mildew")
    st.write("**Symptoms:**")
    st.write("- **Leaves:** White, powdery fungal growth on the upper surfaces of leaves. The affected leaves may become distorted and curled.")
    st.write("- **Petals and Fruits:** Can also develop a white powdery coating, leading to reduced fruit quality.")
    
    st.write("**Preventive Measures:**")
    st.write("- **Resistant Varieties:** Choose strawberry varieties that are resistant to powdery mildew.")
    st.write("- **Proper Spacing:** Space plants adequately to ensure good air circulation and reduce humidity around plants.")
    st.write("- **Avoid Overhead Watering:** Use drip irrigation instead of overhead watering to keep leaves dry.")
    st.write("- **Prune and Remove Infected Parts:** Regularly remove and destroy infected plant parts.")
    
    st.write("**Treatments:**")
    st.write("- **Fungicides:** Apply sulfur-based fungicides or potassium bicarbonate. Systemic fungicides like myclobutanil can also be effective.")
    st.write("- **Organic Options:** Neem oil can help manage powdery mildew.")

    # Angular Leaf Spot
    st.write("### 2. Angular Leaf Spot")
    st.write("**Symptoms:**")
    st.write("- **Leaves:** Small, round spots with dark brown or purplish centers and yellow halos. Spots can coalesce, leading to significant leaf damage.")
    st.write("- **Petals:** Can also be affected with similar spotting.")
    
    st.write("**Preventive Measures:**")
    st.write("- **Resistant Varieties:** Select strawberry varieties that have resistance to leaf spot diseases.")
    st.write("- **Good Hygiene:** Remove and destroy fallen leaves and other plant debris where spores can overwinter.")
    st.write("- **Proper Watering:** Water at the base of the plants to avoid wetting the foliage.")
    
    st.write("**Medicines/Treatments:**")
    st.write("- **Fungicides:** Apply fungicides such as chlorothalonil or mancozeb to control the spread.")
    st.write("- **Organic Treatments:** Use copper-based fungicides or plant extracts like garlic for control.")

    # Leaf Spot
    st.write("### 3. Leaf Spot")
    st.write("**Symptoms:**")
    st.write("- Small, purple spots on leaves.")
    st.write("- Spots enlarge and turn tan or gray with purple borders.")
    
    st.write("**Preventive Measures:**")
    st.write("- Plant resistant varieties.")
    st.write("- Remove infected leaves and debris.")
    st.write("- Avoid overhead irrigation.")
    
    st.write("**Treatment:**")
    st.write("- Apply fungicides such as captan or myclobutanil.")
    st.write("- Follow label instructions for application frequency.")

    # Leaf Scorch
    st.write("### 4. Leaf Scorch")
    st.write("**Symptoms:**")
    st.write("- Dark purple to red spots that coalesce, causing leaf scorch.")
    st.write("- Infected leaves may curl, wither, and die.")
    
    st.write("**Preventive Measures:**")
    st.write("- Avoid overhead irrigation.")
    st.write("- Remove and destroy infected leaves and debris.")
    st.write("- Maintain proper spacing for air circulation.")
    
    st.write("**Treatment:**")
    st.write("- Apply fungicides such as myclobutanil or captan.")
    st.write("- Follow recommended application intervals.")

    # Adding Soybean Leaf Diseases Information
    st.subheader("#Soybean Leaf Diseases")

    # Soybean Rust
    st.write("### 1. Soybean Rust")
    st.write("**Symptoms:**")
    st.write("- **Leaves:** Small, reddish-brown pustules or lesions on the underside of leaves, which may produce a rusty, orange-brown spore mass.")
    st.write("- **Spread:** Pustules may coalesce, causing large areas of the leaf to turn yellow and die.")
    
    st.write("**Preventive Measures:**")
    st.write("- **Resistant Varieties:** Choose soybean varieties that are resistant to rust.")
    st.write("- **Proper Spacing:** Ensure good air circulation by spacing plants adequately to reduce humidity around the plants.")
    st.write("- **Crop Rotation:** Rotate with non-leguminous crops to reduce pathogen buildup in the soil.")
    st.write("- **Field Sanitation:** Remove and destroy infected plant debris to reduce spore sources.")
    
    st.write("**Treatments:**")
    st.write("- **Fungicides:** Apply systemic fungicides like triazoles (e.g., tebuconazole) or strobilurins (e.g., azoxystrobin) for effective control.")
    st.write("- **Preventive Sprays:** Use fungicides as a preventive measure, especially during humid conditions.")

    # Bacterial Blight
    st.write("### 2. Bacterial Blight")
    st.write("**Symptoms:**")
    st.write("- Small, angular, water-soaked lesions on leaves.")
    st.write("- Lesions may turn brown and become necrotic.")
    
    st.write("**Preventive Measures:**")
    st.write("- Use certified disease-free seed.")
    st.write("- Practice crop rotation and avoid working in wet fields to prevent spreading.")
    
    st.write("**Treatment:**")
    st.write("- **Copper-based Bactericides:** Can reduce bacterial populations.")
    st.write("- **Examples:** Copper hydroxide, Copper oxychloride.")

    # Septoria
    st.write("### 3. Septoria")
    st.write("**Symptoms:**")
    st.write("- Small, angular, brown lesions on lower leaves.")
    st.write("- Severe infections cause leaves to turn yellow and drop prematurely.")
    
    st.write("**Preventive Measures:**")
    st.write("- Rotate crops to reduce disease pressure.")
    st.write("- Use resistant soybean varieties.")
    
    st.write("**Medicines:**")
    st.write("- **Fungicides:** Apply preventive fungicides at the early vegetative stage.")
    st.write("- **Examples:** Thiophanate-methyl, Chlorothalonil.")

    # Powdery Mildew
    st.write("### 4. Powdery Mildew")
    st.write("**Symptoms:**")
    st.write("- White to grayish powdery spots on leaves, stems, and pods.")
    st.write("- Leaves may turn yellow, curl, and drop prematurely.")
    st.write("- Severe infections can reduce photosynthesis, leading to stunted growth and lower yields.")
    
    st.write("**Preventive Measures:**")
    st.write("- **Resistant Varieties:** Plant soybean varieties that are resistant to powdery mildew.")
    st.write("- **Crop Rotation:** Rotate soybeans with non-host crops to break the disease cycle.")
    st.write("- **Field Sanitation:** Remove and destroy crop residues and infected plant debris after harvest.")
    st.write("- **Proper Spacing:** Ensure proper plant spacing to improve air circulation and reduce humidity, which can inhibit fungal growth.")
    st.write("- **Water Management:** Avoid overhead irrigation to keep foliage dry.")
    
    st.write("**Treatment:**")
    st.write("- **Fungicides:** Apply fungicides when the disease is first detected and repeat as necessary.")
    st.write("- **Examples:** Sulfur, Triadimefon, Myclobutanil.")
    st.write("- **Biological Control:** Use biological fungicides containing beneficial organisms that can suppress powdery mildew.")
    st.write("- **Cultural Practices:** Implement good agricultural practices, such as balanced fertilization and timely irrigation, to maintain plant health and vigor.")

    # Frogeye Leaf Spot
    st.write("### 5. Frogeye Leaf Spot")
    st.write("**Symptoms:**")
    st.write("- Circular to angular spots with gray centers and dark brown margins.")
    st.write("- Lesions may merge, causing larger dead areas.")
    
    st.write("**Preventive Measures:**")
    st.write("- Plant resistant varieties.")
    st.write("- Rotate crops with non-host crops.")
    
    st.write("**Medicine:**")
    st.write("- **Fungicides:** Strobilurins and triazoles.")

    # Target Leaf Spot
    st.write("### 6. Target Leaf Spot")
    st.write("**Symptoms:**")
    st.write("- Circular, target-like lesions with concentric rings, usually with a dark border and lighter center.")
    st.write("- Lesions may coalesce, causing large necrotic areas.")
    
    st.write("**Preventive Measures:**")
    st.write("- Rotate crops to non-host plants.")
    st.write("- Use resistant soybean varieties.")
    st.write("- Practice good field sanitation to remove infected plant debris.")
    
    st.write("**Medicine:**")
    st.write("- **Fungicides:** Chlorothalonil or pyraclostrobin during early stages of infection.")

    # Sudden Death Syndrome
    st.write("### 7. Sudden Death Syndrome")
    st.write("**Symptoms:**")
    st.write("- Interveinal chlorosis and necrosis, leading to leaf scorch.")
    st.write("- Leaves may drop, but petioles remain attached.")
    st.write("- Root rot and blue fungal growth on roots in advanced stages.")
    
    st.write("**Preventive Measures:**")
    st.write("- Use resistant varieties and high-quality, disease-free seeds.")
    st.write("- Implement crop rotation and tillage to reduce pathogen levels.")
    st.write("- Improve soil drainage to prevent waterlogged conditions.")
    
    st.write("**Medicine:**")
    st.write("- No direct fungicides for SDS; management focuses on prevention and resistant varieties.")

    # Bacterial Pustule
    st.write("### 8. Bacterial Pustule")
    st.write("**Symptoms:**")
    st.write("- Small, water-soaked lesions that turn into brown, raised pustules.")
    st.write("- Lesions surrounded by a yellow halo.")
    st.write("- Pustules may merge, causing larger dead areas.")
    
    st.write("**Preventive Measures:**")
    st.write("- Use disease-free seed and resistant varieties.")
    st.write("- Avoid overhead irrigation to reduce leaf wetness.")
    st.write("- Implement crop rotation and proper field sanitation.")
    
    st.write("**Medicine:**")
    st.write("- **Copper-based Bactericides:** Can be applied, but effectiveness is variable.")

    # Yellow Mosaic
    st.write("### 9. Yellow Mosaic")
    st.write("**Symptoms:**")
    st.write("- Yellow, mosaic-like patterns on leaves.")
    st.write("- Stunted growth and reduced vigor.")
    st.write("- Leaves may become thickened and brittle.")
    
    st.write("**Preventive Measures:**")
    st.write("- Plant resistant soybean varieties.")
    st.write("- Control vector populations (such as whiteflies) that transmit the virus.")
    st.write("- Implement integrated pest management (IPM) practices.")
    
    st.write("**Medicine:**")
    st.write("- No direct chemical treatment; focus on vector control and resistant varieties.")

    
    # Wheat Leaf Diseases Information
    st.subheader("#Wheat Leaf Diseases")
    
    # Wheat Leaf Brown Rust
    st.write("### 1. Wheat Leaf Brown Rust")
    st.write("**Symptoms:**")
    st.write("- **Lesions:** Small, round, reddish-brown pustules or lesions on the leaves.")
    st.write("- **Spots:** These pustules can expand, causing the leaves to turn yellow and die prematurely.")
    
    st.write("**Preventive Measures:**")
    st.write("- **Resistant Varieties:** Plant rust-resistant wheat varieties.")
    st.write("- **Crop Rotation:** Avoid growing wheat in the same field consecutively to reduce pathogen buildup.")
    st.write("- **Field Sanitation:** Remove and destroy infected plant debris.")
    
    st.write("**Treatments:**")
    st.write("- **Fungicides:** Apply systemic fungicides such as triadimefon, propiconazole, or tebuconazole according to recommended guidelines.")
    
    # Septoria
    st.write("### 2. Septoria")
    st.write("**Symptoms:**")
    st.write("- **Small, chlorotic (yellow) spots** on the leaves that enlarge and turn brown.")
    st.write("- **Characteristic black pycnidia:** Fruiting bodies of the fungus develop within the lesions.")
    st.write("- **Lesions may coalesce:** Causing large necrotic areas on the leaves.")
    st.write("- **The disease often starts on the lower leaves:** And progresses upwards.")
    
    st.write("**Preventive Measures:**")
    st.write("1. **Crop Rotation:** Rotate wheat with non-host crops to reduce the inoculum in the soil.")
    st.write("2. **Resistant Varieties:** Plant resistant wheat varieties where available.")
    st.write("3. **Field Hygiene:** Remove and destroy crop residues after harvest to reduce fungal inoculum.")
    st.write("4. **Proper Spacing:** Ensure adequate plant spacing to improve air circulation and reduce humidity around the leaves.")
    st.write("5. **Optimal Fertilization:** Avoid excessive nitrogen fertilization as it can promote dense canopy growth, which favors disease development.")
    
    st.write("**Medicines:**")
    st.write("1. **Fungicides:**")
    st.write("   - **Triazoles (e.g., Tebuconazole, Prothioconazole):** Effective in controlling septoria leaf blotch when applied preventively or at the onset of symptoms.")
    st.write("   - **Strobilurins (e.g., Azoxystrobin):** Can be used in combination with triazoles for better control.")
    st.write("   - **SDHIs (e.g., Fluxapyroxad):** Another class of fungicides effective against septoria when used in combination with triazoles or strobilurins.")
    
    # Wheat Yellow Rust
    st.write("### 3. Wheat Yellow Rust") 
    st.write("**Symptoms:**")
    st.write("- **Lesions:** Yellow to orange pustules (stripes) that form on the leaves, often seen in parallel rows.")
    st.write("- **Leaf Chlorosis:** Leaves may exhibit chlorosis (yellowing) and eventually die, affecting the overall plant health and yield.")
    
    st.write("**Preventive Measures:**")
    st.write("- **Resistant Varieties:** Use wheat varieties that are resistant to yellow rust.")
    st.write("- **Timely Planting:** Plant wheat at recommended times to avoid the peak periods of rust infection.")
    st.write("- **Field Sanitation:** Remove and destroy infected plant residues to reduce pathogen load.")
    
    st.write("**Medicines:**")
    st.write("- **Fungicides:** Effective treatments include fungicides such as propiconazole, epoxiconazole, and fenpropimorph. Apply as soon as symptoms are detected and according to the recommended schedule.")


    # Rice Leaf Diseases
    st.subheader("#Rice Leaf Diseases")

    st.write("### 1. Rice Leaf Blast")
    st.write("**Symptoms:**")
    st.write("- **Lesions:** Large, round or oval lesions with a grayish center and a dark border on the leaves, often with a pointed tip.")
    st.write("- **Spots:** Lesions can coalesce, causing significant leaf death and reducing photosynthetic area.")
    st.write("**Preventive Measures:**")
    st.write("- **Resistant Varieties:** Use rice varieties that are resistant to rice blast.")
    st.write("- **Proper Spacing:** Maintain proper plant spacing to improve air circulation and reduce humidity.")
    st.write("- **Field Sanitation:** Remove and destroy infected plant debris to reduce pathogen inoculum.")
    st.write("**Medicines:**")
    st.write("Use fungicides such as tricyclazole, pyraflufen-ethyl, or carbendazim according to the recommended schedule and dosage.")
    
    st.write("### 2. Rice Brown Spot")
    st.write("**Symptoms:**")
    st.write("- **Lesions:** Small, brownish spots on leaves that have a yellow halo. These spots can become larger and cause significant leaf damage.")
    st.write("- **Leaf Blight:** In severe cases, extensive leaf blight can occur, leading to reduced yield and quality.")
    st.write("**Preventive Measures:**")
    st.write("- **Resistant Varieties:** Choose and plant varieties that are resistant to brown spot.")
    st.write("- **Nutrient Management:** Avoid excessive nitrogen fertilization as it can exacerbate the disease. Ensure balanced nutrient application.")
    st.write("- **Water Management:** Maintain proper water management practices to avoid excessive moisture, which can promote the disease.")
    st.write("**Medicines:**")
    st.write("Apply fungicides such as carbendazim or mancozeb. Follow recommended application rates and timings.")
    
    st.write("### 3. Rice Sheath Blight")
    st.write("**Symptoms:**")
    st.write("- **Lesions:** Water-soaked lesions on the leaf sheaths and stems that can become brown and necrotic.")
    st.write("- **Sheath Rot:** Infected sheaths may rot, causing affected plants to collapse or lodge.")
    st.write("**Preventive Measures:**")
    st.write("- **Resistant Varieties:** Use rice varieties with resistance to sheath blight.")
    st.write("- **Field Sanitation:** Remove infected plant debris and practice proper crop rotation.")
    st.write("- **Balanced Fertilization:** Avoid excessive nitrogen fertilization and provide balanced nutrient applications to reduce disease severity.")
    st.write("**Medicines:**")
    st.write("Use fungicides such as flutriafol or benomyl for controlling sheath blight. Apply as recommended to manage the disease effectively.")
    
    st.write("### 4. Bacterial Leaf Blight")
    st.write("**Symptoms:**")
    st.write("- **Lesions:** Small, water-soaked spots on the leaves that gradually turn yellowish-brown with a characteristic yellow halo.")
    st.write("- **Blight:** In severe cases, the lesions can coalesce, leading to extensive leaf blight and reduced plant vigor.")
    st.write("**Preventive Measures:**")
    st.write("- **Resistant Varieties:** Plant rice varieties that are resistant to leaf blight.")
    st.write("- **Field Sanitation:** Remove and destroy infected plant debris and practice proper field management to reduce pathogen survival.")
    st.write("- **Proper Spacing:** Ensure adequate plant spacing for improved air circulation and reduced humidity around the plants.")
    st.write("**Medicines/Treatments:**")
    st.write("Apply copper-based fungicides such as copper oxychloride or copper hydroxide. Follow application guidelines for effective control.")
    st.write("Systemic antibiotics like streptomycin may be used in some cases, but these should be applied as per local agricultural recommendations.")


# Mango Leaf Diseases
    st.subheader("#Mango Leaf Diseases")

    st.write("### 1. Anthracnose")
    st.write("**Symptoms:**")
    st.write("- **Dark, sunken spots** on leaves, stems, flowers, and fruit.")
    st.write("- **Leaves may curl and die back.**")
    st.write("**Preventive Measures:**")
    st.write("- **Prune trees** to improve air circulation.")
    st.write("- **Avoid overhead irrigation** to keep leaves dry.")
    st.write("- **Remove and destroy infected plant debris.**")
    st.write("**Treatment:**")
    st.write("- **Chemical:** Apply copper-based fungicides or mancozeb. Start spraying at the beginning of the flowering season and continue at intervals recommended by the product label.")
    st.write("- **Organic:** Neem oil can also be used as a preventive measure.")

    st.write("### 2. Powdery Mildew")
    st.write("**Symptoms:**")
    st.write("- **White, powdery fungal growth** on leaves, stems, and flowers.")
    st.write("- **Affected leaves may become distorted and fall off.**")
    st.write("**Preventive Measures:**")
    st.write("- **Ensure good air circulation** around trees by proper spacing and pruning.")
    st.write("- **Avoid excessive nitrogen fertilization** which can promote tender growth susceptible to infection.")
    st.write("**Treatment:**")
    st.write("- **Chemical:** Apply sulfur-based fungicides or potassium bicarbonate at the first sign of infection.")
    st.write("- **Organic:** Use neem oil or horticultural oils as preventive measures.")

    st.write("### 3. Bacterial Canker")
    st.write("**Symptoms:**")
    st.write("- **Water-soaked lesions** on leaves, which turn brown and necrotic.")
    st.write("- **Leaf margins may have a scorched appearance.**")
    st.write("- **Gummy exudates** on stems and fruit.")
    st.write("**Preventive Measures:**")
    st.write("- **Use disease-free planting material.**")
    st.write("- **Avoid injuries** to plants that can serve as entry points for bacteria.")
    st.write("- **Remove and destroy infected plant parts.**")
    st.write("**Treatment:**")
    st.write("- **Chemical:** Spray with copper-based bactericides. Ensure thorough coverage, especially on the undersides of leaves.")
    st.write("- **Organic:** Use compost tea sprays and maintain tree vigor with proper fertilization and watering.")

    st.write("### 4. Dieblack")
    st.write("**Symptoms:**")
    st.write("- **Gradual dying of shoots, branches, or entire trees** from the tips backward.")
    st.write("- **Initial symptoms** include small, brownish-black lesions on the stem, which enlarge and cause the death of the branch.")
    st.write("- **Leaves on affected branches turn yellow, wither, and fall off.**")
    st.write("- **Eventually, the disease can spread** to larger branches and the main stem, leading to tree death.")
    st.write("**Preventive Measures:**")
    st.write("- **Ensure proper irrigation and soil management** to reduce plant stress.")
    st.write("- **Prune and dispose of infected branches** to prevent the spread of the disease.")
    st.write("- **Disinfect pruning tools** to avoid spreading the pathogen.")
    st.write("- **Improve tree vigor** through balanced fertilization and proper care.")
    st.write("**Medicine:**")
    st.write("- **Apply fungicides** like Thiophanate-methyl or Carbendazim to affected areas.")
    st.write("- **Copper-based fungicides** can also be used as a preventive measure.")

    st.write("### 5. Sooty Mold")
    st.write("**Symptoms:**")
    st.write("- **Black, sooty growth** on leaves and stems.")
    st.write("- **Reduced photosynthesis and stunted growth.**")
    st.write("**Preventive Measures:**")
    st.write("- **Control sap-sucking insects** like aphids and mealybugs.")
    st.write("- **Prune and destroy affected plant parts.**")
    st.write("**Medicine:**")
    st.write("- **Insecticides** to control vectors (e.g., Imidacloprid) and fungicides (e.g., Mancozeb).")

    st.write("### 6. Gall Midge")
    st.write("**Symptoms:**")
    st.write("- **Formation of galls (swellings)** on leaves, flowers, and young shoots.")
    st.write("- **Galls appear as small, round, and greenish-yellow or reddish swellings.**")
    st.write("- **Infested leaves may become distorted, curled, and drop prematurely.**")
    st.write("- **Severe infestations can lead to reduced growth and yield.**")
    st.write("**Preventive Measures:**")
    st.write("- **Monitor trees regularly** for early detection of gall midge activity.")
    st.write("- **Remove and destroy infested plant parts** to reduce the midge population.")
    st.write("- **Implement cultural practices** like pruning and maintaining tree health to reduce susceptibility.")
    st.write("**Medicine:**")
    st.write("- **Use systemic insecticides** like Imidacloprid or Thiamethoxam to control gall midge infestations.")
    st.write("- **Insecticidal sprays** can be applied during the early stages of infestation for effective control.")




if selected == "Contact/Help":
    st.markdown(f"<h1 class='custom-header'>Contact/Help</h1>", unsafe_allow_html=True)
    st.write("""
    If you need assistance or have any questions about using our system, feel free to contact us:

    **You can contact us at:** neupaneoman90@gmail.com   itachisenku7@gmail.com   Rakeysh36@gmail.com   mbista313@gmail.com panchalal2025@gmail.com
    
    **Help Desk:** Available via email. Please reach out to us for immediate assistance.
  
    ### FAQs
    - **How accurate is the system?**
             
      Our system is designed to provide reliable predictions, but it is always a good idea to consult with experts for confirmation.

    - **What types of diseases can be detected?** 
             
      The system is trained to identify a wide range of common plant diseases. For a full list, visit the Disease Information page.

    - **Can I contribute to the dataset?**
             
      Yes, we welcome contributions to improve our model. Contact us if you are interested in collaborating.
    """)












