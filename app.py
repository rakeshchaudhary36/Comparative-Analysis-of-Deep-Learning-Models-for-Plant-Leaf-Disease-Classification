import streamlit as st
from PIL import Image
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from torch import nn
from efficientnet_pytorch import EfficientNet

# Define the model architectures
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
        # model = models.efficientnet_b0(pretrained=False)
        model = EfficientNet.from_pretrained('efficientnet-b0')
        # model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)  # Update the final layer
        model._fc = nn.Linear(model._fc.in_features, num_classes)
    else:
        raise ValueError(f"Unknown model name: {name}")
    return model

# Load models with custom state dict loading
def load_model(name, model_path):
    model = get_model(name)
    state_dict = torch.load(model_path)

    # Remove keys related to auxiliary logits for InceptionV3
    if name == "Inception V3":
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
    "EfficientNet": "efficientnet.pth"
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
# Define transforms for image preprocessing
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def predict(image, models):
    inputs = preprocess(image).unsqueeze(0)
    results = {}
    
    for name, model in models.items():
        with torch.no_grad():
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            confidence = torch.nn.functional.softmax(outputs, dim=1).max().item()
            results[name] = (class_names[preds[0]], confidence)
    
    # Select the model with the highest confidence
    best_model = max(results, key=lambda x: results[x][1])
    return best_model, results[best_model]

# Streamlit interface
# st.title("Plant Leaf Disease Detection System")

# Navigation
# pages = {
#     "Homepage": "",
#     "Disease Detection": "Upload an image of a leaf to get a disease prediction.",
#     "About Us": "",
#     "Disease Information": "Detailed information about various leaf diseases, their causes, symptoms, and prevention methods.",
#     "Chatbot": "Interact with our chatbot to get more information.",
#     "Contact/Help": "Contact us for support or get help with our system."
# }

# Sidebar for navigation
st.sidebar.title("Dashboard")
page = st.sidebar.selectbox("Choose a page", ["Home", "Disease Detection", "About Us", "Disease Information", "Contact/Help"])

# Display content based on selected page
# st.write(pages[page])

# Home Page
if page == "Home":
    st.markdown(f"<h1 class='custom-header'>PLANT DISEASE DETECTION SYSTEM</h1>", unsafe_allow_html=True)
    image_path = "back.jpeg"
    st.image(image_path, use_column_width=True)
    st.markdown(""" 
    WELCOME TO THE PLANT LEAF DISEASE DETECTION SYSTEM! 🌿🔍
            
    Use the sidebar to navigate through the features.

    Our mission is to help in identifying plant diseases efficiently. Upload an image of a plant, and our system will analyze it to detect any signs of diseases. Together, let's protect our crops and ensure a healthier harvest!

    ### How It Works
    1. **Upload Image:** Go to the **Disease Detection** page and upload an image of a plant with suspected diseases.
    2. **Analysis:** Our system will process the image using advanced algorithms to identify potential diseases.
    3. **Results:** View the results and recommendations for further action.

    ### Why Choose Us?
    - **Accuracy:** Our system utilizes state-of-the-art machine learning techniques for accurate disease detection.
    - **User-Friendly:** Simple and intuitive interface for seamless user experience.
    - **Fast and Efficient:** Receive results in seconds, allowing for quick decision-making.

    ### Get Started
    Click on the **Disease Detection** page in the sidebar to upload an image and experience the power of our Plant Disease Recognition System!

    ### About Us
    Learn more about the project, our team, and our goals on the **About** page.
    """)

#Disease Detection page
elif page == "Disease Detection":
    st.subheader("Upload Image")
    uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "png", "jpeg"], key="upload_image")

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption='Uploaded Image', use_column_width=True)
        
        # Make prediction
        model_name, (predicted_class, confidence) = predict(image, models)
        st.write(f"Based on the {model_name}, this leaf is classified as '{predicted_class}' with a confidence of {confidence*100:.2f}%.")
        
        # Store prediction state
        if 'prediction_done' not in st.session_state:
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
            
            # Option to reset prediction state
            if st.button("Reset Prediction"):
                st.session_state.prediction_done = False
                st.experimental_rerun()

# About Us page
elif page == "About Us":
    st.markdown(f"<h1 class='custom-header'>About Us</h1>", unsafe_allow_html=True)
    st.markdown("""
We are a team of five passionate computer engineering students from Khwopa Engineering College, currently in our 8th semester, dedicated to developing a plant leaf disease detection system. Our team consists of:

  * **Oman Neupane:** 760324
  * **Aashish Pandey:** 760302
  * **Rakesh Kumar Chaudhary:** 760330
  * **Ishan Bista:** 760317
  * **Nijal Kachhepati:** 760323
                
Driven by a desire to contribute to the agricultural sector, we embarked on this project to create a user-friendly and effective tool for early detection of plant diseases. Our plant leaf disease detection system leverages the power of computer vision and machine learning to analyze images of leaves and identify potential diseases.

##### **Our Mission:**

Our mission is to empower farmers and agricultural professionals with an innovative and accessible technology that aids in early disease detection. By enabling prompt identification of plant diseases, our system aims to:

* **Increase crop yield and quality by minimizing damage caused by undetected diseases.**
* **Reduce reliance on chemical pesticides by facilitating targeted treatment based on the specific disease identified.**
* **Enhance agricultural sustainability by promoting preventative measures and optimizing resource use.**

##### **Our Approach:**

Our plant leaf disease detection system is designed to be user-friendly and adaptable to various environments. We are committed to:

Developing a user interface that is intuitive and easy to navigate for users of all technical backgrounds.
Training the system on a comprehensive dataset of plant leaf images encompassing a wide range of diseases to ensure accurate detection.
Continuing to refine and improve the system's capabilities through ongoing research and development.

##### **Our Vision:**

We envision a future where our plant leaf disease detection system becomes an essential tool for farmers worldwide, contributing to a more productive, sustainable, and resilient agricultural sector.
#### About Dataset
This dataset is recreated using offline augmentation from the original dataset. The original dataset can be found on this GitHub repo.
This dataset consists of about 87K RGB images of healthy and diseased crop leaves categorized into 38 different classes. The total dataset is divided into 80/20 ratio of training and validation set preserving the directory structure.
A new directory containing 33 test images is created later for prediction purposes.
    
#### Content
1. Train (70295 images)
2. Test (33 images)
3. Validation (17572 images)
    """)

# Disease Information page
elif page == "Disease Information":
    st.markdown(f"<h1 class='custom-header'>Disease Imformation</h1>", unsafe_allow_html=True)
    st.write("Detailed information about leaf diseases, their causes, symptoms, and prevention methods.")




    # Grape Leaf Diseases
    st.subheader("Grape Leaf Diseases")
    
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

    # Tomato Leaf Diseases
    st.subheader("Tomato Leaf Diseases")
    
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
    
    # Corn Leaf Diseases
    st.subheader("Corn Leaf Diseases")
    
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

      # Apple Leaf Diseases Information
    st.subheader("Apple Leaf Diseases")
    
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

    # Potato Leaf Diseases Information
    st.subheader("Potato Leaf Diseases")

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
    st.subheader("Strawberry Leaf Diseases")

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
    st.subheader("Soybean Leaf Diseases")

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
    st.subheader("Wheat Leaf Diseases")
    
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
    
    # Rice Leaf Diseases Information
    st.subheader("Rice Leaf Diseases")
    
    # Rice Leaf Blast
    st.write("### 1. Rice Leaf Blast")
    st.write("**Symptoms:**")
    st.write("- **Lesions:** Large, round or oval lesions with a grayish center and a dark border on the leaves, often with a pointed tip.")
    st.write("- **Spots:** Lesions can coalesce, causing significant leaf death and reducing photosynthetic area.")
    
    st.write("**Preventive Measures:**")
    st.write("- **Resistant Varieties:** Use rice varieties that are resistant to blast.")
    st.write("- **Crop Rotation:** Rotate rice with non-host crops to reduce the inoculum.")
    st.write("- **Field Sanitation:** Remove and destroy infected plant debris after harvest.")
    
    st.write("**Treatments:**")
    st.write("- **Fungicides:** Apply systemic fungicides such as pyriculamide or tricyclazole to manage the disease.")
    
    # Rice Leaf Blight
    st.write("### 2. Rice Leaf Blight")
    st.write("**Symptoms:**")
    st.write("- **Lesions:** Irregular, water-soaked lesions that turn brown and can cause extensive leaf dieback.")
    st.write("- **Drying:** Affected leaves dry out and may curl or become necrotic.")
    
    st.write("**Preventive Measures:**")
    st.write("- **Resistant Varieties:** Plant varieties resistant to rice blight.")
    st.write("- **Field Management:** Use proper irrigation and drainage to reduce moisture stress.")
    st.write("- **Sanitation:** Remove and destroy infected plant debris to reduce disease spread.")
    
    st.write("**Treatments:**")
    st.write("- **Fungicides:** Apply products such as copper-based fungicides or systemic fungicides like azoxystrobin to manage the disease.")

    st.write("Feel free to contact us if you have more questions or need further assistance!")







# elif page == "Chatbot":
#     st.subheader("Chatbot")
#     st.write("Interact with our chatbot to get more information.")


#Contact us page
elif page == "Contact/Help":
    st.subheader("Contact Us")
    st.write("For assistance, you can contact us at neupaneoman90@gmail.com\nitachisenku@gmail.com")







