import streamlit as st

# Inject CSS for h2 with gradient
st.markdown("""
    <style>
        .main-header {
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            padding: 2rem;
            border-radius: 10px;
            text-align: center;
            color: white;
            margin-bottom: 2rem;
        }
        
        a img:hover {
            opacity: 0.7;
            transform: scale(1.1);
            transition: 0.2s;
        }

    </style>
""", unsafe_allow_html=True)

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Brain Cancer", "X RAY", "Melanoma"])

# Contact Section in Sidebar
st.sidebar.markdown("---")
st.sidebar.subheader("📬 Contact")
st.sidebar.write("For questions or support, reach out:")
st.sidebar.write("📧 zikrullarakhmatov@gmail.com")
st.sidebar.write("🌐 [Visit Website](https://example.com)")
st.sidebar.write("📞 +82 10-7598-6468")

st.sidebar.markdown("""
    <div style="display: flex; gap: 10px; align-items: center;">
        <a href="https://github.com/ZikrullaRaxmatov" target="_blank">
            <img src="https://img.icons8.com/ios-glyphs/30/000000/github.png" alt="GitHub"/>
        </a>
        <a href="https://linkedin.com/in/zikrulla-rakhmatov-5aa470272" target="_blank">
            <img src="https://img.icons8.com/ios-filled/30/000000/linkedin.png" alt="LinkedIn"/>
        </a>
        <a href="https://t.me/ZikrullaRakhmatov" target="_blank">
            <img src="https://img.icons8.com/ios-filled/30/000000/telegram-app.png" alt="Telegram"/>
        </a>
    </div>
""", unsafe_allow_html=True)

# Main title (keeps default)
st.title("Medical Diagnosis")

# Page content
if page == "Home":
    st.markdown('<h2 class="main-header">Welcome to the <em>RZR.AI</em> Medical Diagnosis</h2>', unsafe_allow_html=True)
    st.write("Use the sidebar to navigate to different sections.")
    st.image("https://keck.usc.edu/news/wp-content/uploads/sites/68/2023/09/ai-in-medicine-target.jpg", caption="Virus detector")
    
elif page == "Brain Cancer":
    st.markdown('<h2 class="main-header">Brain Cancer Detection</h2>', unsafe_allow_html=True)
    st.write("Brain cancer refers to the abnormal growth of cells in the brain. Early diagnosis and treatment are crucial.")
    
    img = st.file_uploader("Upload Brain MRI", type=["jpg", "png", "jpeg"])
    
    if img:

        class_names = {
            0 : 'Brain Glioma',
            1 : 'Brain Menin',
            2 : 'Brain Tumor'
        }

        # Load the image
        uploaded_img = Image.open(img)

        # Use transformation
        input_tensor = test_transform(uploaded_img)

        # Add a batch dimension
        input_batch = input_tensor.unsqueeze(0)

        # Move the input to the device
        device = torch.device("cpu")
        input_batch = input_batch.to(device)

        # Load the trained model
        best_model = CNN()
        best_model.load_state_dict(torch.load('best_model4.pt', map_location=torch.device('cpu')))
        best_model.to(device)

        best_model.eval()  # Set the model to evaluation mode

        # Get the model's output
        with torch.no_grad():
            output = best_model(input_batch)

        # Interpret the output
        _, predicted_class = torch.max(output, 1)
        class_index = predicted_class.item()

        st.image(
            uploaded_img,
            caption= class_names[class_index],
            width=400,
            channels="RGB"
        )


elif page == "X RAY":
    st.markdown('<h2 class="main-header">Chest X-RAY Analysis</h2>', unsafe_allow_html=True)
    st.write("X-rays are used to detect issues like pneumonia, tuberculosis, and other lung diseases.")
    st.file_uploader("Upload Chest X-RAY Image", type=["jpg", "png", "jpeg"])

elif page == "Melanoma":
    st.markdown('<h2 class="main-header">Melanoma Skin Cancer Detection</h2>', unsafe_allow_html=True)
    st.write("Melanoma is a serious form of skin cancer that begins in melanocytes.")
    st.file_uploader("Upload Skin Image", type=["jpg", "png", "jpeg"])
