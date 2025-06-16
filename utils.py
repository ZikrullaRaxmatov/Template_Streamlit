import streamlit as st

def result_card(uploaded_img, class_names, class_index, confidence):
    # Create two columns
        col1, col2 = st.columns(2)

        # Load and show image in left column
        with col1:
            st.image(
            uploaded_img,
            caption= class_names[class_index],
            width=360,
            channels="RGB"
        )

        # Show description in right column
        with col2:
            st.markdown(f"""
                <div style="
                    border: 3px;
                    border-radius: 10px;
                    padding: 1rem;
                    background-color: #202020;
                ">
                    <h4>Description</h4>
                    <ul>
                        <li><b>Type</b>: {class_names[class_index]}</li>
                        <li><b>Size</b>: 224x224 pixels</li>
                        <li><b>Purpose</b>: Identify possible lung abnormalities</li>
                        <li><b>Model Confidence</b>: {confidence.item()*100:.2f}%</li>
                    </ul>
                </div>
            """, unsafe_allow_html=True)