import os
import streamlit as st
import requests


API_URL = os.environ.get("API_URL", "http://localhost:8000")

st.set_page_config(page_title="VectraDB AI", page_icon="🧠", layout="centered")

st.title("🧠 VectraDB AI Image Search")
st.markdown("Hardware-Accelerated C++ Vector Database with OpenAI CLIP")

tab_upload, tab_search = st.tabs(["📤 Upload to Database", "🔍 Semantic Search"])



with tab_upload:
    st.header("Add Images to the Database")
    
    upload_category = st.text_input("Image Category Label (e.g., car, dog, landscape)")
    upload_file = st.file_uploader("Choose an image to upload...", type=["jpg", "jpeg", "png"], key="up")
    
    if st.button("Extract Vector & Save to DB"):
        if upload_file and upload_category:
            with st.spinner("AI is extracting 512D concepts..."):
                
                files = {"file": (upload_file.name, upload_file.getvalue(), upload_file.type)}
                data = {"category": upload_category}
                
                response = requests.post(f"{API_URL}/upload/", files=files, data=data)
                
                if response.status_code == 200 and response.json().get("status") == "success":
                    st.success(f"Success! Saved to C++ Database with Document ID: {response.json()['db_id']}")
                else:
                    st.error("Failed to upload to database.")
        else:
            st.warning("Please provide both a category and an image.")



with tab_search:
    st.header("Search by Concept")
    st.markdown("Upload an image, and the AI will find the closest mathematical matches.")
    
    search_file = st.file_uploader("Upload a query image...", type=["jpg", "jpeg", "png"], key="search")
    k_value = st.slider("Number of results to fetch (k)", min_value=1, max_value=10, value=5)
    
    if st.button("Run Search"):
        if search_file:
            
            st.image(search_file, caption="Your Query Concept", width=300)
            
            with st.spinner("Searching millions of vectors in milliseconds..."):
                
                files = {"file": (search_file.name, search_file.getvalue(), search_file.type)}
                data = {"k": k_value}
                
                response = requests.post(f"{API_URL}/search/", files=files, data=data)
                
                if response.status_code == 200 and response.json().get("status") == "success":
                    matches = response.json()["matches"]
                    
                    st.subheader(f"Top {len(matches)} Closest Matches found:")
                    
                    cols = st.columns(len(matches))
                    
                    for i, match in enumerate(matches):
                        with cols[i]:
                            doc_id = match['id']
                            dist = match['distance']
                            cat = match['category']
                            
                            image_path = f"uploads/{doc_id}.jpg"
                            
                            try:
                                st.image(image_path, use_container_width=True)
                                st.markdown(f"**ID:** {doc_id}")
                                st.markdown(f"**Category:** {cat}")
                                st.markdown(f"**Dist:** `{dist:.4f}`")
                            except FileNotFoundError:
                                st.error("Image file not found on disk.")
                else:
                    st.error("Search failed.")
        else:
            st.warning("Please upload an image to search.")