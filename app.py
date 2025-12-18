import streamlit as st
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics.pairwise import cosine_similarity

# --- Configuration ---
st.set_page_config(page_title="Taliyo Service Recommender", layout="wide")

@st.cache_data
def load_and_clean():
    # Use the cleaned file we created in Phase 1
    data = pd.read_csv('cleaned_data.csv')
    return data

df = load_and_clean()

# --- Logic ---
def recommend(user_input, data):
    features = ['Target_Business_Type', 'Price_Category', 'Language_Support', 'Location_Area']
    encoder = OneHotEncoder(handle_unknown='ignore')
    encoded_matrix = encoder.fit_transform(data[features])
    
    user_df = pd.DataFrame([user_input])
    for col in user_df.columns: user_df[col] = user_df[col].str.lower()
    user_encoded = encoder.transform(user_df[features])
    
    similarities = cosine_similarity(user_encoded, encoded_matrix).flatten()
    data['Match_Score'] = similarities
    data['Match_Quality'] = data['Match_Score'].apply(lambda x: 'High' if x > 0.7 else 'Medium')
    
    results = data.sort_values(by='Match_Score', ascending=False).head(3)
    
    # Explanations
    exps = []
    for _, row in results.iterrows():
        reasons = [k.replace('_', ' ') for k, v in user_input.items() if row[k] == v.lower()]
        exps.append(f"Recommended because of: {', '.join(reasons)}" if reasons else "General match.")
    results['Explanation'] = exps
    return results

# --- UI ---
st.title("🤝 ML Service Recommendation System")
st.write("Match with the best services based on your unique needs.")

with st.sidebar:
    st.header("Search Preferences")
    bus = st.selectbox("Business Type", df['Target_Business_Type'].unique())
    pri = st.selectbox("Budget", df['Price_Category'].unique())
    lan = st.selectbox("Language", df['Language_Support'].unique())
    loc = st.selectbox("Location", df['Location_Area'].unique())

if st.button("Find Top 3 Services"):
    user_query = {'Target_Business_Type': bus, 'Price_Category': pri, 'Language_Support': lan, 'Location_Area': loc}
    res = recommend(user_query, df)
    
    for _, row in res.iterrows():
        with st.container():
            st.subheader(f"✨ {row['Service_Name']}")
            c1, c2 = st.columns(2)
            c1.metric("Score", f"{int(row['Match_Score']*100)}%")
            c2.write(f"**Quality:** {row['Match_Quality']}")
            st.write(f"**Why?** {row['Explanation']}")
            st.divider()
