import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity



df= pd.read_csv("HoteRecommend_final.csv")

tfidf=TfidfVectorizer(max_features=1000)


def recommend(type, country, city , property ,starrating,  user_id= None):
    # Check if user_id matches from historical data
    if user_id:
        user_history = get_user_history(user_id, df1)
        if not user_history.empty:
            user_tags = ' '.join(user_history['tags'].tolist())
        else:
            user_tags = f"{type} {country} {city} {property} {starrating}"
    else:
        user_tags = f"{type} {country} {city} {property} {starrating}"

    # Filter the dataset based on the country and city
    temp = df[(df['country'] == country) & (df['city'] == city) & (df['starrating'] >= starrating) &
            (df['roomtype'] == type) & (df['propertytype'] == property) ]
     # Append user preferences to the filtered DataFrame

    if temp.empty: #an empty list is returned if no hotels matches the criteria
        return pd.DataFrame()

    # Create a DataFrame from user tags
    user_tags_df = pd.DataFrame({'tags': [user_tags]})
    
    temp = pd.concat([temp, user_tags_df],ignore_index=True)
    
    # Fit and transform the TF-IDF vectorizer
    vector = tfidf.fit_transform(temp['tags']).toarray()
    user_index = len(temp)-1
    # Calculate cosine similarity matrix
    similarity = cosine_similarity(vector)
    
    # Get indices of the filtered hotels
    filtered_indices = temp[temp['tags'] == user_tags].index.tolist()
    
    # Recommend top 5 similar hotels for each filtered hotel

    similar_hotels = sorted(list(enumerate(similarity[user_index])), key=lambda x: x[1],reverse= True)[1:6]
    # Skip the first match (itself)
    recommended_hotels=[]
    for hotel in similar_hotels:
            #print(tuple(temp.loc[hotel[0]][['hotelname', 'roomtype','starrating']]))
        hotel_details=temp.loc[hotel[0]][['hotelname', 'roomtype','starrating','url']]
        recommended_hotels.append(hotel_details)
    #recommended_hotels = [temp.iloc[i[0]]['hotelname'] for i in similar_hotels]
        
    
    return pd.DataFrame(recommended_hotels)

st.title(":blue[Hotel Recommender System]")

st.header(":rainbow[Enter your preference as per your choice]")

room_type = st.selectbox(
'Roomtype(The kind of room you want)',
df['roomtype'].unique()
)

country = st.selectbox(" Select Country", df['country'].unique())

city = st.selectbox("Select City", df['city'].unique())

property_type = st.selectbox(
'Select Property Type',df['propertytype'].unique()
)

starrating = st.slider('Select Minimum Star Rating', 1, 5, 3)

if st.button('Get Recommendations'):
    recommendations = recommend(room_type, country, city, property_type, starrating)
    
    #if recommendations:
       # st.write('Recommended Hotels:')
        #for hotel in recommendations:
         #   st.write(f"*Hotel Name:* {hotel['hotelname']}")
          #  st.write(f"*Room Type:* {hotel['roomtype']}")
           # st.write(f"*Star Rating:* {hotel['starrating']}")
            #st.write(f"[*Book Now*]({hotel['url']})")
            #t.write("---")
    if not recommendations.empty:
        st.write('Recommended Hotels:')
        
        # Adding a clickable URL in the DataFrame
        recommendations['url'] = recommendations['url'].apply(lambda x: f'<a href="{x}" target="_blank">Book Now</a>')
        
        # Set up the Streamlit table with clickable URLs
        st.write(recommendations.to_html(escape=False, index=False), unsafe_allow_html=True)
    else:
        st.write('No hotels found matching your preferences.')

#Content-Based Filtering Module
#df = pd.read_excel("HotelRecommend2.xlsx")
df = pd.read_csv("HotelRecommend2.csv")
#df = pd.read_csv("HoteRecommend_final.csv")

#tfidf=TfidfVectorizer(max_features=1000)

# Retrive Data Similiarity with Hotel City, Hotel Country and Hotel Amenities
df['hotel_features'] = df['city'] + " " + df['country'] + " " + df['tags']
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(df['hotel_features'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

def cbf_recommendations(hotelName, cosine_sim=cosine_sim):
    try:
        hotel_index = df[df['hotelname'] == hotelName].index[0]
        similar_hotels = list(enumerate(cosine_sim[hotel_index])) # Algorithm Process 1
        similar_hotels = sorted(similar_hotels, key=lambda x: x[1], reverse=True) # Algorithm Process 2

        similar_hotels = similar_hotels[1:41]  # Top 40 similar hotels
        recommended_hotels = [df['hotelname'].iloc[i[0]] for i in similar_hotels]
        #+ "\t Country: "  + df['country'].iloc[i[0]] 

        return recommended_hotels
    
    except IndexError:
        return ["Hotel not found!"]

st.subheader("Content-Based Filtering Module by Tee Zhen Yu")
st.text("Description: Recommend Hotels With City and Room Amenities")

hotelName = st.text_input("Enter Hotel Name for Content-Based Filtering")

if st.button("Get Recommendation"):

    if hotelName:
        st.subheader(f"Hotels Similar To '{hotelName}':")
        recommendation = cbf_recommendations(hotelName)

        for hotel in range(len(recommendation)):

            for rate in range(len(recommendation)):

                st.write(f"Hotel: {recommendation[rate]}")
                try:
                    similarity = cosine_sim[hotel][rate]
                    st.write(f"Similarity Rate: {similarity:.2f}")
                except:
                    print("Rate Not Found")

#Hybrid Module
def hybrid_recommendation(hotel_name, type, country, city, property, starrating):

    # Get content-based recommendations
    content_recommendations = cbf_recommendations(hotel_name)
    
    # Get collaborative filtering recommendations
    collaborative_recommendations = cf_recommend(type, country, city, property, starrating)
    
    # Combine the two recommendations (can use weighted average or intersection)
    hybrid_recommendations = list(set(content_recommendations) | set(collaborative_recommendations))
    
    return hybrid_recommendations

st.subheader("Hybrid Module by Lee Yuan Le")

# Input for hotel name (for content-based filtering)
hotel_name_input = st.selectbox("Enter Hotel Name for Recommendation", df['hotelname'].unique())

# Generate hybrid recommendations
if hotel_name_input:
    hybrid_results = hybrid_recommendation(hotel_name_input, type, country, city, property, starrating)
    
    # Display recommendations
    st.write(f"Hybrid Recommendations for '{hotel_name_input}'")
    for hotel in hybrid_results:
        st.write(hotel)
