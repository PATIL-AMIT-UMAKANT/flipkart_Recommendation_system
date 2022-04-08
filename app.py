import pandas as pd
import streamlit as st
st.set_page_config('Flipkart women dresses Recommendation System')

odf = pd.read_csv('flipkartdata.csv')
sdf = pd.read_csv('similar_feat.csv')


def get_recommendation(id):
    rec_df = odf.iloc[sdf.iloc[id][1:]]
    return rec_df

st.header('Flipkart women dresses Recommendation System')


product_list = odf['item'].values


selected_product = st.selectbox(
    "Type or select a product from the dropdown",
    range(len(product_list)),
    format_func=lambda x: product_list[x]
)

col1, col2 = st.columns((1,2))

with col1:
    st.markdown(
    f"""
    <div class="container" style="float:right;">
        <img class="logo-img" src="{odf['image'].values[selected_product]}">
    </div>
    """,
    unsafe_allow_html=True
)
with col2:
    st.markdown('<p style="font-weight: bold;">{}</p>'.format(odf['brand'].values[selected_product]), unsafe_allow_html=True)
    st.markdown('<p>{}</p>'.format(odf['item'].values[selected_product]), unsafe_allow_html=True)
    st.markdown('<p><span style="font-size: 18px;color: #B12704;">₹{}</span>&nbsp;<span style="color: grey; text-decoration: line-through; color: #565959;">₹{}</span></p>'.format(odf['discount_price'].values[selected_product], odf['price'].values[selected_product]), unsafe_allow_html=True)


st.markdown('<p style="font-size: 1.2rem;">Products related to this item:</p>', unsafe_allow_html=True)

st.write()

rec_df = get_recommendation(selected_product)

count = 0

for i in range(4):
    for j in st.columns(3):
        
        if count == 10:
            break
        with j:
            st.image(rec_df['image'].values[count])
            st.markdown('<p style="text-align: center; font-weight: bold;">{}</p>'.format(rec_df['brand'].values[count]), unsafe_allow_html=True)
            st.markdown('<p style="text-align: center;">{}</p>'.format(rec_df['item'].values[count]), unsafe_allow_html=True)
            st.markdown('<p style="text-align: center;"><span style="font-size: 18px;color: #B12704;">₹{}</span>&nbsp;<span style="color: grey; text-decoration: line-through; color: #565959;">₹{}</span></p>'.format(rec_df['discount_price'].values[count], rec_df['price'].values[count]), unsafe_allow_html=True)
                                                                                                                                                                                                                                                                      
        count += 1
