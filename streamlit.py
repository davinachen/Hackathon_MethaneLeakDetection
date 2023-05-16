from utils import *
import streamlit as st
from streamlit_folium import st_folium
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from PIL import Image
import tempfile
import folium
from geopy.geocoders import Nominatim
import requests
import json
import pandas as pd
from PIL import Image
from folium.plugins import Draw

# Load previously trained model 
model = torch.load('./data/models/dataResNet_pretrained_resnet50.pt')


# Test image transformation
image_transforms = { 
    'test': transforms.Compose([
        transforms.CenterCrop(size=64),
        transforms.ToTensor()
    ])
}

# Define prediction function
def predict(input, model):
    # Evaluation mode
    model.eval()
    # Add batch dimension 
    input = input.unsqueeze(0)
    # Get predicted category for image
    with torch.no_grad():
        outputs = model(input)
        ret, prediction = torch.max(outputs.data, 1)
    return prediction

# Streamlit Interface Building

head = st.container()
image_upload = st.container()
map_coord = st.container()
dashboard = st.container()

with head: 
    st.title(' ')
#     logo = Image.open('logo.png')
#     st.image(logo, width=200)
#     st.write('Clearing the Air, One Emission at a Time.')

with image_upload: 
    centered_text1 = """
    <div style="text-align: center;">
        <h3>CleanR Methane Detection Model</h1>
        <p>Upload an image for methane leaks detection:</p>
    </div>
    """

    # Display centered Markdown text
    st.markdown(centered_text1, unsafe_allow_html=True)
    col1, col2 = st.columns([2, 1])
    input = col1.file_uploader("Choose a file", type='tif')
    if input is not None:
        #create a temp dir to store the image file
        temp_dir = tempfile.TemporaryDirectory()
        file_path = os.path.join(temp_dir.name, input.name)
        with open(file_path, 'wb') as f:
            f.write(input.read())
        
        #show the image
        image = Image.open(input)
        converted_image = image.convert("RGB")
        #zoom in the image
        col2.image(converted_image, caption='uploaded image', use_column_width=True)

        #check the file path and make prediction
        image = Image.open(file_path).convert('RGB') # supposedly the code is rgb and not grey scale
        image = image_transforms['test'](image)

        prediction = predict(image, model)
        result = 'Yes' if prediction == 1 else 'No'
        col1.write("Prediction of methane leak: "+ result)

    st.markdown("<hr>", unsafe_allow_html=True)

with map_coord: 
    #radius slider
    enter_radius = st.slider('Radius (meters): ', min_value = 1000, max_value = 10000, value = 10000, step = 1000)
    c1, c2 = st.columns([2,1])

    #read metadata and get image coordinates
    df = pd.read_csv('data/metadata.csv')
    filtered_df = df[df['path'].str.contains(input.name[:-4])]
    lat = filtered_df['lat'].values[0]
    lon = filtered_df['lon'].values[0]

    #show coordinates and find relevant info like country and city 
    geolocator = Nominatim(user_agent="qbhackathon")

    def get_location_by_coordinates(lat, long):
        location = geolocator.reverse([lat, long],language='en')
        return location

    latitude = lat
    longitude = lon

    m = folium.Map(location=[latitude, longitude], zoom_start=10)
    folium.Marker(location=[latitude, longitude],popup='Satellite Image', color='lightgray').add_to(m)
    # m.add_child(folium.LatLngPopup()) #only if we want to see the coord when click
    
    #functuion to fetch data from overpass api
    def get_nearby_data(lat, lon, radius=10000, purpose='"landuse"="industrial"'):
        overpass_url = "http://overpass-api.de/api/interpreter"
        query = f"""
        [out:json];
        (
        node[{purpose}](around:{radius},{lat},{lon});
        way[{purpose}](around:{radius},{lat},{lon});
        relation[{purpose}](around:{radius},{lat},{lon});
        );
        out center;
        """
        response = requests.get(overpass_url, params={'data': query})
        data = response.json()
        return data

    #checkbox to show points on map
    show_industrial_sites = c2.checkbox('Industrial Sites')
    show_landfill = c2.checkbox('Landfills')
    gasplants = c2.checkbox('Oil Refineries & Natural Gas Processing Plants')

    #get industrial sites with overpass api
    industrial_sites = get_nearby_data(lat, lon ,radius=enter_radius)
    ind_site_eng_list = [item['tags']['name:en'] for item in industrial_sites['elements'] if 'name:en' in item['tags']]
    ind_site_name_list = [item['tags']['name'] for item in industrial_sites['elements'] if 'name' in item['tags']]
    
    #checkbox checked, then show points
    if show_industrial_sites:
        for element in industrial_sites['elements']:
            if element['type'] == 'node':
                folium.Marker([element['lat'], element['lon']], icon=folium.Icon(color="red")).add_to(m)
            elif 'center' in element:
                folium.Marker([element['center']['lat'], element['center']['lon']]\
                            , icon= folium.Icon(icon="industry", prefix='fa', color="green") if 'name' in element['tags'] \
                                else folium.Icon(icon="industry", prefix='fa', color="red") \
                                , popup= 'Major Industrial Site' if 'name' in element['tags'] else 'Industrial Site').add_to(m)

    #get landfill info with overpass api
    landfill = get_nearby_data(lat, lon ,radius=enter_radius, purpose='"landuse"="landfill"')
    lf_site_eng_list = [item['tags']['name:en'] for item in landfill['elements'] if 'name:en' in item['tags']]
    lf_site_name_list = [item['tags']['name'] for item in landfill['elements'] if 'name' in item['tags']]

    if show_landfill: 
        for element in landfill['elements']:
            if element['type'] == 'node':
                folium.Marker([element['lat'], element['lon']], folium.Icon(icon="trash", prefix='fa', color='lightblue')).add_to(m)
            elif 'center' in element:
                folium.Marker([element['center']['lat'], element['center']['lon']]\
                            , icon= folium.Icon(icon="trash", prefix='fa', color='lightblue'), popup= 'Landfill Sites').add_to(m)

    #get oil refineries or natural gas processing plants
    plants = get_nearby_data(lat, lon ,radius=enter_radius, purpose='"man_made"="works"')
    ngpp_site_eng_list = [item['tags']['name:en'] for item in plants['elements'] if 'name:en' in item['tags']]
    ngpp_site_name_list = [item['tags']['name'] for item in plants['elements'] if 'name' in item['tags']]

    if gasplants: 
        for element in plants['elements']:
            if element['type'] == 'node':
                folium.Marker([element['lat'], element['lon']], folium.Icon(icon="fire", prefix='fa', color='pink')).add_to(m)
            elif 'center' in element:
                folium.Marker([element['center']['lat'], element['center']['lon']]\
                            , popup= 'Gas Plants', icon= folium.Icon(icon="fire", prefix='fa', color='pink')).add_to(m)

    with c1: 
        #add map and draw tools to left column
        Draw(export=True).add_to(m)
        output = st_folium(m)

    with c2: 
        #add coordinates info and relevant info to the right column
        location = get_location_by_coordinates(latitude, longitude)
        with st.expander("Coordinates Information: "):
            st.write(lat, lon)
            st.write('Country: ', location.raw['address']['country'])
            st.write('State: ', location.raw['address']['state'])

        #print relevant info in the area of the coordinates
        st.markdown(f'**Major industrial sites**: ')
        for item_eng in ind_site_eng_list: 
            st.write(f'- ', item_eng)
        for item_name in ind_site_name_list: 
            st.write(f'- ', item_name)
        st.markdown(f'**Major landfill sites**: ')
        for item_eng in lf_site_eng_list: 
            st.write(f'- ', item_eng)
        for item_name in lf_site_name_list: 
            st.write(f'- ', item_name)
        st.markdown(f'**Major Oil/Gas plants**: ')
        for item_eng in ngpp_site_eng_list: 
            st.write(f'- ', item_eng)
        for item_name in ngpp_site_name_list: 
            st.write(f'- ', item_name)
    st.markdown("<hr>", unsafe_allow_html=True)

with dashboard: 

    centered_text = """
    <div style="text-align: center;">
        <h3>CleanR Monitoring Dashboard</h1>
        <p>Real-time monitoring on all sites</p>
    </div>
    """

    # Display centered Markdown text
    st.markdown(centered_text, unsafe_allow_html=True)

    file = st.file_uploader('Upload 5 monitoring areas: ', type='tif', accept_multiple_files=True)
    c3, c4, c5, c6, c7 = st.columns(5)
    name_list = ['Joshua', 'Davina', 'Aline', 'Namrata', 'Ugo']
    pos_neg = ['+', '-']
    if file is not None:
        #create a temp dir to store the image file
        temp_dir = tempfile.TemporaryDirectory() 

        #initialise the map
        m = folium.Map(zoom_start=5)

        for i in range(5):
            file_path = os.path.join(temp_dir.name, file[i].name)
            with open(file_path, 'wb') as f:
                f.write(file[i].read())
            image = Image.open(file[i])
            converted_image = image.convert("RGB")

            column = [c3, c4, c5, c6, c7][i]
            column.image(converted_image, caption=f'site {i+1}', use_column_width=True)

            #make a button to show side bar information
            if column.button(f'Site Information', key=f'SiteInformationButton{i}'):
                with st.sidebar:
                    filtered_df = df[df['path'].str.contains(file[i].name[:-4])]
                    lat = filtered_df['lat'].values[0]
                    lon = filtered_df['lon'].values[0]
                    location = get_location_by_coordinates(lat, lon)
                    st.write((lat,lon))
                    st.write('Country: ', location.raw['address']['country'])
                    st.write('State: ', location.raw['address']['state'])
                    st.write('Site Manager: ', name_list[i])
                    bar1 = st.progress(80, text='Production Volume')
                    bar2 = st.progress(90, text='Utilization Rate')
                    st.metric(label="Temperature", value=f'{random.randint(10,90)} °C', delta=f'{random.choice(pos_neg)}{round(random.uniform(0.5,3),2)} °C')
                    with st.expander("Local Authorities info: "):
                        st.button('For Regulatory Compliance')
                        st.button('For Reporting and Communication')
                        st.button('For Incident Management')


            #check the file path and make prediction
            image = Image.open(file[i]).convert('RGB') # supposedly the code is rgb and not grey scale
            image = image_transforms['test'](image)

            prediction = predict(image, model)
            result = 'Yes' if prediction == 1 else 'No'
            column.write("Prediction of methane leak: "+ result)

            #adding site points to a map 
            filtered_df = df[df['path'].str.contains(file[i].name[:-4])]
            lat = filtered_df['lat'].values[0]
            lon = filtered_df['lon'].values[0]

            # coordinates and relevant info 
            geolocator = Nominatim(user_agent="qbhackathon1")

            def get_location_by_coordinates(lat, long):
                location = geolocator.reverse([lat, long],language='en')
                return location

            latitude = lat
            longitude = lon
            
            if result == 'No':
                folium.Marker(location=[latitude, longitude],popup=f'Site {i+1}', icon=folium.Icon(icon="fire", prefix='fa', color='red')).add_to(m)
            else: 
                folium.Marker(location=[latitude, longitude],popup=f'Site {i+1}', icon=folium.Icon(color="green")).add_to(m)
        #add draw option
        Draw(export=True).add_to(m)
        #display map
        output = st_folium(m, width='100%')
    

