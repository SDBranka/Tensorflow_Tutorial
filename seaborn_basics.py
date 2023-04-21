import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import requests 
from bs4 import BeautifulSoup


wikiurl = 'https://en.wikipedia.org/wiki/List_of_countries_by_carbon_dioxide_emissions'
table_class='wikitable sortable jquery-tablesorter'

response = requests.get(wikiurl)
# Check that server answered the call
# # status 200: The server successfully answered the http request 
# print(response.status_code)

# parse the information and pull the table from the response
soup = BeautifulSoup(response.text, 'html.parser')
table = soup.find('table',{'class':"wikitable"})
# print(table)


# format data from the extracted table
df2018 = pd.read_html(str(table))[0]
# print(df2018)
# #                         Country[19]  ...   2018 CO2 emissions[20]
# #                         Country[19]  ... Total excluding LUCF[22]
# # 0                             World  ...                 35247.21
# # 1    World – International Aviation  ...                      NaN
# # 2    World – International Shipping  ...                      NaN
# # 3                       Afghanistan  ...                     7.44
# # 4                           Albania  ...                     5.56
# # ..                              ...  ...                      ...
# # 207                         Vietnam  ...                   257.86
# # 208                  Western Sahara  ...                      NaN
# # 209                           Yemen  ...                     9.31
# # 210                          Zambia  ...                     7.74
# # 211                        Zimbabwe  ...                    12.27

# # [212 rows x 11 columns]


#get lists of data
emi_ = df2018[('2018 CO2 emissions[20]', 'Total excluding LUCF[22]')]
country_ = list(df2018[('Country[19]', 'Country[19]')])
# print(country_)
country_mod = [i.replace('\xa0',' ') for i in country_]
# print(country_mod)

# #create a DataFrame
df = pd.DataFrame(zip(country_mod,emi_), columns = ['countries', 'emission_2018'])


# select the last column and filter only countries with CO2 emissions between 200 to 1000 MTCO2e
# #remove the row of countries that cannot be converted
df = df[df['countries']!='Serbia & Montenegro']
df = df[df['countries']!='World – International Aviation']  
df = df[df['countries']!='World – International Shipping']  
df = df[df['countries']!='Western Sahara']  
df = df[df['countries']!='Anguilla']  
df = df[df['countries']!='Aruba']  
df = df[df['countries']!='Bermuda']  
df = df[df['countries']!='British Virgin Islands']  
df = df[df['countries']!='Cayman Islands']  
df = df[df['countries']!='Congo']  
df = df[df['countries']!='Curaçao']  
df = df[df['countries']!='East Timor']  
df = df[df['countries']!='Falkland Islands']  
df = df[df['countries']!='Faroe Islands']  
df = df[df['countries']!='French Guiana']  
df = df[df['countries']!='French Polynesia']  
df = df[df['countries']!='Gibraltar']  
df = df[df['countries']!='Greenland']  
df = df[df['countries']!='Guadeloupe']  
df = df[df['countries']!='Hong Kong']  
df = df[df['countries']!='Macau']  
df = df[df['countries']!='Martinique']  
df = df[df['countries']!='New Caledonia']  
df = df[df['countries']!='Puerto Rico']  
df = df[df['countries']!='Réunion']  
df = df[df['countries']!='Saint Helena, Ascension and Tristan da Cunha']  
df = df[df['countries']!='Saint Pierre and Miquelon']  
df = df[df['countries']!='Taiwan']  
df = df[df['countries']!='Turks and Caicos Islands']  
df = df[df['countries']!='Western Sahara']  
df = df[df['countries']!='Russia']  
df = df[df['countries']!='China']  
df = df[df['countries']!='United States']  
df = df[df['countries']!='India']  
df = df[df['countries']!='Japan']  
# print(df)

# print(df.iloc[:,1])
# convert the data brought in as the emissions_2018 column to floats
df.iloc[:,1] = df.iloc[:,1].astype('float')
# select the range
df = df[(df['emission_2018']>200) & (df['emission_2018']<1000)]
df['percentage'] = [i*100/sum(df['emission_2018']) for i in df['emission_2018']]
# print(df.head(9))
# print(df)

# sort by emission_2018 in descending order
df_s = df.sort_values(by='emission_2018', ascending=False)
# print(df_s.head(9))

# plot a bar chart for comparing with results from other visualizations later
plt.figure(figsize=(15,6.5))
sns.set_style('darkgrid')
g = sns.barplot(data=df, x='countries', y='emission_2018',
                ci=False, palette='viridis_r')
g.set_xticklabels(df['countries'], rotation=55, fontdict={'fontsize':10})
# plt.show()


# define a function to extract a list of colors for later use with
# each visualization
def get_color(name, number):
    pal = list(sns.color_palette(palette=name, n_colors=number).as_hex())
    return pal

# Apply the function to get a list of colors.
pal_vi = get_color('viridis_r', len(df))
pal_plas = get_color('plasma_r', len(df))
pal_spec = get_color('Spectral', len(df))
pal_hsv = get_color('hsv', len(df))


# Plot a circular bar chart with the DataFrame
import math
plt.gcf().set_size_inches(12, 12)
sns.set_style('darkgrid')

#set max value
max_val = max(df['emission_2018'])*1.01
ax = plt.subplot(projection='polar')

#set the subplot 
ax.set_theta_zero_location('N')
ax.set_theta_direction(1)
ax.set_rlabel_position(0)
ax.set_thetagrids([], labels=[])
ax.set_rgrids(range(len(df)), labels= df['countries'])

#set the projection
ax = plt.subplot(projection='polar')

for i in range(len(df)):
    ax.barh(i, list(df['emission_2018'])[i]*2*np.pi/max_val,
            label=list(df['countries'])[i], color=pal_vi[i])

plt.legend(bbox_to_anchor=(1, 1), loc=2)
# plt.show()


# Plot a circular bar chart with the sorted DataFrame
import math
plt.gcf().set_size_inches(12, 12)
sns.set_style('darkgrid')

#set max value
max_val = max(df_s['emission_2018'])*1.01
ax = plt.subplot(projection='polar')

for i in range(len(df)):
    ax.barh(i, list(df_s['emission_2018'])[i]*2*np.pi/max_val,
            label=list(df_s['countries'])[i], color=pal_plas[i])

#set the subplot 
ax.set_theta_zero_location('N')
ax.set_theta_direction(1)
ax.set_rlabel_position(0)
ax.set_thetagrids([], labels=[])
ax.set_rgrids(range(len(df)), labels= df_s['countries'])

#set the projection
ax = plt.subplot(projection='polar')
plt.legend(bbox_to_anchor=(1, 1), loc=2)
# plt.show()


# Starting from the center with a Radial bar chart
plt.figure(figsize=(12,12))
ax = plt.subplot(111, polar=True)
plt.axis()

#set min and max value
lowerLimit = 0
max_v = df['emission_2018'].max()

#set heights and width
heights = df['emission_2018']
width = 2*np.pi / len(df.index)

#set index and angle
indexes = list(range(1, len(df.index)+1))
angles = [element * width for element in indexes]

bars = ax.bar(x=angles, height=heights, width=width, bottom=lowerLimit,
                linewidth=1, edgecolor="white", color=pal_vi)
labelPadding = 15

for bar, angle, height, label in zip(bars,angles, heights, df['countries']):
    rotation = np.rad2deg(angle)
    alignment = ""
    #deal with alignment
    if angle >= np.pi/2 and angle < 3*np.pi/2:
        alignment = "right"
        rotation = rotation + 180
    else: 
        alignment = "left"
    ax.text(x=angle, y=lowerLimit + bar.get_height() + labelPadding,
            s=label, ha=alignment, va='center', rotation=rotation, 
            rotation_mode="anchor")
    ax.set_thetagrids([], labels=[])
# plt.show()


# Plot a radial bar chart with the sorted DataFrame
plt.figure(figsize=(12,12))
ax = plt.subplot(111, polar=True)
plt.axis()

#set min and max value
lowerLimit = 0
max_v = df_s['emission_2018'].max()

#set heights and width
heights = df_s['emission_2018']
width = 2*np.pi / len(df_s.index)

#set index and angle
indexes = list(range(1, len(df_s.index)+1))
angles = [element * width for element in indexes]

bars = ax.bar(x=angles, height=heights, width=width, bottom=lowerLimit,
                linewidth=1, edgecolor="white", color=pal_plas)
labelPadding = 15

for bar, angle, height, label in zip(bars,angles, heights, df_s['countries']):
    rotation = np.rad2deg(angle)
    alignment = ""
    #deal with alignment
    if angle >= np.pi/2 and angle < 3*np.pi/2:
        alignment = "right"
        rotation = rotation + 180
    else: 
        alignment = "left"
    ax.text(x=angle, y=lowerLimit + bar.get_height() + labelPadding,
            s=label, ha=alignment, va='center', rotation=rotation, 
            rotation_mode="anchor")
    ax.set_thetagrids([], labels=[])
# plt.show()


# Create an interactive graph using area for comparing with Treemap
import plotly.express as px
fig = px.treemap(df, path=[px.Constant('Countries'), 'countries'],
                    values=df['emission_2018'],
                    color=df['emission_2018'],
                    color_continuous_scale='Spectral_r',
                    color_continuous_midpoint=np.average(df['emission_2018'])
                )
fig.update_layout(margin = dict(t=50, l=25, r=25, b=25))
# fig.show()


# Waffle chart
from pywaffle import Waffle

fig = plt.figure(FigureClass=Waffle, 
                    rows=20,
                    columns=50,
                    values=list(df_s['emission_2018']), 
                    colors=pal_spec,
                    labels=[i+' '+format(j, ',') for i,j in zip(df_s['countries'], df_s['emission_2018'])],
                    figsize = (15,6),
                    legend={'loc':'upper right',
                            'bbox_to_anchor': (1.26, 1)
                    }
                )
plt.tight_layout() 
plt.show()






