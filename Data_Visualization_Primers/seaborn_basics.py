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


# ex1
# plot a bar chart for comparing with results from other visualizations later
# plt.figure(figsize=(15,6.5))
# sns.set_style('darkgrid')
# g = sns.barplot(data=df, x='countries', y='emission_2018',
#                 ci=False, palette='viridis_r')
# g.set_xticklabels(df['countries'], rotation=55, fontdict={'fontsize':10})
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


# # ex2
# Plot a circular bar chart with the DataFrame
# import math

# plt.gcf().set_size_inches(12, 12)
# sns.set_style('darkgrid')

# #set max value
# max_val = max(df['emission_2018'])*1.01
# ax = plt.subplot(projection='polar')

# #set the subplot 
# ax.set_theta_zero_location('N')
# ax.set_theta_direction(1)
# ax.set_rlabel_position(0)
# ax.set_thetagrids([], labels=[])
# ax.set_rgrids(range(len(df)), labels= df['countries'])

# #set the projection
# ax = plt.subplot(projection='polar')

# for i in range(len(df)):
#     ax.barh(i, list(df['emission_2018'])[i]*2*np.pi/max_val,
#             label=list(df['countries'])[i], color=pal_vi[i])

# plt.legend(bbox_to_anchor=(1, 1), loc=2)
# plt.show()


# # ex3
# # Plot a circular bar chart with the sorted DataFrame
# import math

# plt.gcf().set_size_inches(12, 12)
# sns.set_style('darkgrid')

# #set max value
# max_val = max(df_s['emission_2018'])*1.01
# ax = plt.subplot(projection='polar')

# for i in range(len(df)):
#     ax.barh(i, list(df_s['emission_2018'])[i]*2*np.pi/max_val,
#             label=list(df_s['countries'])[i], color=pal_plas[i])

# #set the subplot 
# ax.set_theta_zero_location('N')
# ax.set_theta_direction(1)
# ax.set_rlabel_position(0)
# ax.set_thetagrids([], labels=[])
# ax.set_rgrids(range(len(df)), labels= df_s['countries'])

# #set the projection
# ax = plt.subplot(projection='polar')
# plt.legend(bbox_to_anchor=(1, 1), loc=2)
# plt.show()


# # ex4
# # Starting from the center with a Radial bar chart
# plt.figure(figsize=(12,12))
# ax = plt.subplot(111, polar=True)
# plt.axis()

# #set min and max value
# lowerLimit = 0
# max_v = df['emission_2018'].max()

# #set heights and width
# heights = df['emission_2018']
# width = 2*np.pi / len(df.index)

# #set index and angle
# indexes = list(range(1, len(df.index)+1))
# angles = [element * width for element in indexes]

# bars = ax.bar(x=angles, height=heights, width=width, bottom=lowerLimit,
#                 linewidth=1, edgecolor="white", color=pal_vi)
# labelPadding = 15

# for bar, angle, height, label in zip(bars,angles, heights, df['countries']):
#     rotation = np.rad2deg(angle)
#     alignment = ""
#     #deal with alignment
#     if angle >= np.pi/2 and angle < 3*np.pi/2:
#         alignment = "right"
#         rotation = rotation + 180
#     else: 
#         alignment = "left"
#     ax.text(x=angle, y=lowerLimit + bar.get_height() + labelPadding,
#             s=label, ha=alignment, va='center', rotation=rotation, 
#             rotation_mode="anchor")
#     ax.set_thetagrids([], labels=[])
# plt.show()


# # ex5
# # Plot a radial bar chart with the sorted DataFrame
# plt.figure(figsize=(12,12))
# ax = plt.subplot(111, polar=True)
# plt.axis()

# #set min and max value
# lowerLimit = 0
# max_v = df_s['emission_2018'].max()

# #set heights and width
# heights = df_s['emission_2018']
# width = 2*np.pi / len(df_s.index)

# #set index and angle
# indexes = list(range(1, len(df_s.index)+1))
# angles = [element * width for element in indexes]

# bars = ax.bar(x=angles, height=heights, width=width, bottom=lowerLimit,
#                 linewidth=1, edgecolor="white", color=pal_plas)
# labelPadding = 15

# for bar, angle, height, label in zip(bars,angles, heights, df_s['countries']):
#     rotation = np.rad2deg(angle)
#     alignment = ""
#     #deal with alignment
#     if angle >= np.pi/2 and angle < 3*np.pi/2:
#         alignment = "right"
#         rotation = rotation + 180
#     else: 
#         alignment = "left"
#     ax.text(x=angle, y=lowerLimit + bar.get_height() + labelPadding,
#             s=label, ha=alignment, va='center', rotation=rotation, 
#             rotation_mode="anchor")
#     ax.set_thetagrids([], labels=[])
# plt.show()


# # ex6
# # Create an interactive graph using area for comparing with Treemap
# import plotly.express as px

# fig = px.treemap(df, path=[px.Constant('Countries'), 'countries'],
#                     values=df['emission_2018'],
#                     color=df['emission_2018'],
#                     color_continuous_scale='Spectral_r',
#                     color_continuous_midpoint=np.average(df['emission_2018'])
#                 )
# fig.update_layout(margin = dict(t=50, l=25, r=25, b=25))
# fig.show()


# # ex7
# # Waffle chart
from pywaffle import Waffle

# fig = plt.figure(FigureClass=Waffle, 
#                     rows=20,
#                     columns=50,
#                     values=list(df_s['emission_2018']), 
#                     colors=pal_spec,
#                     labels=[i+' '+format(j, ',') for i,j in zip(df_s['countries'], df_s['emission_2018'])],
#                     figsize = (15,6),
#                     legend={'loc':'upper right',
#                             'bbox_to_anchor': (1.26, 1)
#                     }
#                 )
# plt.tight_layout() 
# plt.show()


# # ex7a
# # Plot each country’s waffle chart
# # To avoid the difficulty in reading, let’s plot 
# # each country, one by one, against the other countries
# # Note: each country will display on a separate graph 
# # and each graph will be saved as a png to the parent folder
# save_name = []
# for i,p,n,c in zip(df_s['emission_2018'], df_s['percentage'], df_s['countries'], pal_hsv):
#     fig = plt.figure(FigureClass=Waffle,
#                         rows=10, columns=20,
#                         values=[i, sum(df_s['emission_2018'])-i], 
#                         colors=[c,'gainsboro'],
#                         labels=[n + ' ' + str(round(p,1)) +' %','Other countries'],
#                         figsize = (8,8),
#                         legend={'loc':'upper right', 'bbox_to_anchor': (1, 1), 'fontsize':24}
#                     )
#     save_name.append('waffle_'+ n + '.png')
#     plt.tight_layout()
#     plt.savefig('waffle_'+ n + '.png', bbox_inches='tight')   #export_fig
#     plt.show()
#     plt.close()


# # ex7b
# # create a collage of each country's waffle chart
# from PIL import Image

# def get_collage(cols_n, rows_n, width, height, input_sname, save_name):
#     c_width = width//cols_n
#     c_height = height//rows_n
#     size = c_width, c_height
#     new_im = Image.new('RGB', (width, height))
#     ims = []
#     for p in input_sname:
#         im = Image.open(p)
#         im.thumbnail(size)
#         ims.append(im)
#     i, x, y = 0,0,0
    
#     for col in range(cols_n):
#         for row in range(rows_n):
#             print(i, x, y)
#             try:
#                 new_im.paste(ims[i], (x, y))
#                 i += 1
#                 y += c_height
#             except IndexError:
#                 pass
#         x += c_width
#         y = 0
#     new_im.save(save_name)

# # to create a fit photo collage: 
# # width = number of columns * figure width
# # height = number of rows * figure height
# get_collage(5, 5, 2840, 1445, save_name, 'Collage_waffle.png')


# # ex8
# # Plot an interactive bar chart
# import plotly.express as px

# fig = px.bar(df, x='countries', y='emission_2018', text='emission_2018',
#                 color ='countries', color_discrete_sequence=pal_vi)

# fig.update_traces(texttemplate='%{text:.3s}', textposition='outside')
# fig.update_layout({'plot_bgcolor': 'white',
#                     'paper_bgcolor': 'white'})

# fig.update_layout(width=1100, height=500,
#                     margin = dict(t=15, l=15, r=15, b=15))
# fig.show()


# # ex9
# # Showing percentages in an interactive pie chart
# import plotly.express as px

# fig = px.pie(df_s, values='emission_2018', names='countries',
#                 color ='countries', color_discrete_sequence=pal_vi)
# fig.update_traces(textposition='inside',
#                     textinfo='percent+label',
#                     sort=False)
# fig.update_layout(width=1000, height=550)
# fig.show()



# It appears that ex10, ex11, ex12 can no longer be executed without
# using a version of Pandas earlier than 2.x
# # Get pandas Version
# print(pd.__version__)

# ex10
# Plotting around a circle with a Radar chart
# import plotly.express as px

# fig = px.line_polar(df, r='emission_2018', 
#                     theta='countries', line_close=True
#                     )
# fig.update_traces(fill='toself', 
#                     line = dict(color=pal_spec[5])
#                 )
# fig.show()


# # ex11
# # Plot a radar chart with the sorted DataFrame.
# import plotly.express as px

# fig = px.line_polar(df_s, r='emission_2018',
#                     theta='countries', line_close=True)
# fig.update_traces(fill='toself', line = dict(color=pal_spec[-5]))
# fig.show()


# # ex12
# # Using many circles with a Bubble chart
# # the code below shows how to plot the bubbles vertically
# # If you want to plot the bubbles in a horizontal direction, 
# # alternate the values between the X and Y columns
# #X-axis and Y-axis column
# df_s['X'] = [1]*len(df_s)
# list_y = list(range(0,len(df_s)))
# list_y.reverse()
# df_s['Y'] = list_y

# #labels column
# df_s['labels'] = ['<b>'+i+'<br>'+format(j, ",") for i,j in zip(df_s['countries'], df_s['emission_2018'])]
# df_s


# import plotly.express as px

# fig = px.scatter(df_s, x='X', y='Y',
#                     color='countries', color_discrete_sequence=pal_vi,
#                     size='emission_2018', text='labels', size_max=30
#                 )

# fig.update_layout(width=500, height=1100,
#                     margin = dict(t=0, l=0, r=0, b=0),
#                     showlegend=False
#                 )

# fig.update_traces(textposition='middle right')
# fig.update_xaxes(showgrid=False, zeroline=False, visible=False)
# fig.update_yaxes(showgrid=False, zeroline=False, visible=False)

# fig.update_layout({"plot_bgcolor": 'white', 
#                     "paper_bgcolor": 'white'})
# fig.show()


# # ex13
# # Plot the bubbles in a circular direction
# # create X and Y coordinates in a circle
# import plotly.express as px
# import math

# e = 360/len(df)
# degree = [i*e for i in list(range(len(df)))]
# df_s['X_coor'] = [math.cos(i*math.pi/180) for i in degree]
# df_s['Y_coor'] = [math.sin(i*math.pi/180) for i in degree]
# df_s


# fig = px.scatter(df_s, x='X_coor', y='Y_coor',
#                     color="countries", color_discrete_sequence=pal_vi,
#                     size='emission_2018', text='labels', size_max=40
#                 )
# fig.update_layout(width=800, height=800,
#                     margin = dict(t=0, l=0, r=0, b=0),
#                     showlegend=False
#                 )
# fig.update_traces(textposition='bottom center')
# fig.update_xaxes(showgrid=False, zeroline=False, visible=False)
# fig.update_yaxes(showgrid=False, zeroline=False, visible=False)

# fig.update_layout({'plot_bgcolor': 'white',
#                     'paper_bgcolor': 'white'})
# fig.show()


# ex14
# Clustering the bubbles with Circle packing
import circlify

# compute circle positions:
circles = circlify.circlify(df_s['emission_2018'].tolist(), 
                            show_enclosure=False, 
                            target_enclosure=circlify.Circle(x=0, y=0)
                            )
circles.reverse()

# Plot the circle packing

fig, ax = plt.subplots(figsize=(14, 14), facecolor='white')
ax.axis('off')
lim = max(max(abs(circle.x)+circle.r, abs(circle.y)+circle.r,) for circle in circles)
plt.xlim(-lim, lim)
plt.ylim(-lim, lim)

# print circles
for circle, label, emi, color in zip(circles, df_s['countries'], df_s['emission_2018'], pal_vi):
    x, y, r = circle
    ax.add_patch(plt.Circle((x, y), r, alpha=0.9, color = color))
    plt.annotate(label +'\n'+ format(emi, ","), (x,y), size=15, va='center', ha='center')
plt.xticks([])
plt.yticks([])
plt.show()







