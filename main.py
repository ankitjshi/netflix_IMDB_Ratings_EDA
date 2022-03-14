import numpy as np # linear algebra
import pandas as pd # data processing
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings("ignore")
from wordcloud import WordCloud
import plotly.express as px
import plotly.graph_objects as go
from plotly.offline import init_notebook_mode, iplot
import datetime
import io
import PIL.Image as Image
from IPython.display import display
from tabulate import tabulate



##https://www.kaggle.com/niharika41298/netflix-visualizations-recommendation-eda

def fix_date(year):
    if year > 2021:
        year = year - 100
    return year

def geoplot(ddf):
    country_with_code, country = {}, {}
    shows_countries = ", ".join(ddf['Country_Availability'].dropna()).split(", ")
    for c, v in dict(Counter(shows_countries)).items():
        code = ""
        if c.lower() in country_codes:
            code = country_codes[c.lower()]
        country_with_code[code] = v
        country[c] = v

    data = [dict(
        type='choropleth',
        locations=list(country_with_code.keys()),
        z=list(country_with_code.values()),
        colorscale=[[0, "rgb(5, 10, 172)"], [0.65, "rgb(40, 60, 190)"], [0.75, "rgb(70, 100, 245)"], \
                    [0.80, "rgb(90, 120, 245)"], [0.9, "rgb(106, 137, 247)"], [1, "rgb(220, 220, 220)"]],
        autocolorscale=False,
        reversescale=True,
        marker=dict(
            line=dict(
                color='gray',
                width=0.5
            )),
        colorbar=dict(
            autotick=False,
            title=''),
    )]

    layout = dict(
        title='',
        geo=dict(
            showframe=False,
            showcoastlines=False,
            projection=dict(
                type='Mercator'
            )
        )
    )

    fig = dict(data=data, layout=layout)
    iplot(fig, validate=False, filename='d3-world-map')
    return country

def country_trace(country, flag="movie"):
    netflix_new_actor["from_us"] = netflix_new_actor['Country_Availability'].fillna("").apply(
        lambda x: 1 if country.lower() in x.lower() else 0)
    small = netflix_new_actor[netflix_new_actor["from_us"] == 1]
    cast = ", ".join(small['Actors'].fillna("")).split(", ")
    tags = Counter(cast).most_common(25)
    tags = [_ for _ in tags if "" != _[0]]

    labels, values = [_[0] + "  " for _ in tags], [_[1] for _ in tags]
    trace = go.Bar(y=labels[::-1], x=values[::-1], orientation="h", name="", marker=dict(color="#a678de"))
    return trace

if __name__ == '__main__':
    pd.set_option('display.max_columns', None)  # or 1000
    pd.set_option('display.max_rows', None)  # or 1000
    pd.set_option('display.max_colwidth', None)  # or 199

    netflix = pd.read_csv(r'C:\Users\ankit\PycharmProjects\pythonProject\Netflix_Dataset_Latest_2021.csv')
    print("********************* HEAD ******************************")
    print(netflix.head(10).to_markdown())
    print("********************* SHAPE ******************************")
    print(netflix.shape)
    print("********************* COLUMNS ******************************")
    print(netflix.columns)
    print("********************* ISNULL ******************************")
    print(netflix.isnull().sum())
    print("********************* UNIQUE ******************************")
    print(netflix.nunique())
    print("********************* DUPLICATE ******************************")
    print(netflix.duplicated().sum())
    print("********************* Cleaning ******************************")
    df = netflix[netflix["Type"] == "Movie"]
    # df.drop(['Boxoffice', 'Production_House'], axis=1, inplace=True)
    # df = df.dropna()
    print("********************* Cleaned Data ******************************")
    print(df.shape)
    print("********************* Cleaned ISNULL ******************************")
    print(df.isnull().sum())
    print("********************* Cleaned UNIQUE ******************************")
    print(df.nunique())
    print("********************* Data formatting ******************************")

    # df["Release_Date"] = pd.to_datetime(df['Release_Date'])
    # df['Release_Date_Day'] = df['Release_Date'].dt.day
    # df['Release_Date_Year'] = df['Release_Date'].dt.year
    # df['Release_Date_Month'] = df['Release_Date'].dt.strftime('%B')
    # df['Release_Date_Year'].astype(int);
    # df['Release_Date_Day'].astype(int);
    # df['Release_Date_Year'] = df['Release_Date_Year'].apply(fix_date)
    #
    # df['Netflix_Release_Date'] = pd.to_datetime(df['Netflix_Release_Date'])
    # df['Netflix_Release_Date_Day'] = df['Netflix_Release_Date'].dt.day
    # df['Netflix_Release_Date_Year'] = df['Netflix_Release_Date'].dt.year
    # df['Netflix_Release_Date_Month'] = df['Netflix_Release_Date'].dt.strftime('%B')
    #
    # print("***************** View Ratings Across Series and Movies ******************")
    # plt.figure(figsize=(10, 8))
    # sns.countplot(x='View_Rating', hue='Type', data=netflix)
    # plt.title('View Ratings Across Series and Movies',fontsize=12, fontfamily='calibri', fontweight='bold',
    #           position=(0.20, 1.0 + 0.02))
    # plt.xticks(rotation=90)
    # plt.savefig('ratings.png')
    # plt.show()

    print("***************** Distribution Split of Movies and TV Series ******************")
    labels = ['Movie', 'TV show']
    size = netflix['Type'].value_counts()
    colors = plt.cm.Wistia(np.linspace(0, 1, 2))
    explode = [0, 0.1]
    total = netflix['Type'].value_counts().sum()
    plt.rcParams['figure.figsize'] = (9, 9)
    plt.pie(size, labels=labels, colors=colors, explode=explode, shadow=True, startangle=90, autopct='%1.2f%%')
    plt.title('Distribution Split of Movies and TV Series', fontsize=25)
    plt.legend()
    plt.savefig('distribution.png')
    plt.show()

    print("***************** Most popular Country ******************")
    plt.subplots(figsize=(25, 15))
    wordcloud = WordCloud(
        background_color='white',
        width=1920,
        height=1080
    ).generate(" ".join(df.Country_Availability))
    plt.imshow(wordcloud)
    plt.axis('off')
    plt.savefig('popular_country.png')
    plt.show()

    print("***************** Most popular Star Cast ******************")
    plt.subplots(figsize=(25, 15))
    wordcloud = WordCloud(
        background_color='white',
        width=1920,
        height=1080
    ).generate(" ".join(df.Actors))
    plt.imshow(wordcloud)
    plt.axis('off')
    plt.savefig('popular_star_cast.png')
    plt.show()

    print("***************** Netfix Content Updates across years and months ******************")
#If a producer wants to release some content, which month must he do so?( Month when least amount of content is added)¶
    netflix_shows = df.copy()

    month_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October',
                   'November', 'December'][::-1]
    df = netflix_shows.groupby('Netflix_Release_Date_Year')['Netflix_Release_Date_Month'].value_counts().unstack().fillna(0)[month_order].T
    plt.figure(figsize=(10, 7), dpi=200)
    plt.pcolor(df, cmap='viridis', edgecolors='white', linewidths=2)  # heatmap
    plt.xticks(np.arange(0.5, len(df.columns), 1), df.columns, fontsize=7, fontfamily='serif')
    plt.yticks(np.arange(0.5, len(df.index), 1), df.index, fontsize=7, fontfamily='serif')

    plt.title('Netflix Content Updates across years and months', fontsize=12, fontfamily='calibri', fontweight='bold',
              position=(0.20, 1.0 + 0.02))
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=8)
    cbar.ax.minorticks_on()
    plt.savefig('heatmap_years_months.png')
    plt.show()


    print("***************** Netfix Content Updates across years and months ******************")
    netflix_cat = pd.DataFrame()
    netflix_cat['Title'] = netflix_shows['Title']
    netflix_cat['Genre'] = netflix_shows['Genre']
    netflix_cat['Country_Availability'] = netflix_shows['Country_Availability']
    netflix_cat['IMDb_Score'] = netflix_shows['IMDb_Score']
    netflix_cat['Type'] = netflix_shows['Type']
    netflix_cat['Actors'] = netflix_shows['Actors']
    netflix_cat['Writer'] = netflix_shows['Writer']
    netflix_cat['Director'] = netflix_shows['Director']

    netflix_new_country = netflix_cat.assign(Country_Availability=netflix_cat['Country_Availability'].str.split(',')).explode('Country_Availability')
    netflix_new_cat = netflix_cat.assign(Genre=netflix_cat['Genre'].str.split(',')).explode('Genre')
    netflix_new_actor = netflix_new_country.assign(Actors=netflix_cat['Actors'].str.split(',')).explode('Actors')
    fig = px.sunburst(
        netflix_new_country,
        path=['Title', 'Country_Availability'],
        values='IMDb_Score',
        color='IMDb_Score')
    fig.show()

# Countries with highest rated content.
    print("***************** Top 10 Countries- High Rating Content Published ******************")
    country_count=netflix_new_country.sort_values('IMDb_Score', ascending=False)['Country_Availability']
    topcountriesHighRatings=country_count[0:11]
    print(topcountriesHighRatings)
    print("**************** Top 10 Countries- Maximum Netflix Content Published *******************")
    topcountriesMaxContent=country_count.value_counts().sort_values(ascending=False)[0:11]
    print(topcountriesMaxContent)

# Content Added over Years
    print("**************** Content Added over Years *******************")
    d1 = netflix_shows[netflix_shows["Type"] == "Series"]
    d2 = netflix_shows[netflix_shows["Type"] == "Movie"]

    col = "Netflix_Release_Date_Year"

    vc1 = d1[col].value_counts().reset_index()
    vc1 = vc1.rename(columns={col: "count", "index": col})
    vc1['percent'] = vc1['count'].apply(lambda x: 100 * x / sum(vc1['count']))
    vc1 = vc1.sort_values(col)

    vc2 = d2[col].value_counts().reset_index()
    vc2 = vc2.rename(columns={col: "count", "index": col})
    vc2['percent'] = vc2['count'].apply(lambda x: 100 * x / sum(vc2['count']))
    vc2 = vc2.sort_values(col)

    col_o = "Release_Date_Year"
    vc1_o = d1[col_o].value_counts().reset_index()
    vc1_o = vc1_o.rename(columns={col_o: "count", "index": col_o})
    vc1_o['percent'] = vc1_o['count'].apply(lambda x: 100 * x / sum(vc1_o['count']))
    vc1_o = vc1_o.sort_values(col_o)

    vc2_o = d2[col_o].value_counts().reset_index()
    vc2_o = vc2_o.rename(columns={col_o: "count", "index": col_o})
    vc2_o['percent'] = vc2_o['count'].apply(lambda x: 100 * x / sum(vc2_o['count']))
    vc2_o = vc2_o.sort_values(col_o)

    print("**************** Series : Content Added over Years in comaprison to original release year *******************")

    trace1 = go.Scatter(x=vc1[col], y=vc1["count"], name="Series Netflix", marker=dict(color="#a678de"))
    trace2 = go.Scatter(x=vc1_o[col_o], y=vc1_o["count"], name="Series Original", marker=dict(color="#6ad49b"))
    data = [trace1, trace2]
    layout = go.Layout(title="Series : Content Added over Years in comaprison to original release year", legend=dict(x=0.1, y=1.1, orientation="h"))
    fig = go.Figure(data, layout=layout)
    fig.show()

    print("**************** Movies : Content Added over Years in comparison to original release year *******************")

    trace1 = go.Scatter(x=vc2[col], y=vc2["count"], name="Movies Netflix", marker=dict(color="#a678de"))
    trace2 = go.Scatter(x=vc2_o[col_o], y=vc2_o["count"], name="Movies Original", marker=dict(color="#6ad49b"))
    data = [trace1, trace2]
    layout = go.Layout(title="Movies : Content Added over Years in comparison to original release year ", legend=dict(x=0.1, y=1.1, orientation="h"))
    fig = go.Figure(data, layout=layout)
    fig.show()

    print("**************** Series vs Movies : Content Added over Years *******************")
    trace1 = go.Scatter(x=vc1[col], y=vc1["count"], name="TV Series", marker=dict(color="#a678de"))
    trace2 = go.Scatter(x=vc2[col], y=vc2["count"], name="Movies", marker=dict(color="#6ad49b"))
    data = [trace1, trace2]
    layout = go.Layout(title="Series vs Movies : Content Added over Years", legend=dict(x=0.1, y=1.1, orientation="h"))
    fig = go.Figure(data, layout=layout)
    fig.show()

    print("**************** Series vs Movies : Content Added over Months *******************")
    col = 'Netflix_Release_Date_Month'
    vc1 = d1[col].value_counts().reset_index()
    vc1 = vc1.rename(columns={col: "count", "index": col})
    vc1['percent'] = vc1['count'].apply(lambda x: 100 * x / sum(vc1['count']))
    vc1 = vc1.sort_values(col)

    vc2 = d2[col].value_counts().reset_index()
    vc2 = vc2.rename(columns={col: "count", "index": col})
    vc2['percent'] = vc2['count'].apply(lambda x: 100 * x / sum(vc2['count']))
    vc2 = vc2.sort_values(col)

    trace1 = go.Bar(x=vc1[col], y=vc1["count"], name="TV Series", marker=dict(color="#a678de"))
    trace2 = go.Bar(x=vc2[col], y=vc2["count"], name="Movies", marker=dict(color="#6ad49b"))
    data = [trace1, trace2]
    layout = go.Layout(title="In which month, the conent is added the most?",
                       legend=dict(x=0.1, y=1.1, orientation="h"))
    fig = go.Figure(data, layout=layout)
    fig.show()

    print("******************** Oldest Movies on Netflix *****************************************")

    small = d2.sort_values("Release_Date_Year", ascending=True)
    small = small[small['Runtime'] != ""]
    print(small[['Title', "Release_Date_Year"]][:15])

    print("******************** Oldest TV Series on Netflix *****************************************")
    small = d1.sort_values("Release_Date_Year", ascending=True)
    print(small[['Title', "Release_Date_Year"]][:15])
    # fig.write_image("content1.png")

    print("******************** Content From Different Countries on Netflix *****************************************")

    country_codes = {'afghanistan': 'AFG',
                     'albania': 'ALB',
                     'algeria': 'DZA',
                     'american samoa': 'ASM',
                     'andorra': 'AND',
                     'angola': 'AGO',
                     'anguilla': 'AIA',
                     'antigua and barbuda': 'ATG',
                     'argentina': 'ARG',
                     'armenia': 'ARM',
                     'aruba': 'ABW',
                     'australia': 'AUS',
                     'austria': 'AUT',
                     'azerbaijan': 'AZE',
                     'bahamas': 'BHM',
                     'bahrain': 'BHR',
                     'bangladesh': 'BGD',
                     'barbados': 'BRB',
                     'belarus': 'BLR',
                     'belgium': 'BEL',
                     'belize': 'BLZ',
                     'benin': 'BEN',
                     'bermuda': 'BMU',
                     'bhutan': 'BTN',
                     'bolivia': 'BOL',
                     'bosnia and herzegovina': 'BIH',
                     'botswana': 'BWA',
                     'brazil': 'BRA',
                     'british virgin islands': 'VGB',
                     'brunei': 'BRN',
                     'bulgaria': 'BGR',
                     'burkina faso': 'BFA',
                     'burma': 'MMR',
                     'burundi': 'BDI',
                     'cabo verde': 'CPV',
                     'cambodia': 'KHM',
                     'cameroon': 'CMR',
                     'canada': 'CAN',
                     'cayman islands': 'CYM',
                     'central african republic': 'CAF',
                     'chad': 'TCD',
                     'chile': 'CHL',
                     'china': 'CHN',
                     'colombia': 'COL',
                     'comoros': 'COM',
                     'congo democratic': 'COD',
                     'Congo republic': 'COG',
                     'cook islands': 'COK',
                     'costa rica': 'CRI',
                     "cote d'ivoire": 'CIV',
                     'croatia': 'HRV',
                     'cuba': 'CUB',
                     'curacao': 'CUW',
                     'cyprus': 'CYP',
                     'czech republic': 'CZE',
                     'denmark': 'DNK',
                     'djibouti': 'DJI',
                     'dominica': 'DMA',
                     'dominican republic': 'DOM',
                     'ecuador': 'ECU',
                     'egypt': 'EGY',
                     'el salvador': 'SLV',
                     'equatorial guinea': 'GNQ',
                     'eritrea': 'ERI',
                     'estonia': 'EST',
                     'ethiopia': 'ETH',
                     'falkland islands': 'FLK',
                     'faroe islands': 'FRO',
                     'fiji': 'FJI',
                     'finland': 'FIN',
                     'france': 'FRA',
                     'french polynesia': 'PYF',
                     'gabon': 'GAB',
                     'gambia, the': 'GMB',
                     'georgia': 'GEO',
                     'germany': 'DEU',
                     'ghana': 'GHA',
                     'gibraltar': 'GIB',
                     'greece': 'GRC',
                     'greenland': 'GRL',
                     'grenada': 'GRD',
                     'guam': 'GUM',
                     'guatemala': 'GTM',
                     'guernsey': 'GGY',
                     'guinea-bissau': 'GNB',
                     'guinea': 'GIN',
                     'guyana': 'GUY',
                     'haiti': 'HTI',
                     'honduras': 'HND',
                     'hong kong': 'HKG',
                     'hungary': 'HUN',
                     'iceland': 'ISL',
                     'india': 'IND',
                     'indonesia': 'IDN',
                     'iran': 'IRN',
                     'iraq': 'IRQ',
                     'ireland': 'IRL',
                     'isle of man': 'IMN',
                     'israel': 'ISR',
                     'italy': 'ITA',
                     'jamaica': 'JAM',
                     'japan': 'JPN',
                     'jersey': 'JEY',
                     'jordan': 'JOR',
                     'kazakhstan': 'KAZ',
                     'kenya': 'KEN',
                     'kiribati': 'KIR',
                     'north korea': 'PRK',
                     'south korea': 'KOR',
                     'kosovo': 'KSV',
                     'kuwait': 'KWT',
                     'kyrgyzstan': 'KGZ',
                     'laos': 'LAO',
                     'latvia': 'LVA',
                     'lebanon': 'LBN',
                     'lesotho': 'LSO',
                     'liberia': 'LBR',
                     'libya': 'LBY',
                     'liechtenstein': 'LIE',
                     'lithuania': 'LTU',
                     'luxembourg': 'LUX',
                     'macau': 'MAC',
                     'macedonia': 'MKD',
                     'madagascar': 'MDG',
                     'malawi': 'MWI',
                     'malaysia': 'MYS',
                     'maldives': 'MDV',
                     'mali': 'MLI',
                     'malta': 'MLT',
                     'marshall islands': 'MHL',
                     'mauritania': 'MRT',
                     'mauritius': 'MUS',
                     'mexico': 'MEX',
                     'micronesia': 'FSM',
                     'moldova': 'MDA',
                     'monaco': 'MCO',
                     'mongolia': 'MNG',
                     'montenegro': 'MNE',
                     'morocco': 'MAR',
                     'mozambique': 'MOZ',
                     'namibia': 'NAM',
                     'nepal': 'NPL',
                     'netherlands': 'NLD',
                     'new caledonia': 'NCL',
                     'new zealand': 'NZL',
                     'nicaragua': 'NIC',
                     'nigeria': 'NGA',
                     'niger': 'NER',
                     'niue': 'NIU',
                     'northern mariana islands': 'MNP',
                     'norway': 'NOR',
                     'oman': 'OMN',
                     'pakistan': 'PAK',
                     'palau': 'PLW',
                     'panama': 'PAN',
                     'papua new guinea': 'PNG',
                     'paraguay': 'PRY',
                     'peru': 'PER',
                     'philippines': 'PHL',
                     'poland': 'POL',
                     'portugal': 'PRT',
                     'puerto rico': 'PRI',
                     'qatar': 'QAT',
                     'romania': 'ROU',
                     'russia': 'RUS',
                     'rwanda': 'RWA',
                     'saint kitts and nevis': 'KNA',
                     'saint lucia': 'LCA',
                     'saint martin': 'MAF',
                     'saint pierre and miquelon': 'SPM',
                     'saint vincent and the grenadines': 'VCT',
                     'samoa': 'WSM',
                     'san marino': 'SMR',
                     'sao tome and principe': 'STP',
                     'saudi arabia': 'SAU',
                     'senegal': 'SEN',
                     'serbia': 'SRB',
                     'seychelles': 'SYC',
                     'sierra leone': 'SLE',
                     'singapore': 'SGP',
                     'sint maarten': 'SXM',
                     'slovakia': 'SVK',
                     'slovenia': 'SVN',
                     'solomon islands': 'SLB',
                     'somalia': 'SOM',
                     'south africa': 'ZAF',
                     'south sudan': 'SSD',
                     'spain': 'ESP',
                     'sri lanka': 'LKA',
                     'sudan': 'SDN',
                     'suriname': 'SUR',
                     'swaziland': 'SWZ',
                     'sweden': 'SWE',
                     'switzerland': 'CHE',
                     'syria': 'SYR',
                     'taiwan': 'TWN',
                     'tajikistan': 'TJK',
                     'tanzania': 'TZA',
                     'thailand': 'THA',
                     'timor-leste': 'TLS',
                     'togo': 'TGO',
                     'tonga': 'TON',
                     'trinidad and tobago': 'TTO',
                     'tunisia': 'TUN',
                     'turkey': 'TUR',
                     'turkmenistan': 'TKM',
                     'tuvalu': 'TUV',
                     'uganda': 'UGA',
                     'ukraine': 'UKR',
                     'united arab emirates': 'ARE',
                     'united kingdom': 'GBR',
                     'united states': 'USA',
                     'uruguay': 'URY',
                     'uzbekistan': 'UZB',
                     'vanuatu': 'VUT',
                     'venezuela': 'VEN',
                     'vietnam': 'VNM',
                     'virgin islands': 'VGB',
                     'west bank': 'WBG',
                     'yemen': 'YEM',
                     'zambia': 'ZMB',
                     'zimbabwe': 'ZWE'}

    ## countries
    from collections import Counter

    colorscale = ["#f7fbff", "#ebf3fb", "#deebf7", "#d2e3f3", "#c6dbef", "#b3d2e9", "#9ecae1",
                  "#85bcdb", "#6baed6", "#57a0ce", "#4292c6", "#3082be", "#2171b5", "#1361a9",
                  "#08519c", "#0b4083", "#08306b"
                  ]
    country_vals = geoplot(netflix_new_country)
    tabs = Counter(country_vals).most_common(25)
    labels = [_[0] for _ in tabs][::-1]
    values = [_[1] for _ in tabs][::-1]
    trace1 = go.Bar(y=labels, x=values, orientation="h", name="", marker=dict(color="#a678de"))

    data = [trace1]
    layout = go.Layout(title="Countries with most content", height=700, legend=dict(x=0.1, y=1.1, orientation="h"))
    fig = go.Figure(data, layout=layout)
    fig.show()

    print("******************** Distribution of IMBD Ratings *****************************************")

    import plotly.figure_factory as ff

    x1 = netflix_shows['IMDb_Score'].fillna(0.0).astype(float)
    fig = ff.create_distplot([x1], ['a'], bin_size=0.7, curve_type='normal', colors=["#6ad49b"])
    fig.update_layout(title_text='Distplot with Normal Distribution')
    # fig.update_traces(textinfo='percent+label', textposition='outside')
    fig.show()

    print("******************** Content Across all Categories *****************************************")

    col = "listed_in"
    categories = ", ".join(netflix_new_cat['Genre']).split(", ")
    counter_list = Counter(categories).most_common(50)
    labels = [_[0] for _ in counter_list][::-1]
    values = [_[1] for _ in counter_list][::-1]
    trace1 = go.Bar(y=labels, x=values, orientation="h", name="TV Shows", marker=dict(color="#a678de"))

    data = [trace1]
    layout = go.Layout(title="Content Across all Categories over the years", legend=dict(x=0.1, y=1.1, orientation="h"))
    fig = go.Figure(data, layout=layout)
    # fig.update_traces(texttemplate='%{text:.2s}', textposition='outside')
    fig.show()

    print("******************** Popular Actors in Netflix for the Countries *****************************************")

    from plotly.subplots import make_subplots

    traces = []
    titles = ["United States", "", "India", "", "United Kingdom", "Canada", "", "Spain", "", "Japan"]
    for title in titles:
        if title != "":
            traces.append(country_trace(title))

    fig = make_subplots(rows=2, cols=5, subplot_titles=titles)
    fig.add_trace(traces[0], 1, 1)
    fig.add_trace(traces[1], 1, 3)
    fig.add_trace(traces[2], 1, 5)
    fig.add_trace(traces[3], 2, 1)
    fig.add_trace(traces[4], 2, 3)
    fig.add_trace(traces[5], 2, 5)
    fig.update_layout(height=1200, showlegend=False)
    # fig.update_traces(hoverinfo='label+percent', textinfo='value')
    # fig.update_traces(texttemplate='%{text:.2s}', textposition='outside')

    fig.show()
    print("******************** Co relation Between Attributes *****************************************")
