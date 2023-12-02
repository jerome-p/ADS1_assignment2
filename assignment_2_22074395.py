# Importing libraries
import pandas as pd
import matplotlib.pyplot as plt
import stats as st
import seaborn as sns


def load_data(dataset, country_list):
    """
    This function takes in a csv file and a list of countries that are of 
    interest. Returns two dataframes one with the years as columns and the
    other country names as columns.

    Parameters
    ----------
    dataset : .csv file
    country_list : List
    """
    # skipping first 4 rows, as they contain non essential data.
    world_bank_df = pd.read_csv(dataset, skiprows=4)

    # Removing non essential data.
    world_bank_df.drop(['Country Code', 'Indicator Code', 'Unnamed: 67'],
                       axis=1, inplace=True)

    # subsetting the dataframe to get data for countries we are interested in.
    world_bank_df = world_bank_df[
        world_bank_df['Country Name'].isin(country_list)]

    # Settnig index before transposing the dataframe
    temp_df = world_bank_df.set_index('Country Name')

    return world_bank_df, temp_df.T


def generate_bar_plot(data, year):
    """
    This function generates a bar plot of the Population growth (annual %)
    for the countries specified in the country_list variable. For a given year

    Returns None.
    """

    pop_growth_annual = data[
        data['Indicator Name'] == 'Population growth (annual %)']

    plt.figure()

    plt.bar(pop_growth_annual['Country Name'],
            pop_growth_annual.loc[:, str(year)])

    return


def generate_line_plot(data, countries, indicator, xlabel, ylabel, title):
    """
    This function generates    

    Parameters
    ----------
    data : TYPE
    countries : TYPE
    indicator : TYPE
    xlabel : TYPE
    ylabel : TYPE
    title : TYPE

    Returns None.

    """

    plt.figure(figsize=(15, 8))
    for country in countries:
        temp_df = data[country].T
        subset_df = temp_df[temp_df['Indicator Name'] == indicator]
        subset_df = subset_df.T
        plt.plot(subset_df[1:], label=country)
        plt.xticks(rotation=90)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend()
        
    plt.savefig('figures/line_graph_'+indicator+'.png',
                 bbox_inches='tight',
                 dpi=200)
    plt.show()

    return


def generate_statistics(data, countries, indicator):
    """
    This function prints out statiscal values of the indicator passed to it.
    It prints out count,mean,standard deviation,min,max and 25,50,75 quartiles.
    It also prints out the skewness and kurotsis of the indicator.

    Parameters
    ----------
    data : Pandas DataFrame
    countries : List (list of countries to be evaluated)
    indicator : String 

    Returns None
    """

    for country in countries:
        temp_df = data[country].T
        subset_df = temp_df[temp_df['Indicator Name'] == indicator]
        subset_df = subset_df.T
        print("\nStatistics for: " + str(country) + ": \n")

        print(subset_df[1:].astype(float).describe())
        print("Median: " + str(subset_df[1:].astype(float).median()[country]))

        # Calculating Skewness and Kurtosis of the given indicator
        skewness = st.skew(subset_df[1:].astype(float))
        kurtosis = st.kurtosis(subset_df[1:].astype(float))
        print("Skewness: "+str(skewness[country]))
        print("Kurtosis: "+str(kurtosis[country]))

    return


def generate_corr_between(data, indicator1,
                          indicator2, indicator3, indicator4, country):
    """
    This function calculates the correlation between four indicators.
    It calculates this value for a given country. 

    Parameters
    ----------
    data : Pandas DataFrame
    indicator1 : String
    indicator2 : String
    indicator3 : String
    indicator4 : String
    country : String

    Returns: Pandas DataFrame (correlation matrix)

    """
    data = data[data['Country Name'] == country]
    data = data[
        (data['Indicator Name'] == indicator1) |
        (data['Indicator Name'] == indicator2) |
        (data['Indicator Name'] == indicator3) |
        (data['Indicator Name'] == indicator4)
    ]

    data = data.T
    data.columns = data.loc['Indicator Name']
    data = data.drop(['Country Name', 'Indicator Name'], axis=0)
    corr_result = data.corr(numeric_only=False)

    return corr_result


def generate_pie_chart(data, year, countries, indicator, title):
    """
    Creates a pie chart for the specified year. Uses the dataframe plot 
    function create a pie chart for a list of countries 
    and a specific indicator.

    Parameters
    ----------
    data : Pandas DataFrame
    year : String
    countries : List
    indicator : string

    Returns None

    """
    result = pd.DataFrame()
    for country in countries:
        temp_df = data[country].T
        subset_df = temp_df[temp_df['Indicator Name'] == indicator]
        subset_df = subset_df.T
        result[country] = subset_df[country]

    result = result.drop(['Indicator Name'], axis=0)
    result = result.loc[str(year)]
    plt.figure()
    result.plot(kind='pie', subplots=True, autopct="%1.f%%")
    plt.title(title)
    plt.savefig("figures/pie_chart_"+title+".png",
                bbox_inches='tight',
                dpi=200)
    plt.show()

    return


def create_heatmap(cm_df,image_name, labels=False):
    """
    Creates a heatmape using the seasborn library.

    Parameters
    ----------
    cm_df : DataFrame

    Returns None
    """

    plt.figure()
    if labels:
        sns.heatmap(cm_df, annot=True, center=True,
                xticklabels=labels,yticklabels=labels)
    else:
        sns.heatmap(cm_df, annot=True, center=True)
        
    plt.savefig('figures/heatmap'+image_name+'.png',
                bbox_inches='tight',
                dpi=200)
    plt.show()

    return


def main():
    """
    Main function, it is used to run all the other funcions defined.
    returns None
    """
    countries = ['China', 'India', 'Japan',
                 'United Kingdom', 'United States', 'Germany']

    wb_data_years, wb_data_country = load_data(
        'API_19_DS2_en_csv_v2_5998250.csv',
        countries)

    print("Columns of wb_data_years: \n")
    print(wb_data_years.columns)

    print("Columns of wb_data_country: \n")
    print(wb_data_country.columns)

    print("\nTotal number of Indicators in the dataset: " +
          str(len(wb_data_years['Indicator Name'].unique())))

    generate_statistics(wb_data_country, countries,
                        'CO2 emissions (kt)')

    generate_line_plot(wb_data_country, countries,
                       'CO2 emissions (kt)',
                       "Years",
                       "metric tons",
                       "CO2 emissions-Kilotons")

    generate_line_plot(wb_data_country, countries,
                       'Electricity production from oil sources (% of total)',
                       "Years",
                       "% of land",
                       "Electricity production from oil sources (% of total)")

    #generate_line_plot(wb_data_country, countries,
    #            'Renewable electricity output (% of total electricity output)',
    #            "Years",
    #            "% of total",
    #            "Renewable electricity output(% of total)")

    corr = generate_corr_between(wb_data_years,
        'CO2 emissions (kt)',
        'Electricity production from oil sources (% of total)',
        'Renewable electricity output (% of total electricity output)',
        'Renewable energy consumption (% of total final energy consumption)',
        'India')

    create_heatmap(corr,'1', labels=['CO2','Renewable energy consumption',
                                     'Renewable energy output',
                                     'electricity from oil'])

    generate_pie_chart(wb_data_country,
                       2020,
                       countries,
                       'CO2 emissions (metric tons per capita)',
                       'CO2 emissions per capita')

    #generate_line_plot(wb_data_country, countries,
    #                   'Total greenhouse gas emissions (kt of CO2 equivalent)',
    #                   "Years",
    #                   "kt of CO2",
    #                   "Greenhouse gas emissions")

    #generate_line_plot(wb_data_country, countries,
    #                   'Arable land (% of land area)',
    #                   "Years",
    #                   "% of land",
    #                   "Arable land %")

    #generate_bar_plot(wb_data_years, 2020)
    #generate_bar_plot(wb_data_years, 2019)

    #generate_line_plot(wb_data_country,
    #                   countries,
    #                   'Population, total',
    #                   "Years",
    #                   "Billions",
    #                   'Population over the years in billions')

    #generate_line_plot(wb_data_country, countries,
    #                   'Electric power consumption (kWh per capita)',
    #                   "Years",
    #                   "kWh per capita",
    #                   "Electric power consumption in kWh per capita")

    corr2 = generate_corr_between(wb_data_years,
                    'Total greenhouse gas emissions (kt of CO2 equivalent)',
                    'Electric power consumption (kWh per capita)',
                    'Population, total',
                    'Cereal yield (kg per hectare)',
                    'India') 
    # Electric power consumption (kWh per capita)
    
    create_heatmap(corr2,'2', labels=['population','greenhouse gas',
                                      'electric consumption', 'cereal yield'])

    return


if __name__ == '__main__':
    main()
