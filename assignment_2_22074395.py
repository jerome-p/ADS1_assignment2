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

    # Setting index before transposing the dataframe
    temp_df = world_bank_df.set_index('Country Name')

    return world_bank_df, temp_df.T


def generate_bar_plot(data, year, indicator, image_name, title):
    """
    This function generates a bar plot of the Population growth (annual %)
    for the countries specified in the country_list variable. For a given year

    Returns None.
    """

    # Extracting only the selected indicator's data
    subset_data = data[
        data['Indicator Name'] == indicator]

    plt.figure()

    # Passing X-axis as country names and Y-axis is all rows,
    # in the specified year. Here year is a column name
    plt.bar(subset_data['Country Name'],
            subset_data.loc[:, str(year)])

    # labelling
    plt.xlabel("Country")
    plt.ylabel("%")
    plt.xticks(rotation=20)
    plt.title(title)

    # Saving the generatad plot
    plt.savefig('figures/barplot'+image_name+'.png',
                bbox_inches='tight',
                dpi=200)

    plt.show()

    return


def generate_line_plot(data, countries, indicator, xlabel, ylabel, title):
    """
    This function generates line plots for the given dataframe for a particular
    indicator in the world bank dataset. It also requires a list of country
    names.
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
    # Specifying figure size, as plot is big
    plt.figure(figsize=(15, 8))

    # Iterating over the list of countries
    for country in countries:
        temp_df = data[country].T
        # Subsetting the transposed df. Which now has years as columns
        subset_df = temp_df[temp_df['Indicator Name'] == indicator]
        # Transposing the df again to makes years the index.
        subset_df = subset_df.T

        # Plotting using the subset df.
        # Plotting using for loop to include line plots of every country
        # in the same figure.
        plt.plot(subset_df[1:], label=country)

        # Labelling
        plt.xticks(rotation=90)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend()

    # Saving the figure.
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

        # Transposing DF after subsetting based on Country name is columns.
        temp_df = data[country].T

        # Subsetting the DF based on Indicator name now.
        # Currently years are columns. Meaning we have onlt one row of data.
        subset_df = temp_df[temp_df['Indicator Name'] == indicator]

        # transposing the DF to make country name column and years rows.
        subset_df = subset_df.T

        print("\nStatistics for: " + str(country) + ": \n")

        # Generating all basics statistics of the given data using
        # df.describe() and median() methods.
        print(subset_df[1:].astype(float).describe())
        print("Median: " + str(subset_df[1:].astype(float).median()[country]))

        # Calculating Skewness and Kurtosis of the given indicator
        # The values in the DF are int, converting them to float for
        # Skew and kurtosis methods to work
        skewness = st.skew(subset_df[1:].astype(float))
        kurtosis = st.kurtosis(subset_df[1:].astype(float))

        # Specifying the country name in the skewness DF
        # to get only value from the DF
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

    # Subsetting the dataframe based on the country name passed
    data = data[data['Country Name'] == country]

    # Selecting only the indicators needed from the dataset.
    data = data[
        (data['Indicator Name'] == indicator1) |
        (data['Indicator Name'] == indicator2) |
        (data['Indicator Name'] == indicator3) |
        (data['Indicator Name'] == indicator4)
    ]

    # Transposing the dataframe so that the columns are now the indicator
    # names and each row is the year.
    data = data.T
    data.columns = data.loc['Indicator Name']

    # Dropping first two rows of the dataframe, as the information is
    # repeated and column names are already the indicator names.
    data = data.drop(['Country Name', 'Indicator Name'], axis=0)

    # Using the dataframe corr() method to get correlation matrix
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
    # Creating an  empty Dataframe to store the specified indicator values for
    # each country.
    result = pd.DataFrame()

    # Iterating through the list of countries
    for country in countries:
        # Transposing the dataframe to be able to extract indicator values
        temp_df = data[country].T
        subset_df = temp_df[temp_df['Indicator Name'] == indicator]

        # Transposing the subset to make the years rows from columns.
        subset_df = subset_df.T

        # Creating a new column in the result df and add the current country's
        # indicator value, i.e the subset_df
        result[country] = subset_df

    # Since the dataframe contains values of only one indicator value and
    # only country name is needed, dropping the first row with indicator name.
    result = result.drop(['Indicator Name'], axis=0)

    # Selecting a particular year for the pie chart.
    result = result.loc[str(year)]
    plt.figure()

    # Plotting pie chart using the df.plot method
    result.plot(kind='pie', subplots=True, autopct="%1.f%%", ylabel="")

    # using arguments passed while calling funtion to create title.
    plt.title(title+" in "+str(year))

    # Saving the pie chart.
    plt.savefig("figures/pie_chart_"+title+".png",
                bbox_inches='tight',
                dpi=200)
    plt.show()

    return


def create_heatmap(cm_df, image_name, country, labels=False):
    """
    Creates a heatmape using the seasborn library. This takes in a correlation 
    matrix generated by the generate_corr_between() method.

    Parameters
    ----------
    cm_df : DataFrame
    image_name : String
    country : String
    labels: Default=False, takes in List.

    Returns None
    """

    plt.figure()

    # If no labels are given, generates heatmap with default label setting.
    if labels:
        sns.heatmap(cm_df, annot=True, center=True,
                    xticklabels=labels, yticklabels=labels)
    else:
        sns.heatmap(cm_df, annot=True, center=True)

    # Adding title to the heatmap using argument passed during function call.
    plt.title("Correlation Heatmap of Indicators for "+country)

    # Saving the heatmap
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

    # Create a list of Countries interested in evaluating.
    countries = ['China', 'India', 'Japan',
                 'United Kingdom', 'United States', 'Germany']

    # Loading the data from csv file. Function return two dataframes,
    # one of which is the transpose of the other.
    wb_data_years, wb_data_country = load_data(
        'API_19_DS2_en_csv_v2_5998250.csv',
        countries)

    # Displaying column names of wb_data_years
    print("Columns of wb_data_years: \n")
    print(wb_data_years.columns)

    # Displaying column names of wb_data_country
    print("Columns of wb_data_country: \n")
    print(wb_data_country.columns)

    # Checking how many Indicators are present in the dataset.
    print("\nTotal number of Indicators in the dataset: " +
          str(len(wb_data_years['Indicator Name'].unique())))

    # Generating all basic stastics and two additional methods.
    generate_statistics(wb_data_country, countries,
                        'CO2 emissions (kt)')

    # Generating a line plot of CO2 Emissions for the selected countries
    generate_line_plot(wb_data_country, countries,
                       'CO2 emissions (kt)',
                       "Years",
                       "metric tons",
                       "CO2 emissions-Kilotons")

    # Generating a line plot of % of electricity produced from oil
    # for the selected countries
    generate_line_plot(wb_data_country, countries,
                       'Electricity production from oil sources (% of total)',
                       "Years",
                       "% of land",
                       "Electricity production from oil sources (% of total)")

    # Generating line plot of % of Renewable energy output (% of total energy)
    generate_line_plot(
        wb_data_country, countries,
        'Renewable electricity output (% of total electricity output)',
        "Years",
        "% of total",
        "Renewable electricity output(% of total)"
    )

    # Generating correlation between the selected indicators for a particular
    # country, in this case India.
    corr = generate_corr_between(
        wb_data_years,
        'CO2 emissions (kt)',
        'Electricity production from oil sources (% of total)',
        'Renewable electricity output (% of total electricity output)',
        'Renewable energy consumption (% of total final energy consumption)',
        'India'
    )

    # Creating a heatmap using the correlation matrix generated above.
    create_heatmap(corr,
                   '1',
                   country='India',
                   labels=['CO2', 'Renewable energy consumption',
                           'Renewable energy output',
                           'electricity from oil']
                   )

    # Creating a pie chart of an indicator for a selected country and year.
    generate_pie_chart(
        wb_data_country,
        2020,
        countries,
        'CO2 emissions (metric tons per capita)',
        'CO2 emissions per capita'
    )

    # Generating line plot of greenhouse gas emissions
    generate_line_plot(wb_data_country, countries,
                       'Total greenhouse gas emissions (kt of CO2 equivalent)',
                       "Years",
                       "kt of CO2",
                       "Greenhouse gas emissions")

    # Generating barplot of popultion growth of countries in the year 2020
    generate_bar_plot(
        wb_data_years,
        2020,
        'Population growth (annual %)',
        image_name='population_growth_2020',
        title="Population growth (%) in 2020")

    # Generating barplot of popultion growth of countries in the year 1996
    generate_bar_plot(
        wb_data_years,
        1996,
        'Population growth (annual %)',
        image_name='population_growth_1996',
        title="Population growth (%) in 1996")

    # Generating line plot of total population of each country.
    generate_line_plot(wb_data_country,
                       countries,
                       'Population, total',
                       "Years",
                       "Billions",
                       'Population over the years in billions')

    # Calculating correlation between indicators for India.
    corr2 = generate_corr_between(
        wb_data_years,
        'Total greenhouse gas emissions (kt of CO2 equivalent)',
        'Urban population (% of total population)',
        'Population, total',
        'Cereal yield (kg per hectare)',
        'India'
    )

    # Creating heatmap using the heatmap genarated from above.
    create_heatmap(corr2, '2', country='India',
                   labels=['Urban Population %',
                           'Total popultation',
                           'Greenhouse Gas',
                           'Cereal Yield']
                   )

    return


if __name__ == '__main__':
    # Calling the main function.
    main()
