import pandas as pd
import sciris as sc
import covasim as cv


def validate_country_households_range():
    """
    Validate household size is withing the expected range.
    """
    print('Loading data...')
    ch = cv.data.loaders.get_household_sizes()

    assert 2 <= ch['usa'] <= 5
    assert 1 <= ch['korea'] <= 3
    assert 1 <= ch['south korea'] <= 3

    for c in ch:
        assert 0 < ch[c] < 10

    print('Household size values are in the expected range.')

    return ch


def plot_validate_country_households():
    """
    Plot to help spot if some countries have unexpected values.
    """
    # Convert to dataframe for plotting.
    ch = cv.data.loaders.get_household_sizes()
    df = pd.DataFrame.from_dict(ch, orient='index', columns=['size']).reset_index()
    df['size'] = df['size'].astype(float)
    df.columns = ['country', 'size']
    df = df.sort_values(by=['size'], ascending=False)

    # Plot
    ax = df.plot(kind='barh', x=df.columns[0], y=df.columns[1], figsize=(20, 50))
    fig = ax.get_figure()

    return fig


# %% Run as a script
if __name__ == '__main__':
    sc.tic()

    ch = validate_country_households_range()
    fig = plot_validate_country_households()

    sc.toc()

    print('Done.')
