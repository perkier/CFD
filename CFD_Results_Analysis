import pandas as pd
import matplotlib.pyplot as plt
import sys

def find_path():
    # Return DATA Folder Path

    data_path = sys.path[0].split('CODE')[0]
    data_path = f'{data_path}\\WINGS\\Simulations\\'

    return data_path


def plot_styling():

    plt.style.use('dark_background')

    plt.gca().yaxis.grid(True, color='gray')

    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = 'Ubuntu'
    plt.rcParams['font.monospace'] = 'Ubuntu Mono'
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.labelsize'] = 10
    plt.rcParams['axes.labelweight'] = 'bold'
    plt.rcParams['xtick.labelsize'] = 8
    plt.rcParams['ytick.labelsize'] = 8
    plt.rcParams['legend.fontsize'] = 10
    plt.rcParams['figure.titlesize'] = 12

    for spine in plt.gca().spines.values():
        spine.set_visible(False)



def plot_titles(data, title, x_label, y_label):


    # plt.tick_params(top='False', bottom='False', left='False', right='False', labelleft='False', labelbottom='True')

    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    max_y = data.loc[:,y_label].max() + 0.15*(data.loc[:,y_label].max())
    max_x = data.loc[:,x_label].max() + 0.15*(data.loc[:,x_label].max())

    plt.ylim((0, max_y))
    plt.xlim(0,max_x)

    x = data.loc[:, x_label]
    y = data.loc[:, y_label]

    plt.plot([x], [y], marker='o', markersize=7)

    plt.show()


def csv_func(name):

    name = f'{name}'

    csv_reader = open(f'{find_path()}{name}.csv', 'rb')
    csv_read = pd.read_csv(csv_reader, encoding='latin1')
    csv_reader.close()

    # csv_read = csv_read.sample(frac=1).reset_index(drop=True)

    return csv_read


def see_all():
    # Alongate the view on DataFrames

    pd.set_option('display.max_rows', 1000)
    pd.set_option('display.max_columns', 1000)
    pd.set_option('display.width', 1000)


def get_abs_values(df):
    # Transform the drag and lift coefficients into absolute values

    rho = 2.4078596
    u = 20
    m_kg = 40

    first_term = 2 * m_kg * 9.81
    sec_term_lift = rho * u * u * df.loc[:, 'perimeter'] * df.loc[:, 'lift_coeff']

    df['length_40kg'] = first_term / sec_term_lift

    # df['drag_1m'] = df.loc[:, 'drag_coeff'] * df.loc[:, 'perimeter'] * middle_term
    # df['lift_1m'] = df.loc[:, 'lift_coeff'] * df.loc[:, 'perimeter'] * middle_term

    return df


def get_lift_drag(df):
    # Get coefficient between lift and drag values

    df['cl/cd'] = df.loc[:, 'lift_coeff'] / df.loc[:, 'drag_coeff']

    return df




def choice_based_lift(df, num_choice):

    df = df.loc[df['angle'] == 0].reset_index(drop=True)
    df = df.sort_values(by='lift_coeff', ascending=False).reset_index(drop=True)

    choice = df.head(num_choice)

    geoms = choice.loc[:,'wing'].unique()

    return geoms


def choice_based_drag(df, num_choice):

    df = df.loc[df['angle'] == 0].reset_index(drop=True)
    df = df.sort_values(by='drag_coeff', ascending=True).reset_index(drop=True)

    choice = df.head(num_choice)

    geoms = choice.loc[:,'wing'].unique()

    return geoms


def choice_based_coeff(df, num_choice):

    df = df.loc[df['angle'] == 0].reset_index(drop=True)
    df = df.sort_values(by='cl/cd', ascending=False).reset_index(drop=True)

    choice = df.head(num_choice)

    geoms = choice.loc[:,'wing'].unique()

    return geoms


def get_geoms_final(geoms_lift, geoms_drag, geoms_coeff):

    geoms_lift = geoms_lift.tolist()
    geoms_coeff = geoms_coeff.tolist()
    geoms_drag = geoms_drag.tolist()

    geoms_lift.extend(geoms_coeff)
    geoms_lift.extend(geoms_drag)

    geoms_final = list(set(geoms_lift))

    return geoms_final


def main():

    see_all()

    airfoil_data = csv_func('Airfoil_DataSheets')
    airfoil_data = get_abs_values(airfoil_data)
    airfoil_data = get_lift_drag(airfoil_data)

    plot_styling()
    # plot_titles(airfoil_data, 'Drag vs. Lift Coefficient at 20 m/s', 'drag_coeff', 'lift_coeff')
    # plot_titles(airfoil_data, 'Drag vs. Lift at 20 m/s with wing length of 1m', 'drag_1m', 'lift_1m')

    print(airfoil_data.sort_values(by='lift_coeff', ascending=False))

    geoms_lift = choice_based_lift(airfoil_data, 5)
    geoms_drag = choice_based_drag(airfoil_data, 3)
    geoms_coeff = choice_based_coeff(airfoil_data, 5)

    geoms_final = get_geoms_final(geoms_lift, geoms_drag, geoms_coeff)
    print(geoms_final)


if __name__ == '__main__':

    main()
