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

    print(df)
    quit()

    return df


def main():

    see_all()

    airfoil_data = csv_func('Airfoil_DataSheets')
    airfoil_data = get_abs_values(airfoil_data)

    plot_styling()
    # plot_titles(airfoil_data, 'Drag vs. Lift Coefficient at 20 m/s', 'drag_coeff', 'lift_coeff')
    # plot_titles(airfoil_data, 'Drag vs. Lift at 20 m/s with wing length of 1m', 'drag_1m', 'lift_1m')

    print(airfoil_data)


if __name__ == '__main__':

    main()
