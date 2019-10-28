import os
import pandas as pd
import numpy as np
from scipy import interpolate
import decimal
import matplotlib.pyplot as plt
from scipy.spatial import distance
from datetime import datetime


def directory_get():

    og_path = os.path.dirname(os.path.abspath(__file__))

    og_path_1 = og_path.split(f'\\')

    length = len(og_path_1)
    og_path_1 = og_path_1[length-1]

    og_path = og_path.split(f'{og_path_1}')[0]
    og_path = og_path + f'ejectors\\'

    return og_path


def new_path():

    directory = directory_get()

    date_today = datetime.today().strftime('%Y-%m-%d')

    newdirectory = f'{directory}Delivery\\{date_today}'

    if not os.path.exists(newdirectory):

        os.makedirs(newdirectory)

    return newdirectory


def plotting(df):

    df.plot(kind='scatter', x='x', y='y')
    plt.show()


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

    # plt.tick_params(top='False', bottom='False', left='False', right='False', labelleft='False', labelbottom='True')

    for spine in plt.gca().spines.values():
        spine.set_visible(False)


def plot_titles(data):

    plt.title('Temperature vs Time')

    plt.ylabel('Temperature [ºC]')
    plt.xlabel('Time [mins]')

    max_y = data.loc[:,'T_room'].max() + 0.15*(data.loc[:,'T_room'].max())
    max_x = data.loc[:,'Minutes'].max() + 0.15*(data.loc[:,'Minutes'].max())

    plt.ylim((-5, max_y))
    plt.xlim(0,max_x)

    plt.plot(data.loc[:, 'Minutes'], data.loc[:, 'T_room'],
             ',', markersize=3,
             label=f'Room Temperature',
             zorder=1)

    plt.plot(data.loc[:, 'Minutes'], data.loc[:, 'T_amb'],
             ',', markersize=1, alpha= 0.3,
             label=f'Amb. Temperature',
             zorder=3)

    plt.plot(data.loc[:, 'Minutes'], data.loc[:, 'Power']/6000, ',',
             markersize=10, alpha= 0.5, label= 'Power/6000')

    plt.legend(title='Legend:')
    plt.show()


def warehouse():

    base_directory = directory_get()

    directory = f'{base_directory}Default\\Warehouse\\'

    if not os.path.exists(directory):

        os.makedirs(directory)

    return directory


def interpolate_spline(points_df, x):

    x_points_up = points_df.loc[:, 'x_up'].values.tolist()
    y_points_up = points_df.loc[:, 'y_up'].values.tolist()

    x_points_down = points_df.loc[:, 'x_down'].values.tolist()
    x_points_down = [x_points_down for x_points_down in x_points_down if str(x_points_down) != 'nan']
    y_points_down = points_df.loc[:, 'y_down'].values.tolist()
    y_points_down = [y_points_down for y_points_down in y_points_down if str(y_points_down) != 'nan']

    tck_up = interpolate.splrep(x_points_up, y_points_up)
    tck_down = interpolate.splrep(x_points_down, y_points_down)

    return interpolate.splev(x, tck_up), interpolate.splev(x, tck_down)


def new_tin_path(date, name):

    base_directory = directory_get()

    newdirectory = f'{base_directory}\\WINGS\\Mesh_ICEM\\{date}\\{name}\\'

    if not os.path.exists(newdirectory):

        os.makedirs(newdirectory)

    return newdirectory


def clean_array(raw_array):

    x = raw_array[2]
    y = raw_array[4].split('\n')[0]

    return x,y


def create_replay(path, factor_mult, name):

    csv = edges(path)

    base_directory = directory_get()

    ICEM_read = open(f'{base_directory}WINGS\\Mesh_ICEM\\Default\\Replay.txt', 'r')
    ICEM_geom = ICEM_read.read()
    ICEM_read.close()

    k= 0

    for i in range(22):

        old_32_33_up = str(f'ic_hex_split_edge 32 33 {i} X_32_33_up Y_32_33_up 0')

        num = 22-i

        y_up = csv.iloc[num].loc['y_up'] * factor_mult
        x = csv.iloc[num].loc['x'] * factor_mult

        y_up = str(decimal.Decimal(y_up.item(0)))
        x = str(decimal.Decimal(x.item(0)))

        new_32_33_up = str(f'ic_hex_split_edge 32 33 0 {x} {y_up} 0')

        ICEM_geom = ICEM_geom.replace(old_32_33_up, new_32_33_up)

        k += 1


    k_up = 0
    k_down = 0

    for i in range(22):

        old_32_33_down = str(f'ic_hex_split_edge 32 33 {i} X_32_33_down Y_32_33_down 0')

        y_down = csv.iloc[i].loc['y_down'] * factor_mult
        x = csv.iloc[i].loc['x'] * factor_mult

        y_down = str(decimal.Decimal(y_down.item(0)))
        x = str(decimal.Decimal(x.item(0)))

        new_32_33_down = str(f'ic_hex_split_edge 32 33 0 {x} {y_down} 0')

        ICEM_geom = ICEM_geom.replace(old_32_33_down, new_32_33_down)

        k +=1

    for i in range(44, 230):

        old_33_35 = str(f'ic_hex_split_edge 33 35 {k_up} X_33_35 Y_33_35 0')
        old_32_34 = str(f'ic_hex_split_edge 32 34 {k_down} X_32_34 Y_32_34 0')

        y_up = csv.iloc[i].loc['y_up'] * factor_mult
        y_down = csv.iloc[i].loc['y_down'] * factor_mult
        x = csv.iloc[i].loc['x'] * factor_mult

        y_up = str(decimal.Decimal(y_up.item(0)))
        y_down = str(decimal.Decimal(y_down.item(0)))
        x = str(decimal.Decimal(x.item(0)))

        new_33_35 = str(f'ic_hex_split_edge 33 35 {k_up} {x} {y_up} 0')
        new_32_34 = str(f'ic_hex_split_edge 32 34 {k_down} {x} {y_down} 0')

        k_up += 1
        k_down += 1

        ICEM_geom = ICEM_geom.replace(old_33_35, new_33_35)
        ICEM_geom = ICEM_geom.replace(old_32_34, new_32_34)



    rpl_file = open(f'{base_directory}WINGS\\Mesh_ICEM\\{name}\\{factor_mult}\\{name}.rpl', 'w')
    rpl_file.write(ICEM_geom)
    rpl_file.close()


def coordinates_points(path):

    points_df = updown_points(path)
    interp_x = {}

    interp_y_up = {}
    interp_y_down = {}
    x_coord = {}
    i = 0

    for x in np.arange(0, 1, 1/38):

        interp_y_up[i], interp_y_down[i] = interpolate_spline(points_df, x)
        interp_x[i] = x
        x_coord[i] = x

        i = i+1

    coordinates_df = pd.DataFrame({'y_up': interp_y_up,
                                   'y_down': interp_y_down,
                                   'x': x_coord})

    return coordinates_df


def create_project(coordinates_multip, factor_mult, name):

    csv = coordinates_multip

    base_directory = directory_get()

    ICEM_read = open(f'{base_directory}WINGS\\Mesh_ICEM\\Default\\Project File.txt', 'r')
    ICEM_geom = ICEM_read.read()
    ICEM_read.close()

    old = 'NAMEPROJECT'

    ICEM_geom = ICEM_geom.replace(old, name)


    prj_file = open(f'{base_directory}WINGS\\Mesh_ICEM\\{name}\\{factor_mult}\\{name}.prj', 'w')
    prj_file.write(ICEM_geom)
    prj_file.close()


def create_tin(coordinates_multip, factor_mult, name):

    csv = coordinates_multip

    base_directory = directory_get()

    ICEM_read = open(f'{base_directory}WINGS\\Mesh_ICEM\\Default\\Geometry.txt', 'r')
    ICEM_geom = ICEM_read.read()
    ICEM_read.close()

    i = 0

    for i in range(10):

        old_x = str(f'X{i}')
        old_y = str(f'Y{i}')

        new_x = str(decimal.Decimal(csv.iloc[i].loc['x']))
        new_y = str(decimal.Decimal(csv.iloc[i].loc['y']))

        ICEM_geom = ICEM_geom.replace(old_x, new_x)
        ICEM_geom = ICEM_geom.replace(old_y, new_y)


    for i in range(10, len(csv)):

        old_x = str(f'x{i}')
        old_y = str(f'y{i}')

        new_x = str(decimal.Decimal(csv.iloc[i].loc['x']))
        new_y = str(decimal.Decimal(csv.iloc[i].loc['y']))

        ICEM_geom = ICEM_geom.replace(old_x, new_x)
        ICEM_geom = ICEM_geom.replace(old_y, new_y)

    old_x = str(f'last_x')
    old_y = str(f'last_y')

    last_x, last_y = create_curve(coordinates_multip)

    last_x = str(last_x)
    last_y = str(last_y)

    ICEM_geom = ICEM_geom.replace(old_x, last_x)
    ICEM_geom = ICEM_geom.replace(old_y, last_y)

    tin_file = open(f'{base_directory}WINGS\\Mesh_ICEM\\{name}\\{factor_mult}\\{name}.tin', 'w')
    tin_file.write(ICEM_geom)
    tin_file.close()


def wing_multip(wing_path, factor_mult):

    csv_read_1 = coordinates_points(wing_path)

    a = np.empty((len(csv_read_1), 3), dtype=int)
    a[:] = factor_mult

    csv_read_1 = csv_read_1 * a

    return csv_read_1


def get_perimeter(df, filename):

    length_up = 0
    length_down = 0

    coord_up = {}
    coord_down = {}
    coord_up[0] = [df.iloc[0].loc['y_up'] ,df.iloc[0].loc['x']]
    coord_down[0] = [df.iloc[0].loc['y_down'], df.iloc[0].loc['x']]

    for i in range(1, len(df)):

        coord_up[i] = [df.iloc[i].loc['y_up'] ,df.iloc[i].loc['x']]
        coord_down[i] = [df.iloc[i].loc['y_down'], df.iloc[i].loc['x']]

        length_up = length_up + abs(distance.euclidean(coord_up[i], coord_up[i-1]))
        length_down = length_down + abs(distance.euclidean(coord_down[i], coord_down[i - 1]))

    perimeter = length_up + length_down

    print(filename.split('.txt')[0])
    print(perimeter)



def csv_func(path):

    csv_reader = open(path, 'rb')
    csv_read = pd.read_csv(csv_reader, encoding='latin1')
    csv_reader.close()

    # xyz = 'x y z'
    #
    # # if csv_read.columns.values[0] != any(['x', 'x y z']):
    # if any(word in xyz for word in csv_read.columns.values[0]):
    #
    #     print('Invalid geometry: You should create a geometry file with "x y z" as the first row.')
    #     quit()

    column_names = csv_read.columns.values[0]

    if len(csv_read.iloc[0].loc[column_names].split(' ')) == 3:

        csv_reader = open(path, 'rb')
        csv_read = pd.read_csv(csv_reader, encoding='latin1', sep = ' ')
        csv_reader.close()

    return csv_read


def create_tin(coordinates, date, name):

    print(coordinates)
    print('\n')
    print(date)
    print(name)

    quit()

    base_directory = directory_get()

    ICEM_read = open(f'{base_directory}WINGS\\Mesh_ICEM\\Default\\Geometry.txt', 'r')
    ICEM_geom = ICEM_read.read()
    ICEM_read.close()

    i = 0

    for i in range(10):

        old_x = str(f'X{i}')
        old_y = str(f'Y{i}')

        new_x = str(decimal.Decimal(csv.iloc[i].loc['x']))
        new_y = str(decimal.Decimal(csv.iloc[i].loc['y']))

        ICEM_geom = ICEM_geom.replace(old_x, new_x)
        ICEM_geom = ICEM_geom.replace(old_y, new_y)


    for i in range(10, len(csv)):

        old_x = str(f'x{i}')
        old_y = str(f'y{i}')

        new_x = str(decimal.Decimal(csv.iloc[i].loc['x']))
        new_y = str(decimal.Decimal(csv.iloc[i].loc['y']))

        ICEM_geom = ICEM_geom.replace(old_x, new_x)
        ICEM_geom = ICEM_geom.replace(old_y, new_y)

    old_x = str(f'last_x')
    old_y = str(f'last_y')

    last_x, last_y = create_curve(coordinates_multip)

    last_x = str(last_x)
    last_y = str(last_y)

    ICEM_geom = ICEM_geom.replace(old_x, last_x)
    ICEM_geom = ICEM_geom.replace(old_y, last_y)

    tin_file = open(f'{base_directory}WINGS\\Mesh_ICEM\\{name}\\{factor_mult}\\{name}.tin', 'w')
    tin_file.write(ICEM_geom)
    tin_file.close()


def directories_loop(warehouse_location, new_directory):

    directory = os.fsencode(warehouse_location)

    name = {}
    coordinates = {}
    newest_directory = {}
    txt_name = {}

    cool_df = {}

    i = 0

    for file in os.listdir(directory):

         filename = os.fsdecode(file)

         if filename.endswith(".txt"):

             name[i] = warehouse_location + filename

             coordinates[i] = csv_func(name[i])

             plotting(coordinates[i])

             quit()

             txt_name[i] = f"{filename.split('.txt')[0]}"

             newest_directory[i] = new_directory + txt_name[i] + '.csv'

             directory_len = len(new_directory.split('\\'))
             date = new_directory.split('\\')[directory_len-1]

             new_tin_path(date, txt_name[i])

             create_tin(coordinates[i], date, txt_name[i])

             quit()

             create_replay(name[i], date, txt_name[i])

             create_tin(cool_df[i], factor_mult, txt_name[i])

             create_project(cool_df[i], factor_mult, txt_name[i])

             i += 1

         else:
             continue


def open_geometry():

    pass



def main():

    warehouse_location = warehouse()
    new_directory = new_path()
    plot_styling()

    directories_loop(warehouse_location, new_directory)

    print()
    input('Press any key to exit, my work is done!')

if __name__ == "__main__":

    main()
