import os
import pandas as pd
import numpy as np
from scipy import interpolate
import decimal
import matplotlib as plt


def directory_get():

    og_path = os.path.dirname(os.path.abspath(__file__))

    og_path_1 = og_path.split('\\')
    length = len(og_path_1)
    og_path_1 = og_path_1[length-1]

    og_path = og_path.split(f'{og_path_1}')[0]

    return og_path


def new_path(factor_mult):

    directory = directory_get()

    newdirectory = f'{directory}{factor_mult}_winglength\\'

    if not os.path.exists(newdirectory):

        os.makedirs(newdirectory)

    return newdirectory


def plotting(csv_file):

    df = pd.read_csv(csv_file)
    df.plot(kind='scatter', x='x', y='y')
    plt.show()


def warehouse():

    base_directory = directory_get()

    directory = f'{base_directory}WINGS\\NACA_Coordinates\\Warehouse'

    if not os.path.exists(directory):

        os.makedirs(directory)

    return directory


def create_curve(coordinates_multip):

    x_1 = coordinates_multip.iloc[75].loc['x']
    y_1 = coordinates_multip.iloc[75].loc['y']

    x_2 = coordinates_multip.iloc[37].loc['x']
    y_2 = coordinates_multip.iloc[37].loc['y']

    diam = abs(y_1 - y_2)

    new_y = (y_1 + y_2)/2
    new_x = x_1 + diam/2.8

    return new_x, new_y


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


def new_tin_path(factor_mult, name):

    base_directory = directory_get()

    newdirectory = f'{base_directory}\\WINGS\\Mesh_ICEM\\{name}\\{factor_mult}\\'

    if not os.path.exists(newdirectory):

        os.makedirs(newdirectory)

    return newdirectory


def find_icrit(csv_read_1):

    i_crit = 0
    i_max = 0
    i_min = 0

    for i in range(len(csv_read_1)):
        if csv_read_1.iloc[i].loc['x'] == 0:
            i_min = i
            break

    for i in range(len(csv_read_1)):
        if csv_read_1.iloc[i].loc['x'] == 1:
            i_crit = i
            break

    for i in range(i_crit+1, len(csv_read_1)):
        if csv_read_1.iloc[i].loc['x'] == 1:
            i_max = i
            break

    return i_min, i_crit, i_max


def coordinates_inline(csv_read_1):

    casual_array_1x = {}
    casual_array_1y = {}

    casual_array_2x = {}
    casual_array_2y = {}

    df_inline = csv_read_1.sort_values('x', ascending=True)

    k = 0
    k_1 = 0
    k_2 = 0

    for i in range(len(csv_read_1)):

        if(i % 2) == 0:

            casual_array_1x[k] = df_inline.iloc[i].loc['x']
            casual_array_1y[k] = df_inline.iloc[i].loc['y']

            k_1 += 1


        if(i % 2) != 0:

            casual_array_2x[k] = df_inline.iloc[i].loc['x']
            casual_array_2y[k] = df_inline.iloc[i].loc['y']

            k_2 += 1

        k += 1

    casual_array_x = {}
    casual_array_y = {}

    i_1 = 0

    for i in range(0, len(df_inline), 2):
        casual_array_x[i_1] = casual_array_1x[i]
        casual_array_y[i_1] = casual_array_1y[i]

        i_1 += 1

    for i in range(1, len(df_inline)-1, 2):
        casual_array_x[i_1] = casual_array_2x[i]
        casual_array_y[i_1] = casual_array_2y[i]

        i_1 += 1

    df_il = pd.DataFrame({'x': casual_array_x,
                          'y': casual_array_y})


    return df_il


def clean_array(raw_array):

    x = raw_array[2]
    y = raw_array[4].split('\n')[0]

    return x,y


def updown_points(path):

    file = open(path, 'r')

    file_pulp = {}
    pulp_final = {}

    size = 0

    for line in file:

        file_pulp[size] = line.split(' ')
        size += 1

    k = 0

    up_x = {}
    up_y = {}
    down_x = {}
    down_y = {}

    i_up = 0
    i_down = 0


    for i in range(3, size):

        if k == 0 and file_pulp[i][0] != '\n':

            up_x[i_up], up_y[i_up] = clean_array(file_pulp[i])

            i_up += 1


        if k == 1 and file_pulp[i][0] != '\n':

            down_x[i_down], down_y[i_down] = clean_array(file_pulp[i])

            i_down += 1


        if file_pulp[i][0] == '\n':

            k = 1


    points_df = pd.DataFrame({'x_down': down_x,
                              'y_down': down_y,
                              'x_up': up_x,
                              'y_up': up_y})

    return points_df


def edges(path):

    points_df = updown_points(path)

    interp_x = {}

    interp_y_up = {}
    interp_y_down = {}
    x_coord = {}
    i = 0

    num = 240

    for x in np.arange(0, 1, 1 / num):

        interp_y_up[i], interp_y_down[i] = interpolate_spline(points_df, x)
        interp_x[i] = x
        x_coord[i] = x

        i = i + 1

    coordinates_df = pd.DataFrame({'y_up': interp_y_up,
                                   'y_down': interp_y_down,
                                   'x': x_coord})


    return coordinates_df



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

    print(coordinates_df)

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


def directories_loop(factor_mult, newdirectory):

    base_directory = directory_get()

    wing_directory = f'{base_directory}WINGS\\NACA_Coordinates\\Warehouse\\'

    directory = os.fsencode(wing_directory)

    name = {}
    coordinates_multip = {}
    newest_directory = {}
    txt_name = {}

    cool_df = {}

    i = 0

    for file in os.listdir(directory):

         filename = os.fsdecode(file)

         if filename.endswith(".txt"):

             name[i] = wing_directory + filename

             coordinates_multip[i] = wing_multip(name[i], factor_mult)

             cool_array_x = {}
             cool_array_y = {}

             for n in range(len(coordinates_multip[i])):

                 cool_array_y[n] = coordinates_multip[i].iloc[n].loc['y_up']
                 cool_array_x[n] = coordinates_multip[i].iloc[n].loc['x']

             n_2 = len(coordinates_multip[i])

             for n in range(len(coordinates_multip[i])):

                 cool_array_y[n_2] = coordinates_multip[i].iloc[n].loc['y_down']
                 cool_array_x[n_2] = coordinates_multip[i].iloc[n].loc['x']

                 n_2 += 1

             cool_df[i] = pd.DataFrame({'x': cool_array_x,
                                        'y': cool_array_y})

             txt_name[i] = f'{filename}.csv'
             newest_directory[i] = newdirectory + txt_name[i]

             cool_df[i].to_csv(newest_directory[i], index = None, header=True)

             txt_name[i] = txt_name[i].split('.txt')[0]

             new_tin_path(factor_mult, txt_name[i])

             create_replay(name[i], factor_mult, txt_name[i])

             create_tin(cool_df[i], factor_mult, txt_name[i])

             create_project(cool_df[i], factor_mult, txt_name[i])

             i += 1

         else:
             continue


def main():

    warehouse()

    input_num = float(input('Enter the length of the wing (mm): '))

    try:

        factor_mult = input_num

    except:

        print('Invalid input')

    newdirectory = new_path(factor_mult)

    directories_loop(factor_mult, newdirectory)

    print()
    input('Press any key to exit, my work is done!')

if __name__ == "__main__":

    main()
