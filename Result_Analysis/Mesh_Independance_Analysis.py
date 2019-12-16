import pandas as pd
import numpy as np
import sys
import seaborn as sns
import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from collections import OrderedDict
import matplotlib as mpl



def find_data_path():
    # Return DATA Folder Path

    data_path = sys.path[0].split('CODE')[0]
    data_path = f'{data_path}DATA\\Mesh_Independance\\'

    return data_path


def see_all():
    # Alongate the view on DataFrames

    pd.set_option('display.max_rows', 1000)
    pd.set_option('display.max_columns', 1000)
    pd.set_option('display.width', 1000000)


def csv_func(name):

    csv_reader = open(f'{find_data_path() + name }.txt', 'rb')
    csv_read = pd.read_csv(csv_reader, encoding='latin1', delimiter=',')
    csv_reader.close()

    return csv_read


def convert_float(df):

    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64', 'object']

    for col in df.columns:

        col_type = df[col].dtypes

        if col_type in numerics:

                c_min = df[col].min()
                c_max = df[col].max()

                if str(col_type)[:3] == 'int':

                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)

                    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)

                    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)

                    elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                        df[col] = df[col].astype(np.int64)

                else:

                    if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                        df[col] = df[col].astype(np.float16)

                    elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                        df[col] = df[col].astype(np.float16)

                    else:
                        df[col] = df[col].astype(np.float16)

                if col_type == 'object':
                    df = df.drop(df[col])

        return df


def get_data():

    base_directory = find_data_path()
    base_directory = f'{base_directory}'

    directory = os.fsencode(base_directory)

    j = 0

    files = {}

    for file in os.listdir(directory):

         filename = os.fsdecode(file)

         if filename.endswith(".txt"):

            file_name = filename.split('.txt')[0]

            files[j] = csv_func(file_name)

            j += 1

    return files


def find_neighbours(df, x_value, y_value, x, y):

    exactmatch = df.loc[df[x] == x_value]
    exactmatch = exactmatch.loc[exactmatch[y] == y_value].reset_index(drop=True)

    returning_dict = {}

    if exactmatch.empty:

        neighbour_x0 = df.loc[df[y] <= y_value].reset_index(drop=True)

        try:
            neighbour_x0 = neighbour_x0.loc[neighbour_x0[x] >= x_value].sort_values(by=['x-coordinate', 'y-coordinate'], ascending=True).reset_index(drop=True).iloc[0]

        except:
            neighbour_x0 = neighbour_x0.sort_values(by=['x-coordinate', 'y-coordinate'], ascending=False).reset_index(drop=True).iloc[0]

            # print('error in neighbour 1')
            # print(df.loc[df[y] <= y_value].sort_values(by=['x-coordinate', 'y-coordinate'], ascending=True).tail())
            # print(df.iloc[143321].loc['x-coordinate'])
            # print(x_value)
            # quit()

        neighbour_x1 = df.loc[df[y] <= y_value].reset_index(drop=True)

        try:
            neighbour_x1 = neighbour_x1.loc[neighbour_x1[x] <= x_value].sort_values(by=['x-coordinate', 'y-coordinate'], ascending=True).reset_index(drop=True).iloc[0]

        except:
            pass

        neighbour_x2 = df.loc[df[y] >= y_value].reset_index(drop=True)

        try:
            neighbour_x2 = neighbour_x2.loc[neighbour_x2[x] <= x_value].sort_values(by=['x-coordinate', 'y-coordinate'], ascending=True).reset_index(drop=True).iloc[0]

        except:
            pass

        neighbour_x3 = df.loc[df[y] >= y_value].reset_index(drop=True)

        try:
            neighbour_x3 = neighbour_x3.loc[neighbour_x3[x] >= x_value].sort_values(by=['x-coordinate', 'y-coordinate'], ascending=True).reset_index(drop=True).iloc[0]

        except:
            neighbour_x3 = neighbour_x3.loc[neighbour_x3[x] <= x_value].sort_values(by=['x-coordinate', 'y-coordinate'], ascending=True).reset_index(drop=True).iloc[0]

        returning_dict = {'neighbour_x0': neighbour_x0, 'neighbour_x1': neighbour_x1, 'neighbour_x2': neighbour_x2, 'neighbour_x3': neighbour_x3}

    else:

        returning_dict = {'exact': exactmatch.iloc[0]}

    return returning_dict


def interpolating(point_1, point_2, new, param_1, param_2):

    a = np.array([[point_1[param_1], 1], [point_2[param_1], 1]])
    b = np.array([point_1[param_2], point_2[param_2]])

    curve = np.linalg.solve(a , b)

    new_y = curve[0]*new + curve[1]

    new_pos = {param_1: new, param_2: new_y}

    return new_pos


def interpolate_x_y(returning_dict, max_df_x, max_df_y, x, y):

    x_0 = {'x': returning_dict['neighbour_x0'].loc['x-coordinate'], 'param': returning_dict['neighbour_x0'].loc['velocity-magnitude'] }
    x_1 = {'x': returning_dict['neighbour_x1'].loc['x-coordinate'], 'param': returning_dict['neighbour_x1'].loc['velocity-magnitude'] }

    x_2 = {'x': returning_dict['neighbour_x2'].loc['x-coordinate'], 'param': returning_dict['neighbour_x2'].loc['velocity-magnitude'] }
    x_3 = {'x': returning_dict['neighbour_x3'].loc['x-coordinate'], 'param': returning_dict['neighbour_x3'].loc['velocity-magnitude'] }
    
    if x_0 == x_1:
        param_0 = (x_0 + x_1) / 2

    else:
        param_0 = interpolating(x_0, x_1, max_df_x, 'x', 'param')['param']

    if x_2 == x_3:
        param_1 = (x_2 + x_3) / 2

    else:
        param_1 = interpolating(x_2, x_3, max_df_x, 'x', 'param')['param']

    y_0 = (returning_dict['neighbour_x0'].loc['y-coordinate'] + returning_dict['neighbour_x1'].loc['y-coordinate']) / 2
    y_1 = (returning_dict['neighbour_x2'].loc['y-coordinate'] + returning_dict['neighbour_x3'].loc['y-coordinate']) / 2

    y_tointerp_0 = {'y': y_0, 'param': param_0}
    y_tointerp_1 = {'y': y_1, 'param': param_1}

    if y_0 == y_1:

        param_value = (param_0 + param_1) / 2

    else:

        param_value = interpolating(y_tointerp_0, y_tointerp_1, max_df_y, 'y', 'param')['param']

    return param_value


def get_same_length(DataFrames):

    max_length = 0
    max_index = 0

    for i in range(len(DataFrames)):

        len_i = len(DataFrames[i])

        if len_i > max_length:

            max_index = i
            max_length = len_i

    max_df = DataFrames[max_index]

    interpolated_df = {}

    for i in range(len(DataFrames)):

        interpolated_df[i] = pd.DataFrame(columns=['x-coordinate', 'y-coordinate', 'velocity-to_test', 'velocity-max_mesh'])

        if i == max_index:
            pass

        else:

            df = DataFrames[i].sort_values(by=['x-coordinate', 'x-coordinate'], ascending=True).reset_index(drop=True)

            for j in range(0, len(max_df), 5):

                max_df_x = max_df.iloc[j].loc['x-coordinate']
                max_df_y = max_df.iloc[j].loc['y-coordinate']

                x = 'x-coordinate'
                y = 'y-coordinate'

                returning_dict = find_neighbours(df, max_df_x, max_df_y, x, y)

                if len(returning_dict) == 1:

                    interpolated_df[i] = interpolated_df[i].append({'x-coordinate': max_df_x,
                                                                    'y-coordinate': max_df_y,
                                                                    'velocity-to_test': returning_dict['exact'].loc['velocity-magnitude'],
                                                                    'velocity-max_mesh': max_df.iloc[j].loc['velocity-magnitude']}, ignore_index=True)

                else:

                    interpolated_df[i] = interpolated_df[i].append({'x-coordinate': max_df_x,
                                                                    'y-coordinate': max_df_y,
                                                                    'velocity-to_test': interpolate_x_y(returning_dict, max_df_x, max_df_y, x, y),
                                                                    'velocity-max_mesh': max_df.iloc[j].loc['velocity-magnitude']}, ignore_index=True)

                # if j == 100:
                #     break
                # print(j)

    return interpolated_df


def get_error(df):

    df['error'] = df['velocity-to_test'] - df['velocity-max_mesh']
    df['error'] = 100 * df['error'] / df['velocity-max_mesh']

    return df


def plotting_colors(df, x_column, y_column, z_column, fig, ax, data, title, max_error, min_error):

    data_name = f'{data}'

    fig.suptitle(title, fontsize=20)
    plt.xlabel(f'{x_column}', fontsize=18)
    plt.ylabel(f'{y_column}', fontsize=16)

    color_map = plt.cm.get_cmap('RdYlGn')

    ax.scatter(x=df[x_column], y=df[y_column], c=df[z_column], cmap=color_map,
               linestyle='--', marker='o', s=1, label=data_name)

    ax.legend(markerscale=4., scatterpoints=5)
    # plt.colorbar(ax)

    # im = plt.imshow(df[z_column], cmap=plt.cm.RdBu, extent=(-3, 3, 3, -3), interpolation='bilinear')

    # plt.colorbar(im)

    # if type(max_error) == str:
    #     pass
    #
    # elif type(min_error) == str:
    #     pass
    #
    # else:
    #     ax.set_ylim(y_min * 0.9, y_max * 1.1)




def main():

    sns.set_style("whitegrid")

    fluids_df_dict = get_data()

    fluids_df_dict = get_same_length(fluids_df_dict)

    for i in range(len(fluids_df_dict)):

        if len(fluids_df_dict[i]) == 0:
            pass

        else:

            error_df = get_error(fluids_df_dict[i])

            info = 'NIST'

            plot_title = 'Viscosity vs. x-coordinate'

            max_error = error_df.describe().loc[:, 'error'].loc['max']
            min_error = error_df.describe().loc[:, 'error'].loc['min']

            fig, ax = plt.subplots()

            plotting_colors(error_df, 'x-coordinate', 'y-coordinate', 'error',
                            fig, ax, info,
                            plot_title, max_error, min_error)

            plt.show()


    quit()

if __name__ == '__main__':

    see_all()
    sns.set()
    main()
