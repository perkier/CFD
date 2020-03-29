import warnings
import os, copy, sys
from datetime import date
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns

import pickle


def see_all():
    # Alongate the view on DataFrames

    pd.set_option('display.max_rows', 1000)
    pd.set_option('display.max_columns', 1000)
    pd.set_option('display.width', 1000000)



def edit_case(case, airfoil, angle):

    old_airfoil = str(f'[Airfoil]')
    old_angle = str(f'[Angle]')
    old_radians = str(f'[RADIANS]')

    new_airfoil = str(airfoil)
    new_angle = str(angle)
    new_radians = str(math.radians(angle))

    case = case.replace(old_airfoil, new_airfoil)
    case = case.replace(old_angle, new_angle)
    case = case.replace(old_radians, new_radians)

    return case


def edit_mesh(case, alpha_min, alpha_max, delta_alpha):

    mesh_journals = {}
    new_case = ''

    for i in range(alpha_min, alpha_max, delta_alpha):

        copy_case = copy.copy(case)

        old_pos = str(f'Angle')
        new_pos = str(i)

        copy_case = copy_case.replace(old_pos, new_pos)

        new_case += copy_case

        new_case += '\n'*5

        del copy_case

    return new_case




def get_previous_directory(root_dir):

    for subdir, dirs, files in os.walk(root_dir):

        for dir in dirs:

            if dir == 'DATA':
                data_path = f'{root_dir}{dir}'
                break

            else:
                pass

    new_root_dir = '\\'.join(root_dir.split("\\")[0:-1]) + '\\'

    if root_dir.split("\\")[-1] == '':
        new_root_dir = '\\'.join(root_dir.split("\\")[0:-2]) + '\\'

    try:
        data_path.split('DATA')

    except:
        data_path = ''

    return new_root_dir, data_path




def find_data_path():
    # Return DATA Folder Path

    rootdir = sys.path[0]

    for i in range(len(rootdir.split('\\'))):

        rootdir, data_path = get_previous_directory(rootdir)

        if data_path != '':
            break

    final_data_path = f'{data_path}\\Journals_Airfoil\\'

    if not os.path.exists(final_data_path):
        os.makedirs(final_data_path)

    return final_data_path


def read_jou():

    base_directory = find_data_path()

    journal_list = {}
    i = 0

    for file in os.listdir(base_directory):

        filename = os.fsdecode(file)

        if filename.endswith(".jou"):

            jou_read = open(f'{base_directory}{filename}', 'r')
            jou_script = jou_read.read()
            jou_read.close()

            if filename.split('.jou')[0] == 'MESH':
                mesh_journal = jou_script

            if filename.split('.jou')[0] == 'CASE':
                case_journal = jou_script

            journal_list[i] = jou_script

            i += 1

    try:
        return journal_list, mesh_journal, case_journal

    except:
        return journal_list


def Plotting_2D_lines(df, x_column, y_column, fig, ax, data, title, color):

    data_name = f'{data}'

    fig.suptitle(title, fontsize=20)

    plt.xlabel(f'{x_column}', fontsize=18)
    plt.ylabel(f'{y_column}', fontsize=16)

    plt.plot(df[x_column], df[y_column], '-o', color= color, linewidth=2.5)
    ax.scatter(x=df[x_column], y=df[y_column], linestyle='--', marker='o', s=100, label=data_name, color= color)

    leg = ax.legend()
    leg.set_title(title, prop={'size': 22})

    ax.legend(fontsize=30, markerscale=1., scatterpoints=5, prop={'size': 30})

    # ax.set_ylim(0, 1)


def double_Plotting(df, x_column, y_column, bar_column, fig, ax, data, title, color):

    # Plot_1

    data_name = f'{data}'

    fig.suptitle(title, fontsize=20)

    plt.xlabel(f'{x_column}', fontsize=18)
    plt.ylabel(f'{y_column}', fontsize=16)

    plt.plot(df[x_column], df[y_column], '-o', color= color, linewidth=2.5)
    ax[0].scatter(x=df[x_column], y=df[y_column], linestyle='--', marker='o', s=100, label=data_name, color= color)

    leg = ax[0].legend()
    leg.set_title(title, prop={'size': 22})

    ax[0].legend(fontsize=30, markerscale=1., scatterpoints=5, prop={'size': 30})

    # Plot 2

    objects = [df.iloc[0].loc['Airfoil']]
    y_pos = np.arange(len(objects))

    ax[1].bar(y_pos, df.loc[:,bar_column], align='center', alpha=0.5)
    # ax[1].xticks(y_pos, objects)
    # ax[1].ylabel('Power')
    # ax[1].title('Power Consumption per Airfoil')


class Mesh_Creation(object):

    def __init__(self):

        from NACA_wings import new_path, directories_loop

        warehouse()

        self.input_num = float(input('Enter the length of the wing (mm): '))

        try:

            factor_mult = input_num

        except:

            print('Invalid input')
            quit()

        newdirectory = new_path(factor_mult)

        directories_loop(factor_mult, newdirectory)

        print()
        input('Press any key to exit, my work is done!')
        quit()


class Journal_Creator(object):

    def __init__(self, alpha_min, alpha_max, sim_num, delta_alpha=1):

        see_all()

        self.alpha_dict = {'min': alpha_min, 'max': alpha_max, 'delta_ang': delta_alpha}

        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.delta_alpha = delta_alpha
        self.sim_num = sim_num

        self.final_data_path = find_data_path()

        self.create_dirs_0()


    def read_template_journals(self):

        self.journal_list, self.mesh_journal, self.case_journal = read_jou()

        # new_mesh_journal = edit_mesh(self.mesh_journal, self.alpha_min, self.alpha_max, self.delta_alpha)
        new_mesh_journal = self.mesh_journal

        self.template_journal = self.case_journal + '\n' * 3 + new_mesh_journal

        return self.template_journal


    def create_dirs_0(self):

        self.sim_name = f'Airfoil_Simulation_{self.sim_num}'

        self.sim_directory = self.final_data_path.split('DATA')[0]
        self.sim_directory = f'{self.sim_directory}Simulations\\{self.sim_name}\\'

        if not os.path.exists(self.sim_directory):
            os.makedirs(self.sim_directory)

        self.meshes_directory = f'{self.sim_directory}Meshes\\'

        if not os.path.exists(self.meshes_directory):
            os.makedirs(self.meshes_directory)

        self.journals_directory = f'{self.sim_directory}Journals\\'

        if not os.path.exists(self.journals_directory):
            os.makedirs(self.journals_directory)

        self.cases_directory = f'{self.sim_directory}Cases\\'

        if not os.path.exists(self.cases_directory):
            os.makedirs(self.cases_directory)

        self.airfoils_directory = f'{self.sim_directory}Airfoils\\'

        if not os.path.exists(self.airfoils_directory):
            os.makedirs(self.airfoils_directory)

        print(f'Please put the meshes on {self.meshes_directory}')


    def create_dirs(self):

        _count = 0

        Airfoils_dict = {}
        Angles_dict = {}
        path_dict = {}
        path_Airfoils = {}
        path_Angles = {}

        for file in os.listdir(self.meshes_directory):

            if file.split('.')[1] == 'msh':

                airfoil = file.split('.msh')[0]

                airfoil_directory = f'{self.airfoils_directory}{airfoil}\\'

                if not os.path.exists(airfoil_directory):
                    os.makedirs(airfoil_directory)

                Case_directory = f'{airfoil_directory}CASE\\'

                if not os.path.exists(Case_directory):
                    os.makedirs(Case_directory)

                for angle in range(self.alpha_min, self.alpha_max, self.delta_alpha):

                    angle_directory = f'{airfoil_directory}{angle}\\'

                    if not os.path.exists(angle_directory):
                        os.makedirs(angle_directory)

                    Airfoils_dict[_count] = airfoil
                    Angles_dict[_count] = angle
                    path_Airfoils[_count] = airfoil_directory
                    path_Angles[_count] = angle_directory

                    _count += 1


        self.path_df = pd.DataFrame({'Airfoil': Airfoils_dict,
                                     'Angle': Angles_dict,
                                     'Airfoil_path': path_Airfoils,
                                     'Angles_path': path_Angles})

        return self.path_df


    def edit_journals(self):

        self.create_dirs()
        self.read_template_journals()

        changing_journal = self.template_journal


        old_dir = str(f'MAIN_DICT')
        new_dir = str(self.sim_directory)
        changing_journal = changing_journal.replace(old_dir, new_dir)

        # old_mm = str(f'Num')
        # new_mm = str(self.spindle_num)
        # changing_journal = changing_journal.replace(old_mm, new_mm)
        #
        # print(old_mm)
        # print(new_mm)

        unique_airfoil = self.path_df.loc[:, 'Airfoil'].unique()
        unique_angle = self.path_df.loc[:, 'Angle'].unique()
        i = 0

        sub_dict_journal = ''

        for airfoil in unique_airfoil:
            for angle in unique_angle:

                changed_journal = edit_case(changing_journal, airfoil, angle)

                sub_df = self.path_df

                sub_df = sub_df.loc[sub_df['Angle'] == angle]
                sub_df = sub_df.loc[sub_df['Airfoil'] == airfoil]

                angle_path = sub_df.iloc[0].loc['Angles_path']

                old_sub = str(f'SUB_DICT')
                new_sub = str(angle_path)

                sub_dict_journal += changed_journal.replace(old_sub, new_sub)

                sub_dict_journal += '\n'*5

                i += 1

            sub_dict_journal += '\n' * 10

        sub_dict_journal = sub_dict_journal.replace("\\", "/")
        sub_dict_journal = sub_dict_journal.replace("//", "/")

        self.final_journal = sub_dict_journal

        today = date.today()

        f = open(f"{self.journals_directory}{today}.jou", "w+")
        f.write(self.final_journal)
        f.close()


class post_processing(object):

    def __init__(self, path_df):

        self.path_df = path_df

        self.posprocessing_df = pd.DataFrame({'Airfoil': [],
                                              'Angle': [],
                                              'Lift_Coeff': [],
                                              'Lift_Force': [],
                                              'Drag_Coeff': [],
                                              'Drag_Force': [],
                                              'Iterations': [],
                                              'Std_Lift': [],
                                              'Std_Drag': []})

        self.check_done()


    def take_values(self, part, file, path):

        file = f'{path}{file}'

        csv_reader = open(file, 'rb')
        csv_read = pd.read_csv(csv_reader, encoding='utf-8', delimiter=' ')
        csv_reader.close()

        df_column = list(csv_read)

        final_value = float(csv_read.iloc[len(csv_read) - 1].loc[df_column[0]])

        parameter = part[0].split('-')[0]

        return parameter, final_value, csv_read


    def get_iterations(self, df):

        iteration = df.iloc[len(df) - 1].name

        return iteration


    def check_convergency(self, df, number):

        df_column = list(df)

        try:

            last_ten = df.loc[:, df_column[0]].tail(number).astype(float).reset_index(drop=True)

            maximum = last_ten.describe().loc['max']
            minimum = last_ten.describe().loc['min']

            last_value = df.loc[:, df_column[0]].tail(1).astype(float).reset_index(drop=True).iloc[0]

            diff = abs(maximum - minimum)
            diff_perc = 100 * (diff / last_value)

            std = float(diff_perc)
            # std = diff

        except:

            std = "std_ERROR"

        return std


    def check_done(self):

        j = 0
        done = {}

        for i in range(len(self.path_df)):

            done[i] = []

            arr = os.listdir(self.path_df.iloc[i].loc['Angles_path'])

            drag_coeff = 0
            lift_coeff = 0
            drag_force = 0
            lift_force = 0

            drag_std = 0
            lift_std = 0

            iterarions = 0

            switch = 0

            for file in arr:

                ext = file.split('.')

                if ext[1] == 'out':

                    switch = 1

                    parameter, final_value, df = self.take_values(ext, file, self.path_df.iloc[i].loc['Angles_path'])

                    iterations = self.get_iterations(df)
                    std = self.check_convergency(df, 5)

                    if parameter == 'drag_coeff':
                        drag_coeff = final_value
                        drag_std = std

                    elif parameter == 'lift_coeff':
                        lift_coeff = final_value
                        lift_std = std

                    elif parameter == 'drag_force':
                        drag_force = final_value

                    elif parameter == 'lift_force':
                        lift_force = final_value

            if switch == 1:

                to_append = pd.DataFrame({'Airfoil': [self.path_df.iloc[i].loc['Airfoil']],
                                          'Angle': [self.path_df.iloc[i].loc['Angle']],
                                          'Lift_Coeff': [lift_coeff],
                                          'Lift_Force': [lift_force],
                                          'Drag_Coeff': [drag_coeff],
                                          'Drag_Force': [drag_force],
                                          'Iterations': [iterations],
                                          'Std_Lift': [lift_std],
                                          'Std_Drag': [drag_std]})

                self.posprocessing_df = pd.concat([self.posprocessing_df, to_append], ignore_index=True)

        self.create_LD()

    def create_LD(self):

        self.posprocessing_df['Lift_Drag_Ratio'] = self.posprocessing_df['Lift_Coeff'] / self.posprocessing_df['Drag_Coeff']


    def plotting(self):

        dir = self.path_df.iloc[0].loc['Airfoil_path'].split('Airfoils')[0]

        plt_directory = f'{dir}\\Plots\\'

        if not os.path.exists(plt_directory):
            os.makedirs(plt_directory)

        self.unique_aifoils = self.posprocessing_df.loc[:, 'Airfoil'].unique()

        color_pallete = ['turquoise', 'springgreen', 'khaki', 'violet', 'deepskyblue', 'violet', 'peru','c', 'lime', 'darkkhaki', 'palevioletred', 'deepskyblue', 'violet', 'peru']

        j = 0
        fig, ax = plt.subplots(figsize=(30, 15))

        df = self.posprocessing_df

        parameters = ['Lift_Drag_Ratio', 'Lift_Coeff', 'Drag_Coeff']

        for title in parameters:

            for airfoil in self.unique_aifoils:

                unique_df = df.loc[df['Airfoil'] == airfoil].reset_index(drop=True)

                Plotting_2D_lines(unique_df, 'Angle', title, fig, ax, airfoil, title, color_pallete[j])

                j += 1

                # print(unique_df, '\n')

            plt.savefig(f'{plt_directory}{title}.png', dpi=300)
            plt.clf()



class flight_simulator(object):

    def __init__(self, postprocessing_df):

        self.mass = 30
        self.wing_length = 0.200
        self.depth = 1
        self.area = self.wing_length * self.depth

        self.init_h = 50
        self.init_v = 36.14
        self.Reynolds = 50000
        self.g = 9.81

        self.df = postprocessing_df.loc[postprocessing_df['Std_Lift'] < 1].loc[postprocessing_df['Std_Drag'] < 1].reset_index(drop=True)


    def const_speed(self):

        self.lift_force()
        self.drag_force()

        airfoils_dict = {}

        for i in range(len(self.df)):

            airfoil_name = f"{self.df.iloc[i].loc['Airfoil']}{[self.df.iloc[i].loc['Angle']]}deg"

            Drag = self.df.iloc[i].loc['Drag_Force']
            Lift = self.df.iloc[i].loc['Lift_Force']
            Thrust = Drag
            Power = Thrust * self.init_v

            h = self.init_h
            dist = 0

            safety_breack = 0

            airfoils_dict[airfoil_name] = pd.DataFrame({'Airfoil': [airfoil_name],
                                                        'Angle': [self.df.iloc[i].loc['Angle']],
                                                        'Height': [h],
                                                        'Distance': [dist],
                                                        'Power': [Power],
                                                        'cL/cD': [self.df.iloc[i].loc['Lift_Drag_Ratio']]})

            while h >= 0:

                safety_breack += 1

                h -= Lift / (self.mass * self.g)
                dist += self.init_v

                to_append = pd.DataFrame({'Airfoil': [airfoil_name],
                                          'Angle': [self.df.iloc[i].loc['Angle']],
                                          'Height': [h],
                                          'Distance': [dist],
                                          'Power': [Power],
                                          'cL/cD': [self.df.iloc[i].loc['Lift_Drag_Ratio']]})

                airfoils_dict[airfoil_name] = pd.concat([airfoils_dict[airfoil_name], to_append], ignore_index=True)

                if safety_breack > 1000:
                    break

        self.cte_speed_dict = airfoils_dict


    def plotting_cte_speed(self):

        # dir = self.path_df.iloc[0].loc['Airfoil_path'].split('Airfoils')[0]
        # plt_directory = f'{dir}\\Plots\\'

        # if not os.path.exists(plt_directory):
        #     os.makedirs(plt_directory)

        airfoils = self.cte_speed_dict.keys()

        color_pallete = ['turquoise', 'springgreen', 'khaki', 'violet', 'deepskyblue', 'violet', 'peru', 'c', 'lime',
                         'darkkhaki', 'palevioletred', 'deepskyblue', 'violet', 'peru','turquoise', 'springgreen', 'khaki', 'violet', 'deepskyblue', 'violet', 'peru', 'c', 'lime',
                         'darkkhaki', 'palevioletred', 'deepskyblue', 'violet', 'peru', 'palevioletred', 'deepskyblue', 'violet', 'peru','turquoise', 'springgreen', 'khaki', 'violet', 'deepskyblue', 'violet', 'peru', 'c', 'lime',
                         'darkkhaki', 'palevioletred', 'deepskyblue', 'violet', 'peru']


        j = 0
        fig, axs = plt.subplots(2, figsize=(30, 15))

        title = "Distance covered"

        for airfoil in airfoils:

            unique_df = self.cte_speed_dict[airfoil]

            cL_cD = unique_df.iloc[0].loc['cL/cD']

            if cL_cD >= 40:

                double_Plotting(unique_df, 'Distance', 'Height', 'Power',fig, axs, airfoil, title, color_pallete[j])

            else:
                pass

            j += 1

            # print(unique_df, '\n')

        plt.savefig(f'C:\\Users\\diogo\\Desktop\\inegi\\Simulations\\Airfoil_Simulation_1\\Plots\\{title}.png', dpi=300)
        plt.clf()


    def drag_force(self):

        self.df = self.df.drop(['Drag_Force'], axis=1)

        divided_by = 1.1767 * self.area * ( 0.5 * self.init_v * self.init_v)

        self.df['Drag_Force'] = self.df['Drag_Coeff'].apply(lambda x: x*divided_by)

    def lift_force(self):

        self.df = self.df.drop(['Lift_Force'], axis=1)

        divided_by = 1.1767 * self.area * (0.5 * self.init_v * self.init_v)

        self.df['Lift_Force'] = self.df['Lift_Coeff'].apply(lambda x: x * divided_by)


def main():

    joujou = Journal_Creator(-10, 6, 1, 1)
    path_df = joujou.create_dirs()


    # journal.read_template_journals()
    joujou.edit_journals()

    posproc = post_processing(path_df)
    posproc.plotting()

    results_df = posproc.posprocessing_df

    print('\n'*10)

    sim = flight_simulator(results_df)
    sim.const_speed()

    pickle.dump(sim, open("C:\\Users\\diogo\\Desktop\\inegi\\Simulations\\Airfoil_Simulation_1\\Plots\\pickle_rick.p", "wb"))
    # sim = pickle.load( open("C:\\Users\\diogo\\Desktop\\inegi\\Simulations\\Airfoil_Simulation_1\\Plots\\pickle_rick.p", "rb" ) )

    sim.plotting_cte_speed()

    quit()


if __name__ == '__main__':

    see_all()
    main()
