import pandas as pd
from CoolProp.CoolProp import PropsSI, PhaseSI, HAPropsSI
import math
from math import sqrt

def converttemp(IN, OUT, T):

    if IN == "C":
        return T + 273.15

    if IN == "K":
        return T - 273.15

    else:
        print("converttemp(IN, OUT, T) - In and Out should be 'K' or 'C' ")
        quit()

def main():

    Fluid = "Water"
    Te = 10
    pi = math.pi
    T_c = 30 + 273
    T_g = 90 + 273
    T_e = 10 + 273
    T_sup = 5

    p_g = PropsSI('P', 'T', T_g, 'Q', 1, Fluid)
    h_g = PropsSI('H', 'T', T_g, 'P', p_g, Fluid)
    h_e = PropsSI('H', 'T', T_e, 'Q', 1, Fluid)

    rho_g = PropsSI('D', 'T', T_g + T_sup, 'P', p_g, Fluid)             # fluid density leaving the generator

    eta_noz = 0.8   # nozzle eficiency
    eta_s = 0.65    # mixing efficiency
    eta_dif = 0.80  # diffuser efficiency

    phi_p = 0.9
    phi_m = 0.80

    R_univ = 8.314472       # [kJ * kmol ^ -1 * K ^ -1] - "universal gas constant KJ/MolK"

    cp_noz = PropsSI('Cpmass', 'T', 89.4 + 273.15 + 5, 'P', 101325*3 , Fluid)                               #  specific heat at constant pressure
    cv_noz = PropsSI('Cvmass', 'T', 89.4 + 273.15 + 5, 'P', 101325*3 , Fluid)                                                #  specific heat at constant volume
    gamma = cp_noz / cv_noz                                                                         #  specific heat ratio
    R_fluid = 1000 * R_univ / PropsSI('MOLARMASS', Fluid)                                                 # "gas constant of the fluid, J/kgK"

    dummy_1 = 2 / (gamma + 1)
    dummy_2 = (gamma - 1) / 2
    dummy_3 = (gamma + 1) / (gamma - 1)

    F_act = (gamma / R_fluid) * (dummy_1 ** dummy_3)

    print(gamma)
    print(R_fluid)
    print(F_act)

    quit()

    ## Primary nozzle equations

    p_e = PropsSI('P', 'T', int(Te) + 273.15, 'Q', 1, Fluid)
    p_noz_ex = p_e

    aux = gamma / (gamma - 1)

    Ma_noz_ex = sqrt((((p_g / p_noz_ex) ** (1 / aux)) - 1) / dummy_2)

    m_dot_g = (A_noz * p_g / sqrt(T_g + T_sup)) * sqrt(eta_noz * F_act)                             # mass flow rate in the nozzle according to Huang et al. , eq.(1)"
    A_noz_ex = (A_noz / Ma_noz_ex) * (dummy_1 * (1 + dummy_2 * (Ma_noz_ex ^ 2))) ** (dummy_3 / 2)    # eq.(2)

    # p_noz_ex = p_g / (1 + dummy_2 * (Ma_noz_ex ^ 2)) ^ (aux)                        # "pressure at nozzle exit, eq (3)"

    ## Mixing Section

    aux = -gamma / (gamma - 1)

    Ma_sec_y = 1
    p_sec_y = p_e * (1 + dummy_2 * (Ma_sec_y**2)) ** (aux)         # "pressure os secondary flow at section y - y, eq.6"

    p_pr_y = p_sec_y

    # p_pr_y = p_g * (1 + dummy_2 * (Ma_pr_y**2)) ^ (-gamma / (gamma - 1))           # "pressure of primary flow at section y - y, eq.4"
    Ma_pr_y = sqrt( (( (p_pr_y / p_g)**(1/aux) ) - 1) / dummy_2 )                   # pressure of primary flow at section y - y, eq.4

    A_pr_y = phi_p * A_noz / Ma_pr_y * (dummy_1 * (1 + dummy_2 * (Ma_pr_y**2))) ^ (dummy_3 / 2)        # "primary flow area at section y  -  y, eq.5 "
    d_pr_y = sqrt(4 * A_pr_y / pi)

    A_sec_y = A_y - A_pr_y
    # A_y = A_pr_y + A_sec_y              # "Total area  at section y - y, eq.8"

    A_y = pi * (d_y ** 2)  / 4
    d_ratio = d_y / d_noz

    A_ratio_noz = A_y / A_noz

    A_jet_ratio = A_pr_y / A_sec_y

    m_dot_e = (A_sec_y * p_e / sqrt(T_e)) * sqrt(eta_s * F_act)           # "mass flow rate in the nozzle according to Huang et al., eq. 7"

    T_pr_y_K = (T_g + T_sup) / (1 + dummy_2 * (Ma_pr_y**2))              # "temperature of primary stream at y-y in K, eq. 9."

    T_pr_y = converttemp('K', 'C', T_pr_y_K)                                # "temperature of primary stream at y-y in ºC, eq. 9."
    T_pr_y_ref = PropsSI('T', 'P', p_pr_y, 'Q', 1, Fluid)                 # "temperature of the primary stream considering saturation"

    T_sec_y_K = T_e / (1 + dummy_2 * (Ma_sec_y**2))                      # "temperature of secondary stream at y-y in K, eq. 10"
    T_sec_y = converttemp('K', 'C', T_sec_y_K)                              # "temperature of secondary stream at y-y in ºC, eq. 10"

    v_pr_y = Ma_pr_y * sqrt(gamma * R_fluid * T_pr_y_K)                     # "eq. 13"
    v_sec_y = Ma_sec_y * sqrt(gamma * R_fluid * T_sec_y_K)                  # "eq. 14"

    ER = m_dot_e / m_dot_g

    v_m = (v_pr_y + ER * v_sec_y) * sqrt(phi_m) / (1 + ER)            # "eq 5 Yu"

    Mom_pr_y = m_dot_g * v_pr_y
    Mom_sec_y = m_dot_e * v_sec_y
    Mom_m = (m_dot_g + m_dot_e) * v_m

    cp_mix = cp_noz

    E_tot_pr_y = T_pr_y * cp_mix + 0.5 * (v_pr_y)**2

    E_tot_sec_y = T_sec_y * cp_mix + 0.5 * (v_sec_y)**2

    E_tot_m = (m_dot_g * E_tot_pr_y + m_dot_e * E_tot_sec_y) / (m_dot_g + m_dot_e)       # "energy balance in the mixing section, eq. 12"
    T_m = E_tot_m / (cp_mix + 0.5 * (v_m)**2)

    T_m_K = converttemp('C', 'K', T_m)
    Ma_m = v_m / sqrt(gamma * R_fluid * T_m_k)                              # "eq. 15"


    # Shock Wave

    p_sh = p_pr_y * (1 + dummy_1 * gamma * ((Ma_m ^ 2) - 1))                            # "Pressure after the shock wave, eq. 16"
    Ma_sh = ((1 + dummy_2 * (Ma_m ^ 2)) / (gamma * (Ma_m ^ 2) - dummy_2)) ^ 0.5         # "eq. 17 - this was not correct in previous versions"

    # Diffuser

    p_c = p_sh * (1 + eta_dif * dummy_2 * (Ma_sh ^ 2)) ^ (gamma / (gamma - 1))          # "ejector exit pressure, eq. 18"

    p_c_crit = PropsSI('P', 'T', T_c, 'Q', 1, Fluid)                                         # "pressure corresponding to condenser conditions"

    dummy_5 = p_c - p_c_crit
    dummy_5 = 10

    h_c = (m_dot_g * h_g + m_dot_e * h_e) / (m_dot_g + m_dot_e)
    T_eje_out = PropsSI('T', 'P', p_c, 'H', h_c, Fluid)


def basic_interpolation(d_noz, p_g, T_g, T_sup, eta_noz, F_act, Ma_noz_ex, dummy_1, dummy_2, dummy_3, phi_p, Ma_pr_y, d_y, p_e, T_e, eta_s):

    pi = math.pi

    A_noz = pi * (d_noz ** 2) / 4  # "cross section at the nozzle"

    m_dot_g = (A_noz * p_g / sqrt(T_g + T_sup)) * sqrt(eta_noz * F_act)  # mass flow rate in the nozzle according to Huang et al. , eq.(1)"
    A_noz_ex = (A_noz / Ma_noz_ex) * (dummy_1 * (1 + dummy_2 * (Ma_noz_ex ** 2))) ** (dummy_3 / 2)  # eq.(2)

    d_noz_ex = sqrt(4 * A_noz_ex / pi)

    A_pr_y = phi_p * A_noz / Ma_pr_y * (dummy_1 * (1 + dummy_2 * (Ma_pr_y ** 2))) ** (dummy_3 / 2)  # "primary flow area at section y  -  y, eq.5 "
    d_pr_y = sqrt(4 * A_pr_y / pi)

    A_y = pi * (d_y ** 2) / 4

    A_sec_y = A_y - A_pr_y
    # A_y = A_pr_y + A_sec_y              # "Total area  at section y - y, eq.8"

    m_dot_e = (A_sec_y * p_e / sqrt(T_e)) * sqrt(eta_s * F_act)  # "mass flow rate in the nozzle according to Huang et al., eq. 7"

    return m_dot_e, m_dot_g


def basic_interpolation2(m_dot_e, d_noz, p_g, T_g, T_sup, eta_noz, F_act, Ma_noz_ex, dummy_1, dummy_2, dummy_3, phi_p, Ma_pr_y, d_y, p_e, T_e, eta_s):

    pi = math.pi

    A_noz = pi * (d_noz ** 2) / 4  # "cross section at the nozzle"

    m_dot_g = (A_noz * p_g / sqrt(T_g + T_sup)) * sqrt(eta_noz * F_act)  # mass flow rate in the nozzle according to Huang et al. , eq.(1)"
    A_noz_ex = (A_noz / Ma_noz_ex) * (dummy_1 * (1 + dummy_2 * (Ma_noz_ex ** 2))) ** (dummy_3 / 2)  # eq.(2)

    d_noz_ex = sqrt(4 * A_noz_ex / pi)

    A_pr_y = phi_p * A_noz / Ma_pr_y * (dummy_1 * (1 + dummy_2 * (Ma_pr_y ** 2))) ** (dummy_3 / 2)  # "primary flow area at section y  -  y, eq.5 "
    d_pr_y = sqrt(4 * A_pr_y / pi)

    # A_y = pi * (d_y ** 2) / 4

    # A_sec_y = A_y - A_pr_y
    # A_y = A_pr_y + A_sec_y              # "Total area  at section y - y, eq.8"

    A_sec_y = m_dot_e / (sqrt(eta_s * F_act) * (p_e / sqrt(T_e)))                  # "mass flow rate in the nozzle according to Huang et al., eq. 7"

    A_y = A_sec_y + A_pr_y

    d_y = sqrt(4 * A_y / pi)

    return d_y, m_dot_g





def main2():

    Fluid = "R152a"

    pi = math.pi
    T_c = 30 + 273
    T_g = 90 + 273
    T_e = 10 + 273
    T_sup = 5

    p_g = PropsSI('P', 'T', T_g, 'Q', 1, Fluid)
    h_g = PropsSI('H', 'T', T_g + T_sup, 'P', p_g, Fluid)
    h_e = PropsSI('H', 'T', T_e, 'Q', 1, Fluid)

    h_2 = PropsSI('H', 'T', T_c, 'Q', 0, Fluid)                              # "enthalpy at condeser outlet "

    rho_g = PropsSI('D', 'T', T_g + T_sup, 'P', p_g, Fluid)             # fluid density leaving the generator

    eta_noz = 0.8   # nozzle eficiency
    eta_s = 0.65    # mixing efficiency
    eta_dif = 0.80  # diffuser efficiency

    phi_p = 0.9
    phi_m = 0.80

    R_univ = 8.314472       # [kJ * kmol ^ -1 * K ^ -1] - "universal gas constant KJ/MolK"

    cp_noz = PropsSI('Cpmass', 'T', T_g + T_sup, 'P', p_g , Fluid)                               #  specific heat at constant pressure
    cv_noz = PropsSI('Cvmass', 'T', T_g + T_sup, 'P', p_g , Fluid)                                                #  specific heat at constant volume

    gamma = cp_noz / cv_noz                                                                         #  specific heat ratio
    R_fluid = 1000 * R_univ / PropsSI('MOLARMASS', Fluid)                                                 # "gas constant of the fluid, J/kgK"

    dummy_1 = 2 / (gamma + 1)
    dummy_2 = (gamma - 1) / 2
    dummy_3 = (gamma + 1) / (gamma - 1)

    F_act = (gamma / R_fluid) * (dummy_1 ** dummy_3)

    # print(gamma)
    # print(R_fluid)
    # print(F_act)
    #
    # quit()

    ## Primary nozzle equations

    p_e = PropsSI('P', 'T', T_e, 'Q', 1, Fluid)
    p_noz_ex = p_e

    aux = gamma / (gamma - 1)

    Ma_noz_ex = sqrt((((p_g / p_noz_ex) ** (1 / aux)) - 1) / dummy_2)

    # p_noz_ex = p_g / (1 + dummy_2 * (Ma_noz_ex ^ 2)) ^ (aux)                        # "pressure at nozzle exit, eq (3)"

    ## Mixing Section

    aux = -gamma / (gamma - 1)

    Ma_sec_y = 1
    p_sec_y = p_e * (1 + dummy_2 * (Ma_sec_y**2)) ** (aux)         # "pressure os secondary flow at section y - y, eq.6"

    p_pr_y = p_sec_y

    # p_pr_y = p_g * (1 + dummy_2 * (Ma_pr_y**2)) ^ (-gamma / (gamma - 1))           # "pressure of primary flow at section y - y, eq.4"
    Ma_pr_y = sqrt( (( (p_pr_y / p_g)**(1/aux) ) - 1) / dummy_2 )                   # pressure of primary flow at section y - y, eq.4

    T_pr_y_K = (T_g + T_sup) / (1 + dummy_2 * (Ma_pr_y**2))              # "temperature of primary stream at y-y in K, eq. 9."

    T_pr_y = converttemp('K', 'C', T_pr_y_K)                                # "temperature of primary stream at y-y in ºC, eq. 9."
    T_pr_y_ref = PropsSI('T', 'P', p_pr_y, 'Q', 1, Fluid)                 # "temperature of the primary stream considering saturation"

    T_sec_y_K = T_e / (1 + dummy_2 * (Ma_sec_y**2))                      # "temperature of secondary stream at y-y in K, eq. 10"
    T_sec_y = converttemp('K', 'C', T_sec_y_K)                              # "temperature of secondary stream at y-y in ºC, eq. 10"

    v_pr_y = Ma_pr_y * sqrt(gamma * R_fluid * T_pr_y_K)                     # "eq. 13"
    v_sec_y = Ma_sec_y * sqrt(gamma * R_fluid * T_sec_y_K)                  # "eq. 14"

    #####

    # d_y = 0.003779
    # d_noz = 0.001485

    # A_noz = pi * (d_noz ** 2) / 4                    # "cross section at the nozzle"
    #
    # m_dot_g = (A_noz * p_g / sqrt(T_g + T_sup)) * sqrt(eta_noz * F_act)                             # mass flow rate in the nozzle according to Huang et al. , eq.(1)"
    # A_noz_ex = (A_noz / Ma_noz_ex) * (dummy_1 * (1 + dummy_2 * (Ma_noz_ex ** 2))) ** (dummy_3 / 2)    # eq.(2)
    #
    # d_noz_ex = sqrt(4 * A_noz_ex / pi)
    #
    # A_pr_y = phi_p * A_noz / Ma_pr_y * (dummy_1 * (1 + dummy_2 * (Ma_pr_y**2))) ** (dummy_3 / 2)        # "primary flow area at section y  -  y, eq.5 "
    # d_pr_y = sqrt(4 * A_pr_y / pi)
    #
    # A_y = pi * (d_y ** 2) / 4
    #
    # A_sec_y = A_y - A_pr_y
    # # A_y = A_pr_y + A_sec_y              # "Total area  at section y - y, eq.8"
    #
    # m_dot_e = (A_sec_y * p_e / sqrt(T_e)) * sqrt(eta_s * F_act)  # "mass flow rate in the nozzle according to Huang et al., eq. 7"
    #
    # d_ratio = d_y / d_noz
    #
    # A_ratio_noz = A_y / A_noz
    # A_ratio = A_noz_ex / A_noz
    # A_jet_ratio = A_pr_y / A_sec_y


    ####

    d_y = 0.003779
    d_noz = 0.001485

    ER_max = 0
    best_d_y = 0
    best_d_noz = 0
    m_dot_e_ER = 0
    m_dot_g_ER = 0
    Q_e_ER = 0

    m_dot_e_Qe = 0
    m_dot_g_Qe = 0
    ER_Qe = 0
    Q_e_max = 0

    j = 0

    # for d_y in range(2000, 20000):
    for d_y in range(2):
        for d_noz in range(1000, 10000):

            d_y = d_y / 1000000
            d_noz = d_noz / 1000000

            # m_dot_e, m_dot_g = basic_interpolation(d_noz, p_g, T_g, T_sup, eta_noz, F_act, Ma_noz_ex,
            #                                        dummy_1, dummy_2, dummy_3, phi_p, Ma_pr_y,
            #                                        d_y, p_e, T_e, eta_s)
            ## Maybe Delete this

            h_3 = h_2  # "adiabatic expansion"

            m_dot_e = 10000 / (h_e - h_3)

            d_y, m_dot_g = basic_interpolation2(m_dot_e, d_noz, p_g, T_g, T_sup, eta_noz, F_act, Ma_noz_ex,
                                                dummy_1, dummy_2, dummy_3, phi_p, Ma_pr_y,
                                                d_y, p_e, T_e, eta_s)

            ## Until Here

            ER = m_dot_e / m_dot_g

            h_3 = h_2  # "adiabatic expansion"
            Q_e = m_dot_e * (h_e - h_3)  # "secondary mass flow rate"

            if ER > ER_max:

                ER_max = ER
                best_d_y = d_y
                best_d_noz = d_noz
                m_dot_e_ER = m_dot_e
                m_dot_g_ER = m_dot_g

                Q_e_ER = Q_e

                print(j)

            j += 1

            if Q_e > Q_e_max:

                ER_Qe = ER
                m_dot_e_Qe = m_dot_e
                m_dot_g_Qe = m_dot_g

                Q_e_max = Q_e

                # print(Q_e_max)

    d_y = best_d_y
    d_noz = best_d_noz
    m_dot_e = m_dot_e_ER
    m_dot_g = m_dot_g_ER

    ER = ER_max

    # ###
    #
    # m_dot_e = 0.0028965698751339444
    # m_dot_g = 5.475552849506265e-05
    # ER = m_dot_e / m_dot_g


    # Evaporator

    rho_e = PropsSI('D', 'T', T_e, 'Q', 1, Fluid)
    Q_vol_e = m_dot_e / rho_e

    ###

    v_m = (v_pr_y + ER * v_sec_y) * sqrt(phi_m) / (1 + ER)            # "eq 5 Yu"

    Mom_pr_y = m_dot_g * v_pr_y
    Mom_sec_y = m_dot_e * v_sec_y
    Mom_m = (m_dot_g + m_dot_e) * v_m

    cp_mix = cp_noz

    E_tot_pr_y = T_pr_y * cp_mix + 0.5 * (v_pr_y)**2

    E_tot_sec_y = T_sec_y * cp_mix + 0.5 * (v_sec_y)**2

    E_tot_m = (m_dot_g * E_tot_pr_y + m_dot_e * E_tot_sec_y) / (m_dot_g + m_dot_e)       # "energy balance in the mixing section, eq. 12"
    T_m = E_tot_m / (cp_mix + 0.5 * (v_m)**2)

    T_m_K = converttemp('C', 'K', T_m)
    Ma_m = v_m / sqrt(gamma * R_fluid * T_m_K)                              # "eq. 15"


    # Shock Wave

    p_sh = p_pr_y * (1 + dummy_1 * gamma * ((Ma_m ** 2) - 1))                            # "Pressure after the shock wave, eq. 16"
    Ma_sh = ((1 + dummy_2 * (Ma_m ** 2)) / (gamma * (Ma_m ** 2) - dummy_2)) ** 0.5         # "eq. 17 - this was not correct in previous versions"

    # Diffuser

    p_c = p_sh * (1 + eta_dif * dummy_2 * (Ma_sh ** 2)) ** (gamma / (gamma - 1))          # "ejector exit pressure, eq. 18"
    p_c_crit = PropsSI('P', 'T', T_c, 'Q', 1, Fluid)                                         # "pressure corresponding to condenser conditions"

    dummy_5 = p_c - p_c_crit
    dummy_5 = 10

    h_c = (m_dot_g * h_g + m_dot_e * h_e) / (m_dot_g + m_dot_e)
    T_eje_out = PropsSI('T', 'P', p_c, 'H', h_c, Fluid)


    # CONDENSER

    Q_c = (m_dot_g + m_dot_e) * (h_c - h_2)                             # "condenser power output"

    # PUMP

    rho_pump = PropsSI('D', 'T', T_c, 'Q', 0, Fluid)
    W_pump = (p_g - p_c) * m_dot_g / rho_pump                       # "required pump power"

    # GENERATOR

    Q_vol_g = m_dot_g / rho_g                                       # "volumetric flow rate from generator"
    h_1 = h_2 + W_pump / m_dot_g                                    # "generator inlet enthalpy"
    Q_g = m_dot_g * (h_g - h_1)                                     # "generator power"

    print(Q_g)
    print(Q_e)

    quit()

if __name__ == '__main__':
    main2()
