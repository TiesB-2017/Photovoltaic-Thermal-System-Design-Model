
# This code defines a library with a
# Created by: Ties Beijneveld.
# TU Delft
## Version: 1.0

###############################################################################
###########################   Imported modules   ##############################
import time
from numpy import append, array, arange, cos, pi, mean
from scipy.constants import Stefan_Boltzmann as sigma

###############################################################################
###########################   Functions definition  ###########################

###############################################################################
###########################   PVT model definition  ###########################


def T_sky(T_amb, RH=70, N=0):
    from math import exp
#    return 0.0552*(T_amb)**1.5 + 2.652*N
    return ((0.62+0.056*(6.11*(RH/100)*exp(17.63*T_amb/(T_amb+243)))**0.5)**0.25)*T_amb
#    return 0.7

def Wind_forced_convection(v):

    # Duffie and Beckman
    #    return 2.8 + 3*v

    # McAdams correlation
    if v < 5:
        return 5.7 + 3.8*v
    else:   # 5 <= v < 10
        return 6.47 + v**0.78


def radiative_heat_transfer_coefficient(T_amb, T_s, epsilon):
    
    # print("T_s", T_s)
    # print("T_amb", T_amb)
    # print("T_sky", T_sky(T_amb))
    
    return epsilon*sigma*((T_s + 273.15)**2 + (T_sky(T_amb) + 273.15)**2)*((T_s + 273.15) + (T_sky(T_amb) + 273.15))


def Thermal_diffusivity(k, density, c):

    return k/(density*c)


def radiative_heat_transfer_coefficient_two_plates(T_p1, epsilon_1, T_p2, epsilon_2):
    from scipy.constants import Stefan_Boltzmann as sigma

    return sigma*((T_p1 + 273.15)**2 + (T_p2 + 273.15)**2)*((T_p1 + 273.15) + (T_p2 + 273.15))*(epsilon_1**-1 + epsilon_2**-1 - 1)**-1


def conductive_heat_transfer_coefficient(L, k):

    return sum([L[i]/k[i] for i in range(len(k))])**-1


def Rayleigh(T_p1, T_p2, L_gap, beta=3.4e-3, visc=1.562e-5, k=0.024, density=1.184, c=1005, g=9.8):

    #    alpha = k/(density*c)

    return (g*beta*(abs(T_p1 - T_p2))*L_gap**3)/(visc*Thermal_diffusivity(k, density, c))


def Nusselt(T_p1, T_p2, L_gap, tilt=0):
    # https://watermark-silverchair-com.tudelft.idm.oclc.org/189_1.pdf?token=AQECAHi208BE49Ooan9kkhW_Ercy7Dm3ZL_9Cf3qfKAc485ysgAABG0wggRpBgkqhkiG9w0BBwagggRaMIIEVgIBADCCBE8GCSqGSIb3DQEHATAeBglghkgBZQMEAS4wEQQMmeIjkuilybvmOu6eAgEQgIIEIDb4_PENA7Dbd2jCBVnYUS4X2W1seHzJALTJjlFqGBx5NOs9rkDi9MnddnNFwaz0kftoRS7-16r-HX-jRJHnl_0ugd8v4E_LiwpyeS3MQQlipGe2JkW4SXOHHjRu8y2t-NWC9c1azO1WcGJn74lN1Ei7AYML6z7FwAsyr88yyeO8gUOr9wXzatVyBZtjT_lkdXPWhumDfkeFiPLr1O73iZWkhM2SGCISff4K5t09xsHVpROcPIXUPuj9X6rkhJZ7QenkX6XrpnRs91HEhrUELQrfhknDSL1y5wqBlO1WmP5x_mEyyKkFl_V5ycEvecaZKz_zRNpJGc7M8ajSkNiSJXW1aKnCXG3-vHSE6ErV7MzReqUO-_RcZedWuD11pHtgf-7gdy_rSo0BWWEfA_baqbZlYdeAvzenhhePnjgfA4Bema6lNm0fJvlVfbOr0nDablq1IenqMmbK6whUiWKgBdbzKMWLubjVWWTH3fB3jcRjhxEhjKgHoYcbfZkmbt8bXpgOl4YTZWphgD3goTsfgeN0-4hZxuzuZQIXQDY41pGjXRjUnN8XM3jP3jubPAz8ZPcRlPYIBfbq6FThKNMTirn80E77eEmeraKM57UqfMVVAMzRiUi5Vb-sQWWvI49xprdBk9L7jRL0Xw93MCMloKebfscnRBO7icrnyOhEJShNXp8NT-h4EiVbL-b069LYYdENkO-P14xAUQk3pU7eKkw8_JNeExhHFCRVJn_tVo8audf8hEQA0IYrjxPtxMWLl5QbPKI3f-A2_KlWXmTJDquaSSwTK-EjF4cm1V8_h_9GUMAZe9M9qQcuDFsV81Z8or8reRfpGIvPrJfeM9RqczCeqdVNnBfLECrc8ZCOWg5LeHTgB98KqXbXo7bKklFN1Iji9jZ8UuTDoI2eqSnnVPb9HcPkSXu8rDeNJTEXqxNMQGe9AHDxzJhAeycc5H8idMPVE1NFRTOrZEdlbG356bCW_kZKvwCPE3IrKFrE9v0lzJSO4V5f4uG98W1jqCJU3oHvlmGTnmdk-jQLZnvKLNZac7uwxIBHXWGCHPtqtGnFXe_f_WkJC0-W3FQjWwC3AN_mULEXnVVwMsIzClF58wwpUK-Tv0WSAzRQC94XhZJklXhbAoMew_RcaYHBOz0p5j8fxNWIWifLF-rBopwoddm2_sU-NXNFzT3tXBJZIeSixcgi_FRYVBY3N8eZ7uT8hfF-ckX54SfSb710WJ7ZKgKD7gPXNlSK8KEbymMU0ozWCaVMCK0Q0UW3GrgeGa7lEa6Z4fDa_-GQ13jcnVxMWYHt8M-fvBoQQqES9c4_p6BbSbgd8OK3wqvXCDGCTiqs3xyY5y7mj0HeBHlYHWMkGTlNFno1eAixBbP_K_fwqXj1YULF5Ff4h2o94FFXPjNagw
    # having 0° < tilt < 60°, and 0 < Ra < 105
    from math import sin, cos, radians

    return 1 + 1.44*(1 - 1708/(Rayleigh(T_p1, T_p2, L_gap)*cos(radians(tilt))))*(1 - 1708*(sin(radians(1.6*tilt))**1.6)/(Rayleigh(T_p1, T_p2, L_gap)*cos(radians(tilt))))*((Rayleigh(T_p1, T_p2, L_gap)*cos(radians(tilt))/5830)**(1/3)-1)


def Reynolds(m_dot, D_tube, density=1000, visc=1.562e-5):
    from math import pi

    return 4*m_dot/(density*pi*D_tube*visc)


def Prandlt(visc=0.8927e-6, k=0.6071, density=1000, c=4200):
    return visc/Thermal_diffusivity(k, density, c)


def fluid_conductive_heat_transfer(m_dot, k_f, D_tube):

    if m_dot == 0:
        return 2*k_f/D_tube
    elif Reynolds(m_dot, D_tube) < 2300:
        return 4.36*k_f/D_tube
    else:
        return 0.023*k_f*(Reynolds(m_dot, D_tube)**0.8)*(Prandlt()**0.4)/D_tube


def air_gap_convection(T_p1, T_p2, L_gap, k):
    return Nusselt(T_p1, T_p2, L_gap)*k/L_gap


def PVT_model(T_amb, G, v, T_f_in, T_glass_0, T_PV_0, T_a_0, T_f_0, n_STC, N_tubes, D_tube, n_HE=0.8, A_collector=1.77, A_PV=1.77, m_f_dot=0.029085, Len_collector=1, c_f=3800, dt=1):
    #    from math import pi

    A_glass = A_collector
#    A_gap = A_collector
    A_PV = A_PV
    A_a = A_collector
#    A_t_cross = D_tube*Len_collector*N_tubes
#    A_t_surf = 0.5*pi*D_tube*Len_collector*N_tubes
    A_t = 1.48  # A_t_cross + A_t_surf

    rho_glass = 2200
    rho_PV = 2330
    rho_a = 2699
    rho_f = 1050

    L_glass = 0.0032
    L_PV = 0.0004
    L_a = 0.001
    L_gap = 0.02
    L_ins = 0.04

    L_PV_glass = 0.003
    L_PV_EVA = 0.0005
    L_PV_tedlar = 0.0001

    k_PV_glass = 1.8
    k_PV_EVA = 0.35
    k_PV_tedlar = 0.2
    k_air = 0.024
    k_f = 0.6071
    k_ins = 0.035

    if m_f_dot == 0:
        m_f_dot = rho_f*2.77e-5    # kg/s

    c_glass = 670
    c_pv = 900
    c_a = 800
#    c_f = 4200

    alpha_glass = 0.1
    alpha_PV = 0.9
    tau_glass = 0.93**1

    epsilon_glass = 0.9
    epsilon_PV = 0.96

    m_glass = rho_glass*A_glass*L_glass
    m_PV = rho_PV*A_PV*L_PV
    m_a = rho_a*A_a*L_a
    m_f = 0.65*rho_f/1000  # N_tubes*rho_f*(0.125*pi*D_tube**2)*Len_collector

    h_glass_conv = Wind_forced_convection(v)
    h_glass_r = radiative_heat_transfer_coefficient(
        T_sky(T_amb), T_glass_0, epsilon_glass)
    h_gap = (conductive_heat_transfer_coefficient([L_PV_glass, L_PV_EVA], [
             k_PV_glass, k_PV_EVA])**-1 + air_gap_convection(T_glass_0, T_PV_0, L_gap, k_air)**-1)**-1
    h_glassPV_r = radiative_heat_transfer_coefficient_two_plates(
        T_glass_0, epsilon_glass, T_PV_0, epsilon_PV)
    h_PVa_cond = conductive_heat_transfer_coefficient(
        [L_PV_EVA, L_PV_tedlar], [k_PV_EVA, k_PV_tedlar])
    h_af = fluid_conductive_heat_transfer(m_f_dot, k_f, D_tube)
    h_a_cond = conductive_heat_transfer_coefficient([L_ins], [k_ins])

    T_glass = T_glass_0 + dt*A_glass/(m_glass*c_glass)*(h_glass_conv*(T_amb - T_glass_0) + h_glass_r*(
        T_sky(T_amb) - T_glass_0) + h_gap*(T_PV_0 - T_glass_0) + h_glassPV_r*(T_PV_0 - T_glass_0) + alpha_glass*G)
    T_PV = T_PV_0 + (A_PV*dt/(m_PV*c_pv))*(h_gap*(T_glass_0 - T_PV_0) + h_glassPV_r*(T_glass_0 - T_PV_0) +
                                           h_PVa_cond*(T_a_0 - T_PV_0) + alpha_PV*tau_glass*G*(1 - PV_efficiency(T_PV_0, n_STC)))
    T_a = T_a_0 + (dt/(m_a*c_a))*(A_a*h_PVa_cond*(T_PV_0 - T_a_0) +
                                  h_af*A_t*(T_f_0 - T_a_0) + h_a_cond*A_a*(T_amb - T_a_0))
    T_f = T_f_0 + (dt/(m_f*c_f))*(h_af*A_t*(T_a_0 - T_f_0) +
                                  2*m_f_dot*c_f*(T_f_in - T_f_0))


    Q_PVT_dot = n_HE*2*m_f_dot*c_f*(T_f - T_f_in)

    return [Q_PVT_dot, T_glass, T_PV, T_a, T_f]

# def Qdot_PVT(PVT_active, TESS_charge, T_amb, G, v, T_in_Network, T_glass_0, T_PV_0, T_a_0, T_f_0, T_tank_PVT, T0_ATES, n_modules, A_collector=1.77, A_PV=1.77, n_STC=0.184, N_tubes=0, D_tube=0.009, n_HE=0.8, m_f_dot=0, Len_collector=1, t_end=int(0.25*3600), recirculation=True):
def Qdot_PVT(PVT_active, TESS_charge, T_amb, G, v, T_in_Network, T_glass_0, T_PV_0, T_a_0, T_f_0, T_tank_PVT, T0_ATES, m_f_dot, n_STC = 0.184, N_tubes = 0, D_tube = 0.009, n_HE = 0.8, A_collector = 1.77, A_PV = 1.77, Len_collector = 1, t_end = int(0.25*3600), n_modules = 1, recirculation = True):
    
    #    T_tank_PVT_registry = []
    #    Q_PVT_dot_registry = []
    #    Qdot_PVT2Network_registry = []
    #    T_network_registry = []
    #    import matplotlib.pyplot as plt

    if recirculation:

        for step in range(t_end):
            
            [Q_PVT_dot, T_glass_0, T_PV_0, T_a_0, T_f_0] = PVT_model(
                T_amb, G, v, T_tank_PVT, T_glass_0, T_PV_0, T_a_0, T_f_0, n_STC, n_modules*N_tubes, D_tube, n_HE, n_modules*A_collector, n_modules*A_PV, m_f_dot, n_modules*Len_collector)
            [T_out_Network, T_tank_PVT, Qdot_PVT2Network, Qdot_PVT2TESS] = Update_PVT_tank(
                T_fluid_out(T_tank_PVT, T_f_0), T_in_Network, T_tank_PVT, T0_ATES, PVT_active, TESS_charge)
            

        return [Q_PVT_dot, [T_glass_0, T_PV_0, T_a_0, T_f_0], T_tank_PVT, T_out_Network, Qdot_PVT2Network, Qdot_PVT2TESS]
    else:
        
        return False


#    return [Q_PVT_dot, T_glass_0, T_PV_0, T_a_0, T_f_0]

#    return Q_PVT_dot


def T_fluid_out(T_in, T_f):
    return (T_f - 0.5*T_in)/0.5


def PV_efficiency(T_PV, n_STC, beta_PV=-0.00365, T_ref=25):
    return n_STC*(1 - beta_PV*(T_PV - T_ref))


def PV_out(T_PV, G, A_PV=1.77, n_STC=0.2031, kW=True):
    if kW:
        return PV_efficiency(T_PV, n_STC)*G*A_PV

    else:
        return PV_efficiency(T_PV, n_STC)*G*A_PV


def Update_PVT_tank(T_in_PVT, T_in_Network, T0, T0_ATES, PVT_active, TESS_charge=False, mdot=10, m=2000, c=4200, efficiency_TESS=0.8, dt=1):
    # Assuming mass conservation, using energy balance equatiion.

    # Thermal power from the PVT.
    [T_out, Qdot_PVT] = Heat_exchanger(
        T_in_PVT, T0, mdot, efficiency=1, c1=3800, mode="TESS")

    # Thermal power supplied to the thermal network.
    if PVT_active == True:
        [T_out_Network, T_in_tank, Qdot_Network, Qdot_tank] = Heat_exchanger(
            T_in_Network, T0, 50, 10, mode="L2L")
    else:
        T_out_Network = T_in_Network
        Qdot_Network = 0
        Qdot_tank = 0

    if TESS_charge == True:
        [T_out_TESS, Qdot_TESS] = Heat_exchanger(
            T0, T0_ATES, efficiency=efficiency_TESS, mode="TESS")

    else:
        Qdot_TESS = 0

    T_tank = T0 + (dt/(m*c))*(Qdot_PVT - Qdot_tank)# - Qdot_TESS/efficiency_TESS)
    
    
    return [T_out_Network, T_tank, Qdot_Network, Qdot_TESS]


def Heat_exchanger(T1_in, T2_in, mdot1=50, mdot2=10, c1=4200, c2=4200, efficiency=0.8, mode="L2L"):
    # Assuming a small difference between T1_in and T2_in, a large contact surface and T2_in > T1_in.

    if mode == "L2L":
        #    T2_out = T1_in
        #    Qdot2 = mdot2*c2*(T2_out - T2_in)
        #    Qdot1 = -efficiency*Qdot2
        #    dT = Qdot1/(mdot1*c1)
        #    T1_out = T1_in + dT
        Qdot1 = efficiency*mdot2*c2*(T2_in - T1_in)
        T1_out = T1_in + Qdot1/(mdot1*c1)

        # T1_out, T2_out, Qdot1, Qdot2
        return [T1_out, T1_in, Qdot1, Qdot1/efficiency]

    elif mode == "TESS":
        #    T1_out = T2_in
        #    Qdot2 = efficiency*mdot1*c1*(T1_in - T1_out)
        Qdot2 = efficiency*mdot1*c1*(T1_in - T2_in)

        return [T2_in, Qdot2]  # T1_out, Qdot2


#    else:
#        # Put something in here


###############################################################################
########################   Simple SC model definition  ########################

def Qdot_SolarCollector(active, G, A=6, SC_eff=0.45, dt=1*3600):

    if active:
        return A*SC_eff*G/dt
    else:
        return 0




def update_ATES(active, T_in, T_0, r, T_soil, Qdot_SC=0, Qdot_HP=0, Tmax=95 + 273, Tmin=5 + 273, mdot=50, T_network=15 + 273, efficiency=0.8, L_ATE=10, c_w=4200, c_r=830, n=0.2, dt=0.25*3600, rho_f=1050, rho_r=1602):  # , Qdot_SD = 100
    import math
    Qdot_soil = Qdot_ATES2Soil(T_0 - 273, r)

    
    m_w = n*rho_f*math.pi*L_ATE*r**2
    m_r = (1-n)*rho_r*math.pi*L_ATE*r**2

    if T_0 <= Tmin:         # TESS discharged, only charge
        T_new = T_0 + (mdot*c_w*T_in - mdot*c_w*T_0 + 
                       Qdot_soil)*dt/(m_w*c_w + m_r*c_r)


    elif T_0 <= Tmax:       # TESS available for charge and discharge

        T_new = T_0 + (mdot*c_w*T_in - mdot*c_w*T_0 + 
                       Qdot_soil)*dt/(m_w*c_w + m_r*c_r)

    else:                   # TESS fully charged, only discharge

        T_new = T_0 + (Qdot_soil)*dt/(m_w*c_w + m_r*c_r)
        
    Q_ATES = mdot*c_w*(T_new - T_0)    
    
    return [T_new, Qdot_soil, Q_ATES]


def Qdot_ATES2Soil(T_ATES, r, depth=150, L_ATE=10, k_soil=1.19, T_soil=11):
    
    A_top = A_bottom = pi*r**2
    A_side = 2*pi*L_ATE
    
    # T_soil=mean_effective_T(depth)
    # T_soil=11

    h_ATE = conductive_heat_transfer_coefficient([L_ATE], [k_soil])
    
    Qdot_top = h_ATE*A_top*(T_soil-T_ATES)
    Qdot_bottom = h_ATE*A_bottom*(T_soil-T_ATES)
    Qdot_side = h_ATE*A_side*(T_soil-T_ATES)
    
    return Qdot_top + Qdot_bottom + Qdot_side

def P_ATES_pump(eta_ATEP=0.8, mdot=50, rho_f=1050, g=9.81, depth=150, L=150, diameter=13, v=0.001):
    import math
    Vdot = mdot/rho_f
    
    A = math.pi * (diameter / 2) ** 2
    V = Vdot / A
    Re = (rho_f * diameter * V) / v
    if Re > 4000:
        # Turbulent flow: Calculate friction factor using Blasius correlation
        f = 0.3164 * Re ** -0.25
    else:
        # Laminar flow: Calculate friction factor
        f = 64 / Re
        
    P_l = f * (L / diameter) * (rho_f * V ** 2 / 2)
    
    P_ATEP = (Vdot*(rho_f*g*depth + L*P_l))/eta_ATEP
    
    return P_ATEP

###############################################################################
########################   Simple HP model definition  ########################

def HP_Power_0(active, P_in=2.7*1000, COP=4.1):
    if active:
        return [P_in/1000, P_in*COP]
    else:
        return [0, 0]

###############################################################################
#########################   Nikos HP model definition  ########################


# global COP_registry
# COP_registry = []
# COP_air_registry = []


def HP_Power(active, T_HPin, T_ret, m_dot, HE_eff=0.8, T_sup=273 + 53, c=4200):
    from math import e

    if active:

        COP = 7.90471*e**(-0.024*(T_ret - T_HPin))

        if COP > 3.6:
            COP = 3.6

        
        return [HE_eff*m_dot*c*(T_sup - T_ret)/(COP), HE_eff*m_dot*c*(T_sup - T_ret), (HE_eff*m_dot*c*(T_sup - T_ret))*(1-1/COP), COP]
    else:
        return [0, 0, 0]
    
def HP_extra(T_amb, T_HPin, Power):
    from math import e
    
    COP = 7.90471*e**(-0.024*(T_HPin - T_amb))
    if COP > 3.6:
        COP = 3.6
    Q_dot = Power*COP

    # COP_air_registry.append(COP)
    return [Q_dot, COP]
    

def heat_loss(U, Area, T_in, T_out):
    return U * Area * (T_out - T_in)


def new_house_Temperature(T_0, Qdot_Losses, mc_T, Qdot_HP, dt=0.25*3600):
    T_new = T_0 + dt*(Qdot_Losses + Qdot_HP)/(mc_T)

    return T_new


def R_convective(m_dot, k_f=0.6071, D_tube=13e-3):

    return 1/(pi*D_tube*fluid_conductive_heat_transfer(m_dot, k_f, D_tube))


def R_conductive(D_in, D_out, k_tube):
    from math import log

    return log(D_out/D_in)/(2*pi*k_tube)


def Celcius2Farenheit(T_C):
    return T_C*9/5 + 32


def Infiltration_Airflow(T_in, T_out, U=1, A_exposed=119.6, C_stack=0.015, C_wind=0.0012):
    from math import sqrt

    U *= 2.23694    # m/s to mph
    A_exposed *= 10.7639    # m3 to ft3

    dT = Celcius2Farenheit(T_in) - Celcius2Farenheit(T_out)          # °C to °F

    A_unit = 0.01   #

    if dT > 0:
        return A_exposed*A_unit*sqrt(C_stack*abs(dT) + C_wind*U**2)*0.47194745/1000

    else:
        return 0


def Ventilation_Airflow(A_conditioned=120, N_bedrooms=1):
    A_conditioned *= 10.7639    # m3 to ft3
    return (0.03*A_conditioned + 7.5*(1+N_bedrooms))*0.47194745/1000


def Infiltration_Ventilation_Heat_loss(T_in, T_out, U=1, A_exposed=119.6, C_stack=0.015, C_wind=0.0012, A_conditioned=120, N_bedrooms=1, c_air=1012, density_air=1.293):

    return -c_air*density_air*(Infiltration_Airflow(T_in, T_out, U, A_exposed, C_stack, C_wind) + Ventilation_Airflow(A_conditioned, N_bedrooms))*(T_in - T_out)


#Pipe losses
  

def pipe_resistance(m_dot_pipe, diameters=[13, 15, 40], k_f=0.6071, k_tubes=[398, 0.038], k_soil=1.19, H=100):
    
        R_total = R_convective(m_dot_pipe, k_f, diameters[0])
        
        for i in range(len(k_tubes)):
            
                R_total += R_conductive(diameters[i], diameters[i+1], k_tubes[i])
        
        R_total += R_conductive(diameters[2], 2*H, k_soil)
            
        return R_total


def thermal_radiation(T_amb, T_s):
    from scipy.constants import Stefan_Boltzmann as sigma
    return sigma*((T_s + 273.15)**4 - (T_sky(T_amb) + 273.15)**4)

def mean_effective_T(depth):
     
    return 0.0318*depth + 8.01768

def T_surface(T_amb, G, epsilon, alpha_0, k_soil, depth, v, T0_soil):
      
    #T_d = mean_effective_T(depth)
    
    h_conv = Wind_forced_convection(v)
    
    h_r = radiative_heat_transfer_coefficient(T_sky(T_amb), T0_soil, epsilon)
    
    h = h_conv + h_r
    
    delta_R = thermal_radiation(T_amb, T0_soil)
    
    T_e = T_amb + alpha_0*G/h - epsilon*delta_R/h
    
    b = k_soil/(depth*h)
    
    T_new = (b*mean_effective_T(depth) + T_e)/(1 + b)
    


    return T_new



def T_pipe_out(R_total, T_s, T_pipe_in, L, m_dot, c_f=4200):
    from math import exp
    
    T_out = T_s - (T_s - T_pipe_in)*exp(-L/(m_dot*c_f*R_total))
    
    return T_out
    

def Qdot_pipe_ground(T_out, T_pipe_in, m_dot_pipe, L, T_amb, G, epsilon, alpha_0, k_soil, depth, v, T0_soil, c_f=4200):
    
    T_out = T_pipe_out(pipe_resistance(m_dot_pipe), T_surface(T_amb, G, epsilon, alpha_0, k_soil, depth, v, T0_soil), T_pipe_in, L, m_dot_pipe)  

    T_ground = T_surface(T_amb, G, epsilon, alpha_0, k_soil, depth, v, T0_soil)
    
    Qdot_pipe = m_dot_pipe*c_f*(T_out - T_pipe_in)
    

                                                                                                                                           
    return [Qdot_pipe, T_out, T_ground]
  

def House_Thermal_Losses(T_0_in, T_amb, U=1, A_exposed=119.6, C_stack=0.015, C_wind=0.0012, A_conditioned=120, N_bedrooms=1, c_air=1012, density_air=1.293):

    #############################   Paremeters ################################

    # Convective heat transfer coefficients [W/m^2K]
    h_air_wall = 0.9  # 24       # Indoor air -> walls, scaled to R-value of a C-label house
    h_wall_atm = 0.9  # 34       # Walls -> atmosphere, scaled to R-value of a C-label house
    h_air_window = 25            # Indoor air -> windows
    h_window_atm = 32            # Windows -> atmosphere
    h_air_roof = 12            # Indoor air -> roof
    h_roof_atm = 38            # Roof -> atmosphere

    # House

    # Air
    c_air = 1005.4        # Specific heat of air at 273 K [J/kgK]
    airDensity = 1.025         # Densiity of air at 293 K [kg/m^3]
    kAir = 0.0257        # Thermal conductivity of air at 293 K [W/mK]

    # Windows (glass)
    n1_window = 3            # Number of windows in room 1
    n2_window = 2            # Number of windows in room 2
    n3_window = 2            # Number of windows in room 3
    n4_window = 1            # Number of windows in room 4
    htWindows = 1            # Height of windows [m]
    widWindows = 1            # Width of windows [m]
    windows_area = (n1_window + n2_window + n3_window +
                    n4_window) * htWindows * widWindows
    LWindow = 0.004        # Thickness of a single window pane [m]
    # Thickness of the cavity between the double glass window [m]
    LCavity = 0.014
    windowDensity = 2500         # Density of glass [kg/m^3]
    c_window = 840          # Specific heat of glass [J/kgK]
    kWindow = 0.8          # Thermal conductivity of glass [W/mK]
    U_windows = ((1/h_air_window) + (LWindow/kWindow) +
                 (LCavity/kAir) + (LWindow/kWindow) + (1/h_window_atm))**-1
    m_windows = windowDensity * windows_area * LWindow

    # Walls (concrete)
    lenHouse = 15             # House length [m]
    widHouse = 8              # House width [m]
    htHouse = 2.6            # House height [m]
    LWall = 0.25           # Wall thickness [m]
    wallDensity = 2400           # Density [kg/m^3]
    c_wall = 750            # Specific heat [J/kgK]
    kWall = 0.14           # Thermal conductivity [W/mK]
    walls_area = 2*(lenHouse + widHouse) * htHouse - windows_area
    U_wall = ((1/h_air_wall) + (LWall / kWall) + (1/h_wall_atm))**-1
    m_walls = wallDensity * walls_area * LWall

    # Roof (glass fiber)
    pitRoof = 40/180/pi      # Roof pitch (40 deg)
    LRoof = 0.2            # Roof thickness [m]
    roofDensity = 2440           # Density of glass fiber [kg/m^3]
    c_roof = 835            # Specific heat of glass fiber [J/kgK]
    kRoof = 0.04           # Thermal conductivity of glass fiber [W/mK]
    roof_Area = 2 * (widHouse/(2*cos(pitRoof))*lenHouse)
    U_roof = ((1/h_air_roof) + (LRoof/kRoof) + (1/h_roof_atm))**-1
    m_roof = roofDensity * roof_Area * LRoof

    m_air = airDensity * lenHouse * widHouse * htHouse

    mc_T = m_air*c_air + m_roof*c_roof + m_windows*c_window + m_walls*c_wall

    ############################   Calculations ###############################
    ################### Thermal carrier ######################

    # Thermal losses
    # Roof
    Qdot_roof = heat_loss(U_roof, roof_Area, T_0_in, T_amb)

    # Windows
    Qdot_windows = heat_loss(U_windows, windows_area, T_0_in, T_amb)

    # Walls
    Qdot_wall = heat_loss(U_wall, walls_area, T_0_in, T_amb)

    # Infiltration and ventilation
    Qdot_iv = Infiltration_Ventilation_Heat_loss(
        T_0_in - 273, T_amb - 273, U, (walls_area + windows_area), C_stack, C_wind, roof_Area, N_bedrooms, c_air, airDensity)
    
    return [Qdot_roof + Qdot_windows + Qdot_wall + Qdot_iv, mc_T]



def House_Thermal_Losses_1(T_0_in, T_amb, U=1, A_exposed=119.6, C_stack=0.015, C_wind=0.0012, A_conditioned=120, N_bedrooms=1, c_air=1012, density_air=1.293):

    #############################   Paremeters ################################

    # Convective heat transfer coefficients [W/m^2K]
    h_air_wall = 0.9  # 24       # Indoor air -> walls, scaled to R-value of a C-label house
    h_wall_atm = 0.9  # 34       # Walls -> atmosphere, scaled to R-value of a C-label house
    h_air_window = 25            # Indoor air -> windows
    h_window_atm = 32            # Windows -> atmosphere
    h_air_roof = 12            # Indoor air -> roof
    h_roof_atm = 38            # Roof -> atmosphere

    # House

    # Air
    c_air = 1005.4        # Specific heat of air at 273 K [J/kgK]
    airDensity = 1.025         # Densiity of air at 293 K [kg/m^3]
    kAir = 0.0257        # Thermal conductivity of air at 293 K [W/mK]

    # Windows (glass)
    # n1_window = 3            # Number of windows in room 1
    # n2_window = 2            # Number of windows in room 2
    # n3_window = 2            # Number of windows in room 3
    # n4_window = 1            # Number of windows in room 4
    # htWindows = 1            # Height of windows [m]
    # widWindows = 1            # Width of windows [m]
    windows_area = 213
    # windows_area = (n1_window + n2_window + n3_window +
    #                 n4_window) * htWindows * widWindows
    LWindow = 0.004        # Thickness of a single window pane [m]
    # Thickness of the cavity between the double glass window [m]
    LCavity = 0.014
    windowDensity = 2500         # Density of glass [kg/m^3]
    c_window = 840          # Specific heat of glass [J/kgK]
    kWindow = 0.8          # Thermal conductivity of glass [W/mK]
    U_windows = ((1/h_air_window) + (LWindow/kWindow) +
                 (LCavity/kAir) + (LWindow/kWindow) + (1/h_window_atm))**-1
    m_windows = windowDensity * windows_area * LWindow

    # Walls (concrete)
    lenHouse = 35             # House length [m]
    widHouse = 25             # House width [m]
    # lenHouse = 15             # House length [m]
    # widHouse = 8             # House width [m]
    htHouse = 6            # House height [m]
    LWall = 0.25           # Wall thickness [m]
    wallDensity = 2400           # Density [kg/m^3]
    c_wall = 750            # Specific heat [J/kgK]
    kWall = 0.14           # Thermal conductivity [W/mK]
    walls_area = 2*(lenHouse + widHouse) * htHouse - windows_area
    U_wall = ((1/h_air_wall) + (LWall / kWall) + (1/h_wall_atm))**-1
    m_walls = wallDensity * walls_area * LWall

    # Roof (glass fiber)
    # pitRoof = 40/180/pi      # Roof pitch (40 deg)
    LRoof = 0.2            # Roof thickness [m]
    roofDensity = 2440           # Density of glass fiber [kg/m^3]
    c_roof = 835            # Specific heat of glass fiber [J/kgK]
    kRoof = 0.04           # Thermal conductivity of glass fiber [W/mK]
    # roof_Area = 2 * (widHouse/(2*cos(pitRoof))*lenHouse)
    roof_Area = 874
    U_roof = ((1/h_air_roof) + (LRoof/kRoof) + (1/h_roof_atm))**-1
    m_roof = roofDensity * roof_Area * LRoof

    m_air = airDensity * lenHouse * widHouse * htHouse

    mc_T = m_air*c_air + m_roof*c_roof + m_windows*c_window + m_walls*c_wall

    ############################   Calculations ###############################
    ################### Thermal carrier ######################

    # Thermal losses
    # Roof
    Qdot_roof = heat_loss(U_roof, roof_Area, T_0_in, T_amb)

    # Windows
    Qdot_windows = heat_loss(U_windows, windows_area, T_0_in, T_amb)

    # Walls
    Qdot_wall = heat_loss(U_wall, walls_area, T_0_in, T_amb)

    # Infiltration and ventilation
    Qdot_iv = Infiltration_Ventilation_Heat_loss(
        T_0_in - 273, T_amb - 273, U, (walls_area + windows_area), C_stack, C_wind, roof_Area, N_bedrooms, c_air, airDensity)

    
    return [Qdot_roof + Qdot_windows + Qdot_wall + Qdot_iv, mc_T]

def House_Thermal_Losses_2(T_0_in, T_amb, U=1, A_exposed=119.6, C_stack=0.015, C_wind=0.0012, A_conditioned=120, N_bedrooms=1, c_air=1012, density_air=1.293):

    #############################   Paremeters ################################

    # Convective heat transfer coefficients [W/m^2K]
    h_air_wall = 0.9  # 24       # Indoor air -> walls, scaled to R-value of a C-label house
    h_wall_atm = 0.9  # 34       # Walls -> atmosphere, scaled to R-value of a C-label house
    h_air_window = 25            # Indoor air -> windows
    h_window_atm = 32            # Windows -> atmosphere
    h_air_roof = 12            # Indoor air -> roof
    h_roof_atm = 38            # Roof -> atmosphere

    # House

    # Air
    c_air = 1005.4        # Specific heat of air at 273 K [J/kgK]
    airDensity = 1.025         # Densiity of air at 293 K [kg/m^3]
    kAir = 0.0257        # Thermal conductivity of air at 293 K [W/mK]

    # Windows (glass)
    # n1_window = 6            # Number of windows in room 1
    # n2_window = 4            # Number of windows in room 2
    # n3_window = 4            # Number of windows in room 3
    # n4_window = 2            # Number of windows in room 4
    # htWindows = 2            # Height of windows [m]
    # widWindows = 2            # Width of windows [m]
    # windows_area = (n1_window + n2_window + n3_window +
    #                 n4_window) * htWindows * widWindows
    windows_area = 400
    LWindow = 0.004        # Thickness of a single window pane [m]
    # Thickness of the cavity between the double glass window [m]
    LCavity = 0.014
    windowDensity = 2500         # Density of glass [kg/m^3]
    c_window = 840          # Specific heat of glass [J/kgK]
    kWindow = 0.8          # Thermal conductivity of glass [W/mK]
    U_windows = ((1/h_air_window) + (LWindow/kWindow) +
                 (LCavity/kAir) + (LWindow/kWindow) + (1/h_window_atm))**-1
    m_windows = windowDensity * windows_area * LWindow

    # Walls (concrete)
    lenHouse = 50             # House length [m]
    widHouse = 30              # House width [m]
    htHouse = 7            # House height [m]
    LWall = 0.25           # Wall thickness [m]
    wallDensity = 2400           # Density [kg/m^3]
    c_wall = 750            # Specific heat [J/kgK]
    kWall = 0.14           # Thermal conductivity [W/mK]
    walls_area = 2*(lenHouse + widHouse) * htHouse - windows_area
    U_wall = ((1/h_air_wall) + (LWall / kWall) + (1/h_wall_atm))**-1
    m_walls = wallDensity * walls_area * LWall

    # Roof (glass fiber)
    # pitRoof = 40/180/pi      # Roof pitch (40 deg)
    LRoof = 0.2            # Roof thickness [m]
    roofDensity = 2440           # Density of glass fiber [kg/m^3]
    c_roof = 835            # Specific heat of glass fiber [J/kgK]
    kRoof = 0.04           # Thermal conductivity of glass fiber [W/mK]
    # roof_Area = 2 * (widHouse/(2*cos(pitRoof))*lenHouse)
    roof_Area = 1504
    U_roof = ((1/h_air_roof) + (LRoof/kRoof) + (1/h_roof_atm))**-1
    m_roof = roofDensity * roof_Area * LRoof

    m_air = airDensity * lenHouse * widHouse * htHouse

    mc_T = m_air*c_air + m_roof*c_roof + m_windows*c_window + m_walls*c_wall

    ############################   Calculations ###############################
    ################### Thermal carrier ######################

    # Thermal losses
    # Roof
    Qdot_roof = heat_loss(U_roof, roof_Area, T_0_in, T_amb)

    # Windows
    Qdot_windows = heat_loss(U_windows, windows_area, T_0_in, T_amb)

    # Walls
    Qdot_wall = heat_loss(U_wall, walls_area, T_0_in, T_amb)

    # Infiltration and ventilation
    Qdot_iv = Infiltration_Ventilation_Heat_loss(
        T_0_in - 273, T_amb - 273, U, (walls_area + windows_area), C_stack, C_wind, roof_Area, N_bedrooms, c_air, airDensity)

    return [Qdot_roof + Qdot_windows + Qdot_wall + Qdot_iv, mc_T]

def House_Thermal_Losses_3(T_0_in, T_amb, U=1, A_exposed=119.6, C_stack=0.015, C_wind=0.0012, A_conditioned=120, N_bedrooms=1, c_air=1012, density_air=1.293):

    #############################   Paremeters ################################

    # Convective heat transfer coefficients [W/m^2K]
    h_air_wall = 0.9  # 24       # Indoor air -> walls, scaled to R-value of a C-label house
    h_wall_atm = 0.9  # 34       # Walls -> atmosphere, scaled to R-value of a C-label house
    h_air_window = 25            # Indoor air -> windows
    h_window_atm = 32            # Windows -> atmosphere
    h_air_roof = 12            # Indoor air -> roof
    h_roof_atm = 38            # Roof -> atmosphere

    # House

    # Air
    c_air = 1005.4        # Specific heat of air at 273 K [J/kgK]
    airDensity = 1.025         # Densiity of air at 293 K [kg/m^3]
    kAir = 0.0257        # Thermal conductivity of air at 293 K [W/mK]

    # Windows (glass)
    # n1_window = 6            # Number of windows in room 1
    # n2_window = 4            # Number of windows in room 2
    # n3_window = 4            # Number of windows in room 3
    # n4_window = 2            # Number of windows in room 4
    # htWindows = 2            # Height of windows [m]
    # widWindows = 2            # Width of windows [m]
    # windows_area = (n1_window + n2_window + n3_window +
    #                 n4_window) * htWindows * widWindows
    windows_area = 647
    LWindow = 0.004        # Thickness of a single window pane [m]
    # Thickness of the cavity between the double glass window [m]
    LCavity = 0.014
    windowDensity = 2500         # Density of glass [kg/m^3]
    c_window = 840          # Specific heat of glass [J/kgK]
    kWindow = 0.8          # Thermal conductivity of glass [W/mK]
    U_windows = ((1/h_air_window) + (LWindow/kWindow) +
                 (LCavity/kAir) + (LWindow/kWindow) + (1/h_window_atm))**-1
    m_windows = windowDensity * windows_area * LWindow

    # Walls (concrete)
    lenHouse = 70             # House length [m]
    widHouse = 40              # House width [m]
    htHouse = 8            # House height [m]
    LWall = 0.25           # Wall thickness [m]
    wallDensity = 2400           # Density [kg/m^3]
    c_wall = 750            # Specific heat [J/kgK]
    kWall = 0.14           # Thermal conductivity [W/mK]
    walls_area = 2*(lenHouse + widHouse) * htHouse - windows_area
    U_wall = ((1/h_air_wall) + (LWall / kWall) + (1/h_wall_atm))**-1
    m_walls = wallDensity * walls_area * LWall

    # Roof (glass fiber)
    # pitRoof = 40/180/pi      # Roof pitch (40 deg)
    LRoof = 0.2            # Roof thickness [m]
    roofDensity = 2440           # Density of glass fiber [kg/m^3]
    c_roof = 835            # Specific heat of glass fiber [J/kgK]
    kRoof = 0.04           # Thermal conductivity of glass fiber [W/mK]
    # roof_Area = 2 * (widHouse/(2*cos(pitRoof))*lenHouse)
    roof_Area = 3130
    U_roof = ((1/h_air_roof) + (LRoof/kRoof) + (1/h_roof_atm))**-1
    m_roof = roofDensity * roof_Area * LRoof

    m_air = airDensity * lenHouse * widHouse * htHouse

    mc_T = m_air*c_air + m_roof*c_roof + m_windows*c_window + m_walls*c_wall

    ############################   Calculations ###############################
    ################### Thermal carrier ######################

    # Thermal losses
    # Roof
    Qdot_roof = heat_loss(U_roof, roof_Area, T_0_in, T_amb)

    # Windows
    Qdot_windows = heat_loss(U_windows, windows_area, T_0_in, T_amb)

    # Walls
    Qdot_wall = heat_loss(U_wall, walls_area, T_0_in, T_amb)

    # Infiltration and ventilation
    Qdot_iv = Infiltration_Ventilation_Heat_loss(
        T_0_in - 273, T_amb - 273, U, (walls_area + windows_area), C_stack, C_wind, roof_Area, N_bedrooms, c_air, airDensity)

    return [Qdot_roof + Qdot_windows + Qdot_wall + Qdot_iv, mc_T]


#global enable_HP_2_TESS
#enable_HP_2_TESS = [True]

def Thermal_Electrical_model(Thermal_Components, Q_storage_switch, T_in_0, T5, T6, T7, T_ATES_0, T_tank_PVT, T_PVT_layers, TESS_min_op_T, TESS_max_op_T, SoC_0_BESS, T_amb, T_set, G, Tsoil, P_Load_new, t_final, T_s_0, n_modules, radius_ATES, PowerPrices_15min_list, start_time, Intensity_array, P_BESS_max=1.28, P_Grid_max=0, Capacity_BESS=3.36, PVT_2_TESS=False, HP_2_TESS=False, Network_losses=False, m=4000, T_sup=50 + 273, T_TESS_max=273 + 95, m_dot=10, m_dot_pipe=50, c_f=4200, A_PVT=1.77, dt=60*15, L_1 =100, epsilon = 0.9, alpha_0 = 0.1, k_soil = 1.19, depth = 6, v = 1, HE_eff = 0.8):

    TESS_active = True

    T_in =  [[T_in_0, T_in_0], [T_in_0, T_in_0], [T_in_0, T_in_0], [T_in_0, T_in_0], [T_in_0, T_in_0], [T_in_0, T_in_0], [T_in_0, T_in_0], [T_in_0, T_in_0], [T_in_0, T_in_0], [T_in_0, T_in_0], [T_in_0, T_in_0], [T_in_0, T_in_0], [T_in_0, T_in_0], [T_in_0, T_in_0], [T_in_0, T_in_0]]
    T_ATES = [T_ATES_0]
    T_s = [T_s_0]

    fluid_registry = [[], [], [], [], [T5 - 273]]
    T_registry = [[[], [], [], [], [], [], [], [], [], [], [], [], [], [], []], [[], [], [], [], [], [], [], [], [], [], [], [], [], [], []], [[]], [[]], [[]], [[]], [[], [], [], [], [], [], [], [], [], [], [], [], [], [], []], [[]]]
    # T_registry = [[], [], [], [], [], [], []]
    # T0_average = []
    # T6_average = []
    # Tin_average =[]
    T_tank_PVT_registry = []
    Qdot_PVT_registry = []
    Qdot_SC_TESS = []
    Qdot_PVT_Network = []
    Qdot_pipe1 = []
    P_PVT = []
    A_PVT = A_PVT*n_modules

#    Qdot_TESS = []
#    Qdot_TESS_SD = []
    # enable_HP_2_TESS = False
    
    street = 15
    P_HP = [[], [], [], [], [], [], [], [], [], [], [], [], [], [], []]
    Qdot_HP = [[], [], [], [], [], [], [], [], [], [], [], [], [], [], []] 
    Qdot_evap = [[], [], [], [], [], [], [], [], [], [], [], [], [], [], []]  
    COP = [[], [], [], [], [], [], [], [], [], [], [], [], [], [], []] 
    # COP_new = [[], [], [], [], [], [], [], [], [], [], [], [], [], [], []]
    COP_new = [[2.8], [2.8], [2.8], [2.8], [2.8], [2.8], [2.8], [2.8], [2.8], [2.8], [2.8], [2.8], [2.8], [2.8], [2.8]]
    Qdot_Losses = [[], [], [], [], [], [], [], [], [], [], [], [], [], [], []] 
    Qdot_network = [[], [], [], [], [], [], [], [], [], [], [], [], [], [], []] 
    Qdot_pipes = [[], [], [], [], [], [], [], [], [], [], [], [], [], [], []]
    P_HP_total = []
    Q_HP_total = []
    Q_evap_total = []
    Q_Losses_total = []
    Q_pipe_total = []
    Qdot_ATES = []
    masscapacity_T = [[], [], [], [], [], [], [], [], [], [], [], [], [], [], []]
    Netheat1 = []
    Surplus_power = []
    Q_storage = []
    Profit = []
    Netprice = []
    Netpower_storage = []
    Power_extraHP = []
    Shortage_power = []
    COP_storageHP = []
    CO2_grid = []
    CO2_PV = []
    

    Qdot_HP_TESS = []
    Qdot_ATES_losses = []


    SoC_BESS = [SoC_0_BESS]
    P_BESS = []
    P_ATE_pump = []

    P_Grid = []
    Netpower = []
    
    Pipelengths = [30, 45, 45, 50, 50, 30, 30, 50, 60, 50, 40, 30, 50, 30, 100]        
    Housetypes = [1, 1, 1, 2, 1, 1, 1, 1, 3, 1, 1, 2, 1, 2, 2]

    
    for step in range(t_final):
        # Thermal
        TESS_HP_control = False
    
        for i in range(street):
            
            
            if Housetypes[i] == 1:
                
                [Qdot_Losses_i, mc_T] = House_Thermal_Losses_1(T_in[i][-1], T_amb[step])
                masscapacity_T[i].append(mc_T)
                Qdot_Losses[i].append(Qdot_Losses_i)
                Q_dot_network = -(mc_T*(T_in[i][-1] - T_in[i][-2])/dt - Qdot_Losses_i)            
                Qdot_network[i].append(Q_dot_network)

            if Housetypes[i] == 2:
                
                [Qdot_Losses_i, mc_T] = House_Thermal_Losses_2(T_in[i][-1], T_amb[step])
                masscapacity_T[i].append(mc_T)
                Qdot_Losses[i].append(Qdot_Losses_i)
                Q_dot_network = -(mc_T*(T_in[i][-1] - T_in[i][-2])/dt - Qdot_Losses_i)            
                Qdot_network[i].append(Q_dot_network)
                
            if Housetypes[i] == 3:
                
                [Qdot_Losses_i, mc_T] = House_Thermal_Losses_3(T_in[i][-1], T_amb[step])
                masscapacity_T[i].append(mc_T)
                Qdot_Losses[i].append(Qdot_Losses_i)
                Q_dot_network = -(mc_T*(T_in[i][-1] - T_in[i][-2])/dt - Qdot_Losses_i)            
                Qdot_network[i].append(Q_dot_network)
            
            T0 = T6[i] + Q_dot_network/(m_dot*c_f)
            T_registry[0][i].append(T0)
    
    
            # HP
            if Thermal_Components[0]:
                if T_in[i][-1] < T_set[step] and T0 < T_sup:
                    #                print('HP - Case 1')
                    HP_active = True
    
                    HP_state = HP_Power(HP_active, T7, T0, m_dot)
                    P_HP[i].append(HP_state[0])
                    Qdot_HP[i].append(HP_state[1])
                    Qdot_evap[i].append(HP_state[2])
                    COP[i].append(HP_state[3])
                    Qdot_HP_TESS.append(0)
                    
    
                elif not(Thermal_Components[1]) or not(TESS_HP_control):
                    #                print('HP - Case 2')
                    P_HP[i].append(0)
                    Qdot_HP[i].append(0)
                    Qdot_evap[i].append(0)
                    COP[i].append(0)
                    Qdot_HP_TESS.append(0)
    
            else:
                P_HP[i].append(0)
                Qdot_HP[i].append(0)
                Qdot_evap[i].append(0)
                Qdot_HP_TESS.append(0)
                T1 = T7
            
                
            T6[i] += (Q_dot_network + Qdot_HP[i][-1])/(m_dot*c_f)
            T_registry[6][i].append(T6[0])
            
            
            if COP[i][-1] == 0:
                if len(COP_new[i]) > 0:
                    COP_new[i].append(COP_new[i][-1])
                else:
                    # Handle the case where there's no previous value, here just appending 0
                    COP_new[i].append(0)
            else:
                COP_new[i].append(COP[i][-1])
            
            
    
            T_in[i].append(new_house_Temperature(T_in[i][-1], Qdot_Losses_i, mc_T, Qdot_HP=Qdot_HP[i][-1])) 
    
            T1 = T7 - Qdot_evap[i][-1]/(m_dot_pipe*c_f)
    
            T_registry[1][i].append(T1)
            
            [Qdot_pipe2, T2, T_groundnew] = Qdot_pipe_ground(T_pipe_out, T1-273, m_dot_pipe, Pipelengths[i], int(T_amb[step] - 273), G[step]/3600, epsilon, alpha_0, k_soil, depth, v, T_s[-1], c_f=4200)
            Qdot_pipes[i].append(Qdot_pipe2)
            T2 += 273 
            
            T7 = T2
        
        T_registry[2][0].append(T2)
        
        # T0_average.append(mean(T_registry[0]))
        # T6_average.append(mean(T_registry[6]))
        # Tin_average.append(mean(T_in))
        
        HP_total = 0
        Qdot_HP_total = 0
        Qdot_evap_total = 0
        Qdot_Losses_total = 0 
        Qdot_pipe_total = 0
        for i in range(street):
            
            HP_total += P_HP[i][-1]
            Qdot_HP_total += Qdot_HP[i][-1]
            Qdot_evap_total += Qdot_evap[i][-1]
            Qdot_Losses_total += Qdot_Losses[i][-1]
            Qdot_pipe_total += Qdot_pipes[i][-1]
        
        P_HP_total.append(HP_total)
        Q_HP_total.append(Qdot_HP_total)
        Q_evap_total.append(Qdot_evap_total)
        Q_Losses_total.append(Qdot_Losses_total)
        Q_pipe_total.append(Qdot_pipe_total)
        
        
        # ATES
        if Thermal_Components[1]:


            ATES_state = update_ATES(TESS_active, T2, T_ATES[-1], radius_ATES, T_soil=Tsoil,)
            P_pump = P_ATES_pump()
            P_ATE_pump.append(P_pump)
            
            T_ATES.append(ATES_state[0])
            Qdot_ATES_losses.append(ATES_state[1])
            Qdot_ATES.append(ATES_state[2])

            T3 = T_ATES[-1]          

        else:
            T_ATES.append(0)
            P_ATE_pump.append(0)
            Qdot_ATES_losses.append(0)
            Qdot_ATES.append(0)
            
            T3 = T2
            
        T_registry[3][0].append(T3)
            
        #PIPE1
       
        [Qdot_pipe, T4, T_groundnew] = Qdot_pipe_ground(T_pipe_out, T3-273, m_dot_pipe, L_1, int(T_amb[step] - 273), G[step]/3600, epsilon, alpha_0, k_soil, depth, v, T_s[-1], c_f=4200)
        
        T4 += 273 
        
        T_registry[4][0].append(T4)
        
        T_s.append(T_groundnew)
        
        Qdot_pipe1.append(Qdot_pipe)

        # PVT

        if Thermal_Components[2]:
            if T_tank_PVT > T4 - 273:
                PVT_active = True

            else:
                PVT_active = False

            if T_tank_PVT > T_ATES[-1] - 273 + 2 and Thermal_Components[2] == True and PVT_2_TESS:
                 PVT_TESS_charge = False

            else:
                 PVT_TESS_charge = False

            [Q_dot_SC, T_PVT_layers, T_tank_PVT, T5, Qdot_PVT2Network, Qdot_PVT2TESS] = Qdot_PVT(PVT_active, PVT_TESS_charge, int(
                T_amb[step] - 273), G[step]/3600, 1, T4-273, T_PVT_layers[0], T_PVT_layers[1], T_PVT_layers[2], T_PVT_layers[3], T_tank_PVT, T_ATES[-1]-273, m_f_dot=0.029085, n_HE=1, t_end=int(15*60), n_modules=n_modules)
            T5 += 273

            Qdot_SC_TESS.append(Qdot_PVT2TESS)
            Qdot_PVT_registry.append(Q_dot_SC)
            Qdot_PVT_Network.append(Qdot_PVT2Network)
            T_tank_PVT_registry.append(T_tank_PVT)

            fluid_registry[0].append(T_PVT_layers[0])
            fluid_registry[1].append(T_PVT_layers[1])
            fluid_registry[2].append(T_PVT_layers[2])
            fluid_registry[3].append(T_PVT_layers[3])
            fluid_registry[4].append(T5 - 273)

            P_PVT.append(PV_out(T_PVT_layers[1], G[step]/3600, A_PVT))

        else:
            T5 = T4
            Qdot_SC_TESS.append(0)
            Qdot_PVT_registry.append(0)
            Qdot_PVT_Network.append(0)
            T_tank_PVT_registry.append(0)
            P_PVT.append(PV_out(T_PVT_layers[1], G[step]/3600, A_PVT))


            fluid_registry[0].append(0)
            fluid_registry[1].append(0)
            fluid_registry[2].append(0)
            fluid_registry[3].append(0)
            fluid_registry[4].append(0)


        T_registry[5][0].append(T5)
        
        Netpower.append(P_PVT[-1] - P_HP_total[-1] - P_ATE_pump[-1] - P_Load_new[step]*1000)
        
        Q_dot_extra = 0
        Surplus = 0
        Shortage = 0
        Price_storage = 0
        Price_no_storage = 0
        
         
        if Q_storage_switch == True:
            
            if  Netpower[-1] > 0:
                
                HPextra_state = HP_extra(int(T_amb[step]) - 273, T5 - 273, Netpower[-1])
                Q_dot_extra = HPextra_state[0]    
                Q_storage.append(Q_dot_extra)
                COP_extra = HPextra_state[1]
                COP_storageHP.append(COP_extra)
            
                T7 = T5 + HE_eff*Q_dot_extra/(m_dot_pipe*c_f)
                Netpower_storage.append(P_PVT[-1] - P_HP_total[-1] - P_ATE_pump[-1] - P_Load_new[step]*1000 - Netpower[-1]) 
                
                Power_extraHP.append(Netpower[-1])
                
            else:
                T7 = T5
                Q_storage.append(0)
                Netpower_storage.append(P_PVT[-1] - P_HP_total[-1] - P_ATE_pump[-1] - P_Load_new[step]*1000) 
                
            Price_storage = sum([a * b for a, b in zip(Netpower_storage, PowerPrices_15min_list)])*0.00000025
            
        if Q_storage_switch == False:
            
            if Netpower[-1] > 0:
                Surplus = Netpower[-1] 
                T7 = T5
                Shortage_power.append(0)
                
            else:
                Shortage = Netpower[-1]
                Surplus_power.append(0)
                T7 = T5
                
            Price_no_storage = sum([a * b for a, b in zip(Netpower, PowerPrices_15min_list)])*0.00000025
        
 
        Surplus_power.append(Surplus) 
        Shortage_power.append(Shortage)
        
        CO2 = Shortage*-0.00025*Intensity_array[step]
        CO2_grid.append(CO2)
        
        CO2PV = P_PVT[-1]*0.00025*40.1
        CO2_PV.append(CO2PV)
           
        Total_surplus_power = sum(Surplus_power)*0.00000025
        Total_shortage_power = sum(Netpower)*0.00000025 - Total_surplus_power
                        
        Total_CO2_grid = sum(CO2_grid)/1000000
        Total_CO2_PV = sum(CO2_PV)/1000000
        
        Total_CO2 = Total_CO2_grid + Total_CO2_PV
        
        T_registry[7][0].append(T7)
        
    
        Netheat1.append(Qdot_HP[0][-1] + Qdot_Losses[0][-1])
        
        Profit.append(Surplus_power[-1]*0.00000025*PowerPrices_15min_list[step])
        Netprice.append(Netpower[-1]*0.00000025*PowerPrices_15min_list[step])
        
        if step % 1000 == 0:
            elapsed_time = time.time() - start_time
            print(f"Step {step}/{t_final} - Elapsed time: {elapsed_time:.2f} seconds")
             
        
        Q_ATES_losses_total = []
        Q_ATES_total = []
        P_ATEP_total =[]
        T_ATES_out = 0
        P_ATEP_total.append(P_ATE_pump)
        Q_ATES_losses_total.append(Qdot_ATES_losses)
        Q_ATES_total.append(Qdot_ATES)
        T_ATES_in = 0
        
        

    return [T_in, Qdot_PVT_Network, Qdot_SC_TESS, Qdot_HP, Qdot_evap, Qdot_HP_TESS, Qdot_pipe, Qdot_ATES_losses, T_registry, T_ATES_out, T_ATES_in, Qdot_Losses, Qdot_network, T_ATES, T_s, fluid_registry, T_tank_PVT_registry, P_Grid, P_PVT, P_HP, P_BESS, SoC_BESS, COP, P_HP_total, P_ATEP_total, Netpower, Netheat1, P_ATE_pump, Total_surplus_power, Total_shortage_power, Total_CO2_grid, Total_CO2_PV, Total_CO2, Q_storage, Q_HP_total, Q_evap_total, Q_Losses_total, Q_pipe_total, Q_ATES_losses_total, Q_ATES_total, Surplus_power, Profit, Netpower_storage, Shortage_power, CO2_grid, CO2_PV, Price_storage, Price_no_storage, COP_new, Power_extraHP, COP_storageHP]

