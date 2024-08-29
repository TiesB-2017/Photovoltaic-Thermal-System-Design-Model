
## Created by: Ties Beijneveld
## TU Delft
## Version: 2.0

###############################################################################
###########################   Imported modules   ##############################
import DHN_library_centralised
import csvreader
import time
from math import ceil
import pandas as pd
import matplotlib.pyplot as plt
import os
import csv
import sys

from numpy import arange, array, empty, zeros, full, trapz, diff, histogram, repeat#, cos, pi, append


###############################################################################
##############################   Parameters   #################################

start_day = 0
end_day = 365
horizon = 4*0

CSVDataTamb = csvreader.read_data(csv='Tamb_15min.csv', address='')
CSVDataP_Load = csvreader.read_data(csv='Load_Profile_15min.csv', address='', delim=',')
CSVDataP_Load_2 = csvreader.read_data(csv='Load_Profile_15min_2.csv', address='', delim=',')
CSVDataRad = csvreader.read_data(csv='Radiation_1min.csv', address='')
CSVDataTsoil = csvreader.read_data(csv='Soil_dy_10cm.csv', address='')
CSVDataPowerPrices = csvreader.read_data(csv='Power_prices_python.csv', address='')



CSVDataTamb.data2array()
CSVDataP_Load.data2array()
CSVDataP_Load_2.data2array()
CSVDataRad.data2array()
CSVDataTsoil.data2cols()
CSVDataPowerPrices.data2array()


b = arange(0,len(CSVDataTamb.ar),4)
T_amb = [i[0]+273 for i in CSVDataTamb.ar]


P_Load = [i for i in CSVDataP_Load.ar[0]]
P_Load_2 = [i[0] for i in CSVDataP_Load_2.ar]

a = arange(0,len(CSVDataRad.ar),15)
G = [CSVDataRad.ar[i][0] for i in a]

# Initial conditions
dt = 60*15                          # In s


# Thermal demand
T_0 = 20 + 273              # In K
T_set_day = [T_0 - 3]*int((6-0)*4) + [T_0]*int((22-6)*4)+ [T_0 - 3]*int((24-22)*4)



#Power prices
b = arange(0,len(CSVDataPowerPrices.ar),1)
PowerPrices = [CSVDataPowerPrices.ar[i][0] for i in b]
PowerPrices_15min = repeat(PowerPrices, 4)
PowerPrices_15min_list = PowerPrices_15min.tolist()

Intensities = [472, 518, 474, 551, 429, 341, 303, 383, 410, 388, 400, 384]
values = []

# Loop to append each value 2920 times
for value in Intensities:
    values.extend([value] * 2920)

# Convert the list to a NumPy array
Intensity_array = array(values)


df = pd.DataFrame(P_Load_2, columns=['Power_kW'])
df['Energy_kWh'] = df['Power_kW'] * 24.2
total_power_kWh = df['Energy_kWh'].sum()

P_Load_new = [i * 24.2 for i in P_Load_2]


PVT_active = True
T_sup = 53 + 273
T_glass_0 = 20
T_PV_0 = 18
T_a_0 = 20
T_f_0 = 20
T_PVT_layers = [T_glass_0, T_PV_0, T_a_0, T_f_0]
T_tank_PVT = 15



# TESS
TESS_vol_list = arange(0,2) #arange(0,11)
depth = 0.2
dy = 0.1
Height_TESS = 1.8
ATES_active = True
TESS_mode = 'cycling' 
Tsoil = 0 + 273 



PVT_2_TESS = True
HP_2_TESS = True

if TESS_mode == 'cycling':
    TESS_min_op_T = T_sup
    TESS_max_op_T = 90 + 273
elif TESS_mode == 'full':
    TESS_min_op_T = 90 + 273
    TESS_max_op_T = 95 + 273


# HP
HP_active = True

Q_storage_switch = True

# Thermal Network
street = 15
T5 = 11 + 273
T7 = 11 + 273
# T6 = 50 + 273
T6 = [50 + 273 for _ in range(street)]
alpha_0 = 0.25
epsilon = 0.25 
k_soil=1.19
depth=6
T_s_0 = 0


#PIPES
L_1 = 100

# BESS
SoCmax = 0.9
SoCmin = 0.2
SoC_0_BESS = 0.5
BESS_capacity = 3.36
P_BESS_max = 1.28
P_Grid_max = 0
Capacity_BESS = 3.36
SoC_BESS = [SoC_0_BESS]                     # In %
charge_efficiency = 0.943
discharge_efficiency = 0.943

# Grid
Thermal_Components = [HP_active, ATES_active, PVT_active]




module = 800
# ATES_radius = 100
T0 = 273 + 11
Modules_array =[800]# [700, 710, 720, 730, 740, 750, 760, 770, 780, 790, 800]
ATES_radius_array = [25]

T0_array = [273 + 11] #[273 + 26.7404188, 273 + 26.08924782, 273 + 25.45262683, 273 + 24.83055583, 273 + 24.22303483, 273 + 23.63006383, 273 + 23.05164282, 273 + 22.4877718,  273 + 21.93845078, 273 + 21.40367976, 273 + 20.88345872]

years = 1

total_co2_data = []

t_final = int((end_day - start_day)*24*3600/dt)         # In s
T_set = array(T_set_day*((end_day + ceil(horizon/(24*4)))-start_day))
    
start_time = time.time()

base_folder = 'Test random'


def save_multiple_lists_to_csv(data_dict, folder_name):
    for file_path, data_list in data_dict.items():
        # Ensure the directory exists
        folder_path = os.path.join(folder_name, os.path.dirname(file_path))
        os.makedirs(folder_path, exist_ok=True)
        
        # Save the list to a CSV file
        full_file_path = os.path.join(folder_path, os.path.basename(file_path))
        with open(full_file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            
            # Write each item in the list to the CSV file
            for item in data_list:
                writer.writerow([item])

for ATES_radius, T0 in zip(ATES_radius_array, T0_array):
    # Loop through Modules_array
    for module in Modules_array:
    
        [T_in, Qdot_PVT_Network, Qdot_SC_TESS, Qdot_HP, Qdot_evap, Qdot_HP_TESS, Qdot_pipe, Qdot_ATES_losses, T_registry, T_ATES_out, T_ATES_in, Qdot_Losses, Qdot_network, T_ATES, T_s, fluid_registry, T_tank_PVT_registry, P_Grid, P_PVT, P_HP, P_BESS, SoC_BESS, COP, P_HP_total, P_ATEP_total, Netpower, Netheat1, P_ATE_pump, Total_surplus_power, Total_shortage_power, Total_CO2_grid, Total_CO2_PV, Total_CO2, Q_storage, Q_HP_total, Q_evap_total, Q_Losses_total, Q_pipe_total, Q_ATES_losses_total, Q_ATES_total, Surplus_power, Profit, Netpower_storage, Shortage_power, CO2_grid, CO2_PV, Price_storage, Price_no_storage, COP_new, Power_extraHP, COP_storageHP] = DHN_library_centralised.Thermal_Electrical_model(Thermal_Components, Q_storage_switch, T_0, T5, T6, T7, T0, T_tank_PVT, T_PVT_layers, TESS_min_op_T, TESS_max_op_T, SoC_0_BESS, T_amb, T_set, G, Tsoil, P_Load_new, t_final, T_s_0, module, ATES_radius, PowerPrices_15min_list, start_time, Intensity_array, P_BESS_max, P_Grid_max, Capacity_BESS, PVT_2_TESS, HP_2_TESS)

        Q_evap_total_negative = [-q for q in Q_evap_total]
        P_Load_new_negative = [-q for q in P_Load_new]
        P_ATE_pump_negative = [-q for q in P_ATE_pump]
        P_HP_total_negative = [-q for q in P_HP_total]
        
        
        
        folder_name = os.path.join(base_folder, f"Module_{module}_ATES_{ATES_radius}")
        print(f"Creating folder: {folder_name}") 
        os.makedirs(folder_name, exist_ok=True)
        
        
        
        data_dict = {f'T_{i}.csv': T_registry[i] for i in range(8)}
        data_dict['T_ATES_in.csv'] = T_registry[2][0]
        data_dict['T_ATES_out.csv'] = T_registry[3][0]
        data_dict['T_PVT_out.csv'] = T_registry[5][0]
        data_dict['T_storage.csv'] =T_registry[7][0]
        data_dict['HPevaporator.csv'] = Q_evap_total_negative
        data_dict['Q_Losses_total.csv'] = Q_Losses_total
        data_dict['Qdot_PVT_Network.csv'] = Qdot_PVT_Network
        data_dict['Qdot_ATES_losses.csv'] = Qdot_ATES_losses
        data_dict['Q_pipe_total.csv'] = Q_pipe_total
        data_dict['Q_storage.csv'] = Q_storage
        data_dict['P_HP_total_negative.csv'] = P_HP_total_negative
        data_dict['P_Load_new_negative.csv'] = P_Load_new_negative
        data_dict['P_PVT.csv'] = P_PVT
        data_dict['P_ATE_pump_negative.csv'] = P_ATE_pump_negative
        data_dict['COP_new.csv'] = COP_new
        data_dict['COP_extra.csv'] = COP_storageHP
        data_dict['T_in_0_2.csv'] = T_in[0][2:]
        data_dict['T_in_3_2.csv'] = T_in[3][2:]
        data_dict['T_in_8_2.csv'] = T_in[8][2:]
        # data_dict['T_ATES_in 2.csv'] = T_ATES_in[5]
        # data_dict['T_ATES_in 3.csv'] = T_ATES_in[11]
        
        # Save all data lists to their respective CSV files
        save_multiple_lists_to_csv(data_dict, folder_name)
        
        summary_data = {
            'P_Load': sum(P_Load_new)*0.00025,
            'P_PV': sum(P_PVT)*0.00000025,
            'P_HP': sum(P_HP_total)*0.00000025,
            'P_Pump': sum(P_ATE_pump)*0.00000025,
            'P_surplus': Total_surplus_power,
            'P_shortage': Total_shortage_power,
            'P_net': sum(Netpower)*0.00000025,
            'Q_PVT': sum(Qdot_PVT_Network)*0.00000025,
            'Q_Building': sum(Q_Losses_total)*0.00000025,
            'Q_evap': -sum(Q_evap_total)*0.00000025,
            'Q_HP': sum(Q_HP_total)*0.00000025,
            'Q_ATES': sum(Qdot_ATES_losses)*0.00000025,
            'Q_pipe': sum(Q_pipe_total)*0.00000025,
            'Q_Storage': sum(Q_storage)*0.00000025,
            'P Storage': sum(Power_extraHP)*0.00000025, 
            'CO2_eq_grid': Total_CO2_grid,
            'CO2_eq_PV': Total_CO2_PV,
            'CO2_eq': Total_CO2,
            'Price no storage': Price_no_storage,
            'Price storage': Price_storage
            }
        
        summary_file = os.path.join(folder_name, 'summary.csv')
        with open(summary_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            for key, value in summary_data.items():
                writer.writerow([key, value])
        
        total_co2_data.append([module, ATES_radius, Total_CO2])

# Save Total_CO2 data for all configurations
total_co2_file = os.path.join(base_folder, 'Total_CO2_summary_2.csv')
with open(total_co2_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Module', 'ATES_Radius', 'Total_CO2'])
    writer.writerows(total_co2_data)
    
    

step = arange(0, t_final)

start_date = pd.to_datetime('2023-01-01')
time_index = pd.date_range(start_date, periods=len(step), freq='15T')
mid_months = pd.date_range(start_date, periods=12, freq='MS') + pd.DateOffset(days=14)

print(sum(P_Load_new)*0.00025, 'E load(MWh)')
print(sum(P_PVT)*0.00000025, 'E PV(MWh)')
print(sum(P_HP_total)*0.00000025, 'E HP(MWh)')
print(sum(P_ATE_pump)*0.00000025, 'E Pump(MWh)')
# print(sum(Netpower_storage)*0.00000025, 'E storage')

print(Total_surplus_power, 'E surplus(MWh)')
print(Total_shortage_power, 'E shortage(MWh)')
print(sum(Netpower)*0.00000025, 'E net(MWh)')


print(sum(Qdot_PVT_Network)*0.00000025, 'Q PVT(MWh)')
print(sum(Q_Losses_total)*0.00000025, 'Q Building(MWh)')
print(-sum(Q_evap_total)*0.00000025, 'Q evap(MWh)')
print(sum(Q_HP_total)*0.00000025, 'Q HP(MWh)')
# print(sum(Q_ATES_total)*0.00000025, 'Q ATES(MWh)')
print(sum(Qdot_ATES_losses)*0.00000025, 'Q ATES(MWh)')
print(sum(Q_pipe_total)*0.00000025, 'Q pipe(MWh)')

print(sum(Q_storage)*0.00000025, 'Q Storage(MWh)')
print(sum(Power_extraHP)*0.00000025, 'P Storage(MWh)')


print(Total_CO2_grid, 'CO2 eq grid(tonnes)')
print(Total_CO2_PV, 'CO2 eq PV(tonnes)')
print(Total_CO2, 'CO2 eq(tonnes)')
print(Price_no_storage, 'Price no storage(EUR)')
print(Price_storage, 'Price with storage(EUR)')


plt.figure()


plt.plot(time_index, T_in[0][2:] - full(t_final, 273), label='T_in_average_small')
plt.plot(time_index, T_in[3][2:] - full(t_final, 273), label='T_in_average_medium')
plt.plot(time_index, T_in[8][2:] - full(t_final, 273), label='T_in_average_large')
        
plt.xticks(mid_months, [date.strftime('%B') for date in mid_months])
plt.xticks(rotation=45)

plt.xlabel('Time')
plt.ylabel('Temperature (C)')
# plt.title('Temperature buildings')
plt.legend(loc='upper right')

    
plt.show()




plt.figure()
    
plt.plot(time_index, T_registry[2][0]- full(t_final, 273), label='T_ATES_in', zorder=4)
# plt.plot(time_index, T_ATES_in[5]- full(t_final, 273), label='T_ATES_in 2', zorder=2)
# plt.plot(time_index, T_ATES_in[11]- full(t_final, 273), color = 'red', label='T_ATES_in 3', zorder=3)
plt.plot(time_index, T_registry[3][0]- full(t_final, 273), label='T_ATES_out', zorder=6)
plt.plot(time_index, T_registry[5][0]- full(t_final, 273), label='T_PVT_out', zorder=5)

# plt.plot(time_index, T_registry[7][0]- full(t_final, 273), label='T_Storage', zorder=3)


plt.xticks(mid_months, [date.strftime('%B') for date in mid_months])
plt.xticks(rotation=45)

plt.xlabel('Time')
plt.ylabel('Temperature (C)')
# plt.title('Temperatures district heating network')
plt.legend(loc='upper right')
plt.grid(True) 
plt.show() 

            

plt.figure()
Q_evap_total_negative = [-q for q in Q_evap_total]
# plt.plot(step, Q_HP_total, label='Qdot_HP')
plt.plot(time_index, Q_evap_total_negative * full(t_final, 0.001), label='Q evaporator HP', zorder=1)
plt.plot(time_index, Q_Losses_total  * full(t_final, 0.001), label='Q losses house', zorder=3)
plt.plot(time_index, Qdot_PVT_Network * full(t_final, 0.001), label='Q PVT',zorder=1)
plt.plot(time_index, Qdot_ATES_losses  * full(t_final, 0.001), color = 'yellow', label='Q ATES', zorder=4)
# plt.plot(time_index, Q_ATES_total * full(t_final, 0.001), color = 'yellow', label='Qdot_ATES', zorder=4)
plt.plot(time_index, Q_pipe_total  * full(t_final, 0.001), label='Q pipe', zorder=5)
plt.plot(time_index, Q_storage * full(t_final, 0.001), label='Q storage', zorder=2)


plt.xticks(mid_months, [date.strftime('%B') for date in mid_months])
plt.xticks(rotation=45)

plt.xlabel('Time')
plt.ylabel('Heat district heating network(kW)', color='k')
plt.tick_params(axis='y', labelcolor='k')
plt.legend(loc='upper right')


plt.show()



plt.figure()
plt.plot(time_index, P_HP_total_negative * full(t_final, 0.001), label='P_HP', zorder=2)
plt.plot(time_index, P_Load_new_negative[:len(step)] , label='P_Load', zorder=1)
plt.plot(time_index, P_PVT * full(t_final, 0.001), label='P_PVT')
plt.plot(time_index, P_ATE_pump_negative * full(t_final, 0.001), label='P_ATES_pump', zorder=3)




plt.xticks(mid_months, [date.strftime('%B') for date in mid_months])
plt.xticks(rotation=45)

plt.xlabel('Time')
plt.ylabel('Power district heating network(kW)', color='k')
plt.tick_params(axis='y', labelcolor='k')
plt.legend(loc='upper right')
        
plt.show()



plt.figure()

# indices = list(range(4)) + list(range(6, 10)) + list(range(12, 16))
# for i in indices:
for i in range(street):
    plt.plot(time_index, COP_new[i][1:], label=f'COP{i+1}', zorder=i)
    
plt.rcParams.update({
#    "text.usetex": True,
"font.family": "Times New Roman",
'font.size': 14        
})
    
plt.xlabel('Time')
plt.ylabel('COP')
# plt.title('COP Heat pumps')
# plt.legend(loc='upper left')

plt.xticks(mid_months, [date.strftime('%B') for date in mid_months])
plt.xticks(rotation=45)
    
plt.show()



plt.figure()

counts, bins = histogram(Netpower * full(t_final, 0.001), bins=30)
percentages = 100 * counts / len(Netpower)


plt.bar(bins[:-1], percentages, width=diff(bins), edgecolor='black', align='edge', color='g', alpha=0.6)

plt.xlabel('P_grid (kW)')
plt.ylabel('Percentage (%)', color='b')
# plt.title('Histogram grid exchange')

plt.show()

print((sum(Qdot_PVT_Network) + sum(Q_storage) + sum(Q_Losses_total) - sum(Q_evap_total) + sum(Qdot_ATES_losses) + sum(Q_pipe_total))*0.00025, 'Net heat district heating network(kWh)')




# for TESS_vol in TESS_vol_list:

#     if TESS_vol != 0 and TESS_active == True:
#         TESS_active_iteration = True
#     else:
#         TESS_active_iteration = False   
    
#     for n_modules in n_modules_list:
        
#         if n_modules != 0 and PVT_active == True:
#             PVT_active_iteration = True
#         else:
#             PVT_active_iteration = False
         

#         ###############################################################################
#         ##############################   Simulation   #################################
        
#         Thermal_Components = [PVT_active_iteration, TESS_active_iteration, HP_active]     # PVT, TESS, HP
          
           
#         start_cycle = time.time()    # The timer is initializad.
         
#         [T_in, Qdot_PVT_Network, Qdot_SC_TESS, Qdot_TESS, Qdot_HP, Qdot_HP_TESS, T_registry, Qdot_Losses, T_TESS, Qdot_TESS_SD, fluid_registry, T_tank_PVT_registry, P_Grid, P_PVT, P_HP, P_BESS, SoC_BESS] = MCES_library.Thermal_Electrical_model(Thermal_Components, T_0, T4, T_TESS_0, T_tank_PVT, T_PVT_layers, TESS_min_op_T, TESS_max_op_T, SoC_0_BESS, T_amb, T_set, G, Tsoil, P_Load, t_final, n_modules, P_BESS_max, P_Grid_max, Capacity_BESS, PVT_2_TESS, HP_2_TESS, m = TESS_vol*1000)
                                                                                                                                                                                                                           
#         end_cycle = time.time()    # The timer is ended.
            
#         elapsed_time_cycle = (end_cycle - start_cycle)/60

#         cold_days = 0
#         for i in range(int(len(T_in[2:])/96)):
#             if min(T_in[2+i*96:2+(i+1)*96]) < 16 + 273:
#                 cold_days += 1        
        
#         ###############################################################################
#         ###############################   CSV logging   ################################
        
# #        file_name = 'PVT_' + str(n_modules) + '_TESS_' + str(int(TESS_vol)) + '.csv'
# #        
# #        results = zip(T_in[2:], Qdot_PVT_Network, Qdot_SC_TESS, Qdot_TESS, Qdot_HP, T_registry[0], T_registry[1], T_registry[2], T_registry[3], T_registry[4], Qdot_Losses, T_TESS[1:], Qdot_TESS_SD, fluid_registry[0], fluid_registry[1], fluid_registry[2], fluid_registry[3], fluid_registry[4][1:], T_tank_PVT_registry, P_Grid, P_PVT, P_HP, P_BESS, SoC_BESS[1:])
# #        
# #        
# #        if 'file_name' in globals():
# #            import csv
# #            with open(file_name, "w") as f:
# #                writer = csv.writer(f)
# #                for row in results:
# #                    writer.writerow(row)
# #                f.close()

#         ###############################################################################
#         ################################   Logging   ##################################
        
#         elapsed_time_cycle_registry[n_modules, TESS_vol] = elapsed_time_cycle
#         cold_days_registry[n_modules, TESS_vol] = cold_days
#         Qdot_Losses_registry[n_modules, TESS_vol] = -0.25*sum(Qdot_Losses)/1000
#         Qdot_PVT_Network_registry[n_modules, TESS_vol] = 0.25*sum(Qdot_PVT_Network)/1000
#         Qdot_SC_TESS_registry[n_modules, TESS_vol] = 0.25*sum(Qdot_SC_TESS)/1000
#         Qdot_TESS_registry[n_modules, TESS_vol] = 0.25*sum(Qdot_TESS)/1000
#         Qdot_TESS_SD_registry[n_modules, TESS_vol] = 0.25*sum(Qdot_TESS_SD)/1000
#         Qdot_HP_registry[n_modules, TESS_vol] = 0.25*sum(Qdot_HP)/1000
#         Qdot_HP_TESS_registry[n_modules, TESS_vol] = 0.25*sum(Qdot_HP_TESS)/1000
#         P_PVT_registry[n_modules, TESS_vol] = 0.25*sum(P_PVT)
#         P_BESS_stored_registry[n_modules, TESS_vol] = -0.25*sum([i for i in P_BESS if i < 0])
#         P_BESS_provided_registry[n_modules, TESS_vol] = 0.25*sum([i for i in P_BESS if i > 0])
#         P_HP_registry[n_modules, TESS_vol] = 0.25*sum(P_HP)
#         P_Grid_consumed_registry[n_modules, TESS_vol] = 0.25*sum([i for i in P_Grid if i > 0])
#         P_Grid_returned_registry[n_modules, TESS_vol] = -0.25*sum([i for i in P_Grid if i < 0])
        
#         T_TESS_registry[n_modules][TESS_vol] = T_TESS
#         SoC_BESS_registry[n_modules][TESS_vol] = SoC_BESS
        
#         ###############################################################################
#         ###############################   Printing   ##################################
        
        
#         print('----------------------------------------------------------------------')
#         print('')
#         print('PVT_' + str(n_modules) + '_TESS_' + str(int(TESS_vol)))
#         print('Elapsed time per run: ', elapsed_time_cycle, ' min')
#         print('')
#         print('Cold days: ', cold_days)
#         print('Thermal energy losses: ',  -0.25*sum(Qdot_Losses)/1000, ' kWh')
#         print('Thermal energy from the PVT to the thermal network: ',  0.25*sum(Qdot_PVT_Network)/1000, ' kWh')
#         print('Thermal energy from the PVT to the TESS: ',  0.25*sum(Qdot_SC_TESS)/1000, ' kWh')
#         print('Thermal energy from the TESS to the thermal network: ',  0.25*sum(Qdot_TESS)/1000, ' kWh')
#         print('Thermal energy lost from the TESS to the soil: ',  0.25*sum(Qdot_TESS_SD)/1000, ' kWh')
#         print('Thermal energy from the HP to the TESS: ',  0.25*sum(Qdot_HP_TESS)/1000, ' kWh')
#         print('Thermal energy from the HP to the thermal network: ',  0.25*sum(Qdot_HP)/1000, ' kWh')
#         print('Energy produced from the PVT: ', 0.25*sum(P_PVT), ' kWh')
#         print('Energy stored in the BESS: ', -0.25*sum([i for i in P_BESS if i < 0]), ' kWh')
#         print('Energy provided by the BESS: ', 0.25*sum([i for i in P_BESS if i > 0]), ' kWh')
#         print('Energy consumed by the HP: ', 0.25*sum(P_HP), 'kWh')
#         print('Energy consumed from the grid: ', 0.25*sum([i for i in P_Grid if i > 0]), ' kWh')
#         print('Energy returned to the grid: ', -0.25*sum([i for i in P_Grid if i < 0]), ' kWh')
#         print('')
    

# COP = [[(Qdot_HP_registry[m][v] + Qdot_HP_TESS_registry[m][v])/(P_HP_registry[m][v]) if P_HP_registry[m][v] != 0 else float('nan') for m in n_modules_list] for v in TESS_vol_list]

# end = time.time()
# elapsed_time = (end - start)/60

# print('----------------------------------------------------------------------')
# print('')
# print('Total elapsed time: ', elapsed_time, ' min')
# print('----------------------------------------------------------------------')

# ###############################################################################
# ################################   Ploting   ##################################

# #



# plt.rcParams.update({
# #    "text.usetex": True,
#     "font.family": "Times New Roman",
#     'font.size': 16
# })

# #fig10, (ax10) = plt.subplots(1, 1, constrained_layout=True, sharey=True)
# #a10 = plt.imshow(elapsed_time_cycle_registry, cmap='plasma', interpolation='bicubic', origin='lower')   # , vmin = -15, vmax = 35, , interpolation='bicubic'
# #cbar10 = fig10.colorbar(a10, ax=ax10)
# ##cbar10.set_ticks(arange(-15, 36, step=5), [round(i,0) for i in arange(-15, 36, step=5)])
# #cbar10.set_label('Time per case, t, [s]')
# ##ax10.set_title('Year 2021')
# ##plt.xticks(arange(1, int((end_day - start_day)*24*4), step=24*31*4), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])  # Set label locations. 
# #ax10.set_xlabel('TESS volume, [m$^{3}$]')
# ##plt.yticks(arange(0, int(y_f/dy), step=10), [round(i,1) for i in arange(0, y_f, step=0.1)])
# #ax10.set_ylabel('Number of PVT modules, [-]')
# #
# #
# fig11, (ax11) = plt.subplots(1, 1, constrained_layout=True, sharey=True)
# a11 = plt.imshow(cold_days_registry, cmap='plasma', interpolation='bicubic', origin='lower')   # , vmin = -15, vmax = 35
# cbar11 = fig11.colorbar(a11, ax=ax11)
# #cbar11.set_ticks(arange(-15, 36, step=5), [round(i,0) for i in arange(-15, 36, step=5)])
# cbar11.set_label('Number of cold days, [-]')
# #ax11.set_title('Year 2021')
# #plt.xticks(arange(1, int((end_day - start_day)*24*4), step=24*31*4), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])  # Set label locations. 
# ax11.set_xlabel('TESS volume, [m$^{3}$]')
# #plt.yticks(arange(0, int(y_f/dy), step=10), [round(i,1) for i in arange(0, y_f, step=0.1)])
# ax11.set_ylabel('Number of PVT modules, [-]')


# fig12, (ax12) = plt.subplots(1, 1, constrained_layout=True, sharey=True)
# a12 = plt.imshow(Qdot_Losses_registry, cmap='plasma', interpolation='bicubic', origin='lower')   # , vmin = -15, vmax = 35
# cbar12 = fig12.colorbar(a12, ax=ax12)
# #cbar12.set_ticks(arange(-15, 36, step=5), [round(i,0) for i in arange(-15, 36, step=5)])
# cbar12.set_label('Thermal losses of the house, Q$_{L}$, [kWh]')
# #ax12.set_title('Year 2021')
# #plt.xticks(arange(1, int((end_day - start_day)*24*4), step=24*31*4), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])  # Set label locations. 
# ax12.set_xlabel('TESS volume, [m$^{3}$]')
# #plt.yticks(arange(0, int(y_f/dy), step=10), [round(i,1) for i in arange(0, y_f, step=0.1)])
# ax12.set_ylabel('Number of PVT modules, [-]')


# fig13, (ax13) = plt.subplots(1, 1, constrained_layout=True, sharey=True)
# a13 = plt.imshow(Qdot_PVT_Network_registry, cmap='plasma', interpolation='bicubic', origin='lower')   # , vmin = -15, vmax = 35
# cbar13 = fig13.colorbar(a13, ax=ax13)
# #cbar13.set_ticks(arange(-15, 36, step=5), [round(i,0) for i in arange(-15, 36, step=5)])
# cbar13.set_label('Thermal energy from the PVT to the network, Q$_{PVT}^{N}$, [kWh]')
# #ax13.set_title('Year 2021')
# #plt.xticks(arange(1, int((end_day - start_day)*24*4), step=24*31*4), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])  # Set label locations. 
# ax13.set_xlabel('TESS volume, [m$^{3}$]')
# #plt.yticks(arange(0, int(y_f/dy), step=10), [round(i,1) for i in arange(0, y_f, step=0.1)])
# ax13.set_ylabel('Number of PVT modules, [-]')

# fig14, (ax14) = plt.subplots(1, 1, constrained_layout=True, sharey=True)
# a14 = plt.imshow(Qdot_SC_TESS_registry, cmap='plasma', interpolation='bicubic', origin='lower')   # , vmin = -15, vmax = 35
# cbar14 = fig14.colorbar(a14, ax=ax14)
# #cbar14.set_ticks(arange(-15, 36, step=5), [round(i,0) for i in arange(-15, 36, step=5)])
# cbar14.set_label('Thermal energy from the PVT to the TESS, Q$_{PVT}^{TESS}$, [kWh]')
# #ax14.set_title('Year 2021')
# #plt.xticks(arange(1, int((end_day - start_day)*24*4), step=24*31*4), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])  # Set label locations. 
# ax14.set_xlabel('TESS volume, [m$^{3}$]')
# #plt.yticks(arange(0, int(y_f/dy), step=10), [round(i,1) for i in arange(0, y_f, step=0.1)])
# ax14.set_ylabel('Number of PVT modules, [-]')


# fig15, (ax15) = plt.subplots(1, 1, constrained_layout=True, sharey=True)
# a15 = plt.imshow(Qdot_TESS_registry, cmap='plasma', interpolation='bicubic', origin='lower')   # , vmin = -15, vmax = 35
# cbar15 = fig15.colorbar(a15, ax=ax15)
# #cbar15.set_ticks(arange(-15, 36, step=5), [round(i,0) for i in arange(-15, 36, step=5)])
# cbar15.set_label('Thermal energy from the TESS, Q$_{TESS}$, [kWh]')
# #ax15.set_title('Year 2021')
# #plt.xticks(arange(1, int((end_day - start_day)*24*4), step=24*31*4), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])  # Set label locations. 
# ax15.set_xlabel('TESS volume, [m$^{3}$]')
# #plt.yticks(arange(0, int(y_f/dy), step=10), [round(i,1) for i in arange(0, y_f, step=0.1)])
# ax15.set_ylabel('Number of PVT modules, [-]')

# fig28, (ax28) = plt.subplots(1, 1, constrained_layout=True, sharey=True)
# a28 = plt.imshow(Qdot_TESS_SD_registry, cmap='plasma', interpolation='bicubic', origin='lower')   # , vmin = -15, vmax = 35
# cbar28 = fig28.colorbar(a28, ax=ax28)
# #cbar28.set_ticks(arange(-15, 36, step=5), [round(i,0) for i in arange(-15, 36, step=5)])
# cbar28.set_label('Thermal energy lost from the TESS, Q$_{TESS}^{SD}$, [kWh]')
# #ax28.set_title('Year 2021')
# #plt.xticks(arange(1, int((end_day - start_day)*24*4), step=24*31*4), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])  # Set label locations. 
# ax28.set_xlabel('TESS volume, [m$^{3}$]')
# #plt.yticks(arange(0, int(y_f/dy), step=10), [round(i,1) for i in arange(0, y_f, step=0.1)])
# ax28.set_ylabel('Number of PVT modules, [-]')


# fig16, (ax16) = plt.subplots(1, 1, constrained_layout=True, sharey=True)
# a16 = plt.imshow(Qdot_HP_registry, cmap='plasma', interpolation='bicubic', origin='lower')   # , vmin = -15, vmax = 35
# cbar16 = fig16.colorbar(a16, ax=ax16)
# #cbar16.set_ticks(arange(-15, 36, step=5), [round(i,0) for i in arange(-15, 36, step=5)])
# cbar16.set_label('Thermal energy from the HP, Q$_{HP}$, [kWh]')
# #ax16.set_title('Year 2021')
# #plt.xticks(arange(1, int((end_day - start_day)*24*4), step=24*31*4), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])  # Set label locations. 
# ax16.set_xlabel('TESS volume, [m$^{3}$]')
# #plt.yticks(arange(0, int(y_f/dy), step=10), [round(i,1) for i in arange(0, y_f, step=0.1)])
# ax16.set_ylabel('Number of PVT modules, [-]')

# fig27, (ax27) = plt.subplots(1, 1, constrained_layout=True, sharey=True)
# a27 = plt.imshow(Qdot_HP_TESS_registry, cmap='plasma', interpolation='bicubic', origin='lower')   # , vmin = -15, vmax = 35
# cbar27 = fig27.colorbar(a27, ax=ax27)
# #cbar27.set_ticks(arange(-15, 36, step=5), [round(i,0) for i in arange(-15, 36, step=5)])
# cbar27.set_label('Thermal energy from the HP to the TESS, Q$_{HP}^{TESS}$, [kWh]')
# #ax27.set_title('Year 2021')
# #plt.xticks(arange(1, int((end_day - start_day)*24*4), step=24*31*4), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])  # Set label locations. 
# ax27.set_xlabel('TESS volume, [m$^{3}$]')
# #plt.yticks(arange(0, int(y_f/dy), step=10), [round(i,1) for i in arange(0, y_f, step=0.1)])
# ax27.set_ylabel('Number of PVT modules, [-]')


# fig17, (ax17) = plt.subplots(1, 1, constrained_layout=True, sharey=True)
# a17 = plt.imshow(P_PVT_registry, cmap='plasma', interpolation='bicubic', origin='lower')   # , vmin = -15, vmax = 35
# cbar17 = fig17.colorbar(a17, ax=ax17)
# #cbar17.set_ticks(arange(-15, 36, step=5), [round(i,0) for i in arange(-15, 36, step=5)])
# cbar17.set_label('Electric energy produced by the PVT, E$_{PVT}$, [kWh]')
# #ax17.set_title('Year 2021')
# #plt.xticks(arange(1, int((end_day - start_day)*24*4), step=24*31*4), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])  # Set label locations. 
# ax17.set_xlabel('TESS volume, [m$^{3}$]')
# #plt.yticks(arange(0, int(y_f/dy), step=10), [round(i,1) for i in arange(0, y_f, step=0.1)])
# ax17.set_ylabel('Number of PVT modules, [-]')


# fig18, (ax18) = plt.subplots(1, 1, constrained_layout=True, sharey=True)
# a18 = plt.imshow(P_BESS_stored_registry, cmap='plasma', interpolation='bicubic', origin='lower')   # , vmin = -15, vmax = 35
# cbar18 = fig18.colorbar(a18, ax=ax18)
# #cbar18.set_ticks(arange(-15, 36, step=5), [round(i,0) for i in arange(-15, 36, step=5)])
# cbar18.set_label('Electric energy stored in the BESS, E$_{BESS}^{in}$, [kWh]')
# #ax18.set_title('Year 2021')
# #plt.xticks(arange(1, int((end_day - start_day)*24*4), step=24*31*4), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])  # Set label locations. 
# ax18.set_xlabel('TESS volume, [m$^{3}$]')
# #plt.yticks(arange(0, int(y_f/dy), step=10), [round(i,1) for i in arange(0, y_f, step=0.1)])
# ax18.set_ylabel('Number of PVT modules, [-]')


# fig19, (ax19) = plt.subplots(1, 1, constrained_layout=True, sharey=True)
# a19 = plt.imshow(P_BESS_provided_registry, cmap='plasma', interpolation='bicubic', origin='lower')   # , vmin = -15, vmax = 35
# cbar19 = fig19.colorbar(a19, ax=ax19)
# #cbar19.set_ticks(arange(-15, 36, step=5), [round(i,0) for i in arange(-15, 36, step=5)])
# cbar19.set_label('Electric energy supplied by the BESS, E$_{BESS}^{out}$, [kWh]')
# #ax19.set_title('Year 2021')
# #plt.xticks(arange(1, int((end_day - start_day)*24*4), step=24*31*4), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])  # Set label locations. 
# ax19.set_xlabel('TESS volume, [m$^{3}$]')
# #plt.yticks(arange(0, int(y_f/dy), step=10), [round(i,1) for i in arange(0, y_f, step=0.1)])
# ax19.set_ylabel('Number of PVT modules, [-]')


# fig20, (ax20) = plt.subplots(1, 1, constrained_layout=True, sharey=True)
# a20 = plt.imshow(P_HP_registry, cmap='plasma', interpolation='bicubic', origin='lower')   # , vmin = -15, vmax = 35
# cbar20 = fig20.colorbar(a20, ax=ax20)
# #cbar20.set_ticks(arange(-15, 36, step=5), [round(i,0) for i in arange(-15, 36, step=5)])
# cbar20.set_label('Electric energy consumed by the HP, E$_{HP}$, [kWh]')
# #ax20.set_title('Year 2021')
# #plt.xticks(arange(1, int((end_day - start_day)*24*4), step=24*31*4), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])  # Set label locations. 
# ax20.set_xlabel('TESS volume, [m$^{3}$]')
# #plt.yticks(arange(0, int(y_f/dy), step=10), [round(i,1) for i in arange(0, y_f, step=0.1)])
# ax20.set_ylabel('Number of PVT modules, [-]')


# fig21, (ax21) = plt.subplots(1, 1, constrained_layout=True, sharey=True)
# a21 = plt.imshow(P_Grid_consumed_registry, cmap='plasma', interpolation='bicubic', origin='lower')   # , vmin = -15, vmax = 35
# cbar21 = fig21.colorbar(a21, ax=ax21)
# #cbar21.set_ticks(arange(-15, 36, step=5), [round(i,0) for i in arange(-15, 36, step=5)])
# cbar21.set_label('Electric energy consumed from the grid, E$_{G}^{in}$, [kWh]')
# #ax21.set_title('Year 2021')
# #plt.xticks(arange(1, int((end_day - start_day)*24*4), step=24*31*4), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])  # Set label locations. 
# ax21.set_xlabel('TESS volume, [m$^{3}$]')
# #plt.yticks(arange(0, int(y_f/dy), step=10), [round(i,1) for i in arange(0, y_f, step=0.1)])
# ax21.set_ylabel('Number of PVT modules, [-]')


# fig22, (ax22) = plt.subplots(1, 1, constrained_layout=True, sharey=True)
# a22 = plt.imshow(P_Grid_returned_registry, cmap='plasma', interpolation='bicubic', origin='lower')   # , vmin = -15, vmax = 35
# cbar22 = fig22.colorbar(a22, ax=ax22)
# #cbar22.set_ticks(arange(-15, 36, step=5), [round(i,0) for i in arange(-15, 36, step=5)])
# cbar22.set_label('Electric energy sent back to the grid, E$_{G}^{in}$, [kWh]')
# #ax22.set_title('Year 2021')
# #plt.xticks(arange(1, int((end_day - start_day)*24*4), step=24*31*4), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])  # Set label locations. 
# ax22.set_xlabel('TESS volume, [m$^{3}$]')
# #plt.yticks(arange(0, int(y_f/dy), step=10), [round(i,1) for i in arange(0, y_f, step=0.1)])
# ax22.set_ylabel('Number of PVT modules, [-]')        
        

# f23 = plt.figure()
# plt.plot([i-273 for i in T_TESS_registry[1][1]], 'g', label='1 PVT - 1 TESS')
# plt.plot([i-273 for i in T_TESS_registry[1][10]], 'b', label='1 PVT - 10 TESS')
# plt.plot([i-273 for i in T_TESS_registry[5][5]], 'r', label='5 PVT - 5 TESS')
# plt.plot([i-273 for i in T_TESS_registry[10][1]], 'c', label='10 PVT - 1 TESS')
# plt.plot([i-273 for i in T_TESS_registry[10][10]], 'y', label='10 PVT - 10 TESS')
# plt.legend(loc='lower center', ncol = 5, prop={'size': 10})
# plt.grid()
# plt.xlim([0, end_day*24])
# plt.xticks(arange(1, 365*24*4, step=24*31*4), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])  # Set label locations. 
# #plt.xlabel('Time [h]')
# plt.ylabel('TESS temperature, T$_{TESS}$, [°C]')
# plt.show()

# #
# f24 = plt.figure()
# #data = [[i-273 for i in T_TESS_registry[0][2]], [i-273 for i in T_TESS_registry[0][4]], [i-273 for i in T_TESS_registry[0][6]], [i-273 for i in T_TESS_registry[0][8]], [i-273 for i in T_TESS_registry[0][10]]]
# data = [[t-273 for t in i] for i in T_TESS_registry[0]]
# #labels = ['1 m$^{3}$', '2 m$^{3}$', '3 m$^{3}$', '4 m$^{3}$', '5 m$^{3}$', '6 m$^{3}$', '7 m$^{3}$', '8 m$^{3}$', '9 m$^{3}$', '10 m$^{3}$']
# labels = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
# plt.xlabel('TESS volume, V, [m$^{3}$]')
# plt.ylabel('TESS temperature, T$_{TESS}$, [°C]')
# #labels = ['2 m$^{3}$', '4 m$^{3}$', '6 m$^{3}$', '8 m$^{3}$', '10 m$^{3}$']
# plt.boxplot(data[1:], showfliers = False, labels = labels)     # , 'g', label=
# #plt.legend(loc='lower center', ncol = 5)
# plt.show()
# #
# #
# #f25 = plt.figure()
# #plt.plot([i*100 for i in SoC_BESS_registry[10][10]], 'y', label='10 PVT - 10 TESS')
# #plt.plot([i*100 for i in SoC_BESS_registry[10][1]], 'c', label='10 PVT - 1 TESS')
# #plt.plot([i*100 for i in SoC_BESS_registry[5][5]], 'r', label='5 PVT - 5 TESS')
# #plt.plot([i*100 for i in SoC_BESS_registry[1][10]], 'b', label='1 PVT - 10 TESS')
# #plt.plot([i*100 for i in SoC_BESS_registry[1][1]], 'g', label='1 PVT - 1 TESS')
# #plt.legend(loc='lower center', ncol = 5)
# #plt.show()
# #
# #
# #f26 = plt.figure()
# #data = [[i*100 for i in SoC_BESS_registry[1][1] if i > 0.2], [i*100 for i in SoC_BESS_registry[10][1] if i > 0.2], [i*100 for i in SoC_BESS_registry[5][5] if i > 0.2], [i*100 for i in SoC_BESS_registry[1][10] if i > 0.2], [i*100 for i in SoC_BESS_registry[10][10] if i > 0.2]]
# #labels = ['1 PVT - 1 TESS', '10 PVT - 1 TESS', '5 PVT - 5 TESS', '1 PVT - 10 TESS', '10 PVT - 10 TESS']
# #
# #plt.boxplot(data, showfliers = False, labels = labels)     # , 'g', label=
# #
# ##plt.legend(loc='lower center', ncol = 5)
# #plt.show()
# #
# #
# #

# thermal_performance = zeros(shape=(len(n_modules_list[0]), len(TESS_vol_list[0])))

# for n in range(len(n_modules_list[0])-1):
#     for vol in range(len(TESS_vol_list[0])-1):
# #        print(100*(Qdot_TESS_registry[n + 1,vol + 1] - 1000*vol*4200*(T_TESS_0 - T_sup)/(1000*3600))/(Qdot_SC_TESS_registry[n + 1,vol + 1] + Qdot_HP_TESS_registry[n + 1,vol + 1]))
#         thermal_performance[n+1, vol+1] = 100*(Qdot_TESS_registry[n + 1,vol + 1] - 1000*vol*4200*(T_TESS_0 - T_sup)/(1000*3600))/(Qdot_SC_TESS_registry[n + 1,vol + 1] + Qdot_HP_TESS_registry[n + 1,vol + 1])

# fig27, (ax27) = plt.subplots(1, 1, constrained_layout=True, sharey=True)
# a27 = plt.imshow(thermal_performance, cmap='plasma', interpolation='bicubic', origin='lower', vmin = 50, vmax = 70)   # , vmin = -15, vmax = 35
# cbar27 = fig27.colorbar(a27, ax=ax27)
# #cbar27.set_ticks(arange(-15, 36, step=5), [round(i,0) for i in arange(-15, 36, step=5)])
# cbar27.set_label('Thermal performance, [%]')
# #ax27.set_title('Year 2021')
# #plt.xticks(arange(1, int((end_day - start_day)*24*4), step=24*31*4), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])  # Set label locations. 
# ax27.set_xlabel('TESS volume, [m$^{3}$]')
# ax27.set_xlim([1,10])
# ax27.set_ylim([1,10])
# #plt.yticks(arange(0, int(y_f/dy), step=10), [round(i,1) for i in arange(0, y_f, step=0.1)])
# ax27.set_ylabel('Number of PVT modules, [-]')     



# thermal_electric_performance = empty(shape=(len(n_modules_list), len(TESS_vol_list)), dtype='float')

# for n in range(len(n_modules_list)-1):
#     for vol in range(len(TESS_vol_list)-1):
        
#         thermal_electric_performance[n+1, vol+1] = (Qdot_PVT_Network_registry[n + 1,vol + 1] + Qdot_SC_TESS_registry[n + 1,vol + 1])/P_PVT_registry[n + 1,vol + 1]

# fig28, (ax28) = plt.subplots(1, 1, constrained_layout=True, sharey=True)
# a28 = plt.imshow(thermal_electric_performance, cmap='plasma', interpolation='bicubic', origin='lower')   # , vmin = -15, vmax = 35
# cbar28 = fig28.colorbar(a28, ax=ax28)
# #cbar28.set_ticks(arange(-15, 36, step=5), [round(i,0) for i in arange(-15, 36, step=5)])
# cbar28.set_label('Termal/electric PVT output ratio, [-]')
# #ax28.set_title('Year 2021')
# #plt.xticks(arange(1, int((end_day - start_day)*24*4), step=24*31*4), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])  # Set label locations. 
# ax28.set_xlabel('TESS volume, [m$^{3}$]')
# ax28.set_xlim([1,10])
# ax28.set_ylim([1,10])
# #plt.yticks(arange(0, int(y_f/dy), step=10), [round(i,1) for i in arange(0, y_f, step=0.1)])
# ax28.set_ylabel('Number of PVT modules, [-]') 


# Net_grid = [i - j for i, j in zip(P_Grid_consumed_registry, P_Grid_returned_registry)]

# fig29, (ax29) = plt.subplots(1, 1, constrained_layout=True, sharey=True)
# a29 = plt.imshow(Net_grid, cmap='plasma', interpolation='bicubic', origin='lower')   # , vmin = -15, vmax = 35
# cbar29 = fig29.colorbar(a29, ax=ax29)
# #cbar29.set_ticks(arange(-15, 36, step=5), [round(i,0) for i in arange(-15, 36, step=5)])
# cbar29.set_label('Net energy consumption, E$_{net}$, [kWh]')
# #ax29.set_title('Year 2021')
# #plt.xticks(arange(1, int((end_day - start_day)*24*4), step=24*31*4), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])  # Set label locations. 
# ax29.set_xlabel('TESS volume, [m$^{3}$]')
# #ax29.set_xlim([1,10])
# #ax29.set_ylim([1,10])
# #plt.yticks(arange(0, int(y_f/dy), step=10), [round(i,1) for i in arange(0, y_f, step=0.1)])
# ax29.set_ylabel('Number of PVT modules, [-]') 


# electric_performance = 100*Qdot_TESS_registry/P_HP_registry

# fig30, (ax30) = plt.subplots(1, 1, constrained_layout=True, sharey=True)
# a30 = plt.imshow(electric_performance, cmap='plasma', interpolation='bicubic', origin='lower', vmin = 65, vmax = 90)   # , vmin = -15, vmax = 35
# cbar30 = fig30.colorbar(a30, ax=ax30)
# #cbar30.set_ticks(arange(-15, 36, step=5), [round(i,0) for i in arange(-15, 36, step=5)])
# cbar30.set_label('Electric performance, [%]')
# #ax30.set_title('Year 2021')
# #plt.xticks(arange(1, int((end_day - start_day)*24*4), step=24*31*4), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])  # Set label locations. 
# ax30.set_xlabel('TESS volume, [m$^{3}$]')
# ax30.set_xlim([1,10])
# ax30.set_ylim([1,10])
# #plt.yticks(arange(0, int(y_f/dy), step=10), [round(i,1) for i in arange(0, y_f, step=0.1)])
# ax30.set_ylabel('Number of PVT modules, [-]') 

# #import matplotlib.pyplot as plt
# #
# #
# f1 = plt.figure()
# plt.plot([i-273 for i in T_amb], 'r', label='Ambient temperature, $T_{amb}$')
# plt.plot([i-273 for i in T_set], 'g', label='Setpoint temperature inside the house, $T_{set}$')
# #plt.plot([i-273 for i in T_amb[0:4*h]], 'r', label='Ambient temperature, $T_{amb}$')
# plt.plot([i-273 for i in T_in[1:]], 'b', label='Temperature inside the house, $T_{in}$')
# plt.legend(loc='lower center')
# plt.grid()
# plt.xlim([0, end_day*24])
# plt.xticks(arange(1, 365*24*4, step=24*31*4), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])  # Set label locations. 
# #plt.xlabel('Time [h]')
# plt.ylabel('Temperature [°C]')
# #plt.title('Temperature inside the house')
# plt.show()   

# #
# f2 = plt.figure()
# plt.plot([i/1000 for i in Qdot_HP], 'g', label='$\dot{Q}_{HP}$')
# plt.plot([i/1000 for i in Qdot_TESS], 'r', label='$\dot{Q}_{TESS}$')
# plt.plot([i/1000 for i in Qdot_PVT_Network], 'b', label='$\dot{Q}_{PVT}$')
# plt.legend(loc='upper center', ncol = 3)
# plt.grid()
# plt.xlim([0, end_day*24])
# plt.xticks(arange(1, 365*24*4, step=24*31*4), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])  # Set label locations. 
# #plt.xlabel('Time [h]')
# plt.ylabel('Thermal power [kW]')
# #plt.title('Temperature inside the house')
# plt.show()   
# #
# #f3 = plt.figure()
# #plt.plot([i-273 for i in T_registry[4]], 'g', label='$T_{4}$')
# #plt.plot([i-273 for i in T_registry[3]], 'r', label='$T_{3}$')
# #plt.plot([i-273 for i in T_registry[2]], 'b', label='$T_{2}$')
# #plt.plot([i-273 for i in T_registry[1]], 'c', label='$T_{1}$')
# ##plt.plot([i-273 for i in T_registry[0]], 'k', label='$T_{0}$')
# #plt.legend(loc='lower center', ncol = 4)
# #plt.grid()
# #plt.xlim([0, end_day*24])
# #plt.xticks(arange(1, 365*24*4, step=24*31*4), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])  # Set label locations. 
# ##plt.xlabel('Time [h]')
# #plt.ylabel('Temperature [°C]')
# ##plt.title('Temperature inside the house')
# #plt.show()   
# #
# f4 = plt.figure()
# plt.plot([-i/1000 for i in Qdot_Losses], linewidth=0.5)
# plt.grid()
# plt.xlim([0, end_day*24])
# plt.ylim([0,3])
# plt.xticks(arange(1, 365*24*4, step=24*31*4), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])  # Set label locations. 
# #plt.xlabel('Time [h]')
# plt.ylabel('Thermal losses, $\dot{Q}_{L}$, [kW]')
# #plt.title('Temperature inside the house')
# plt.show()  
# #
# f5 = plt.figure()
# plt.plot([i-273 for i in T_TESS])
# plt.grid()
# plt.xlim([0, end_day*24])
# plt.xticks(arange(1, 365*24*4, step=24*31*4), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])  # Set label locations. 
# #plt.xlabel('Time [h]')
# plt.ylabel('Temperature in the TESS, $T_{TESS}$, [°C]')
# #plt.title('Temperature in the TESS')
# plt.show()
# #
# ##f6 = plt.figure()
# ##plt.plot([i/3600 for i in G[start_day*24*4:end_day*24*4]], 'c', label='$G$')
# ##plt.plot([i-273 for i in T_amb[start_day*24*4:end_day*24*4]], 'r', label='$T_{amb}$')
# ###plt.plot([i-273 for i in fluid_registry[0]], 'g', label='$T_{glass}$')
# ###plt.plot([i-273 for i in fluid_registry[1]], 'b', label='$T_{PV}$')
# ###plt.plot([i-273 for i in fluid_registry[2]], 'c', label='$T_{a}$')
# ##plt.plot([i for i in fluid_registry[4]], 'b', label='$T_{out}$')
# ##plt.plot([i for i in fluid_registry[3]], 'y', label='$T_{f}$')
# ##
# ##
# ##plt.grid()
# ##f6.legend(loc='lower center', ncol = 3)
# ##plt.show()
# #

# #T_start = 178
# #T_end = 182
# T_start = 104
# T_end = 108
# f6, ax1 = plt.subplots()

# #ax1.set_xlabel('June 27 - June 28 - June 29 - June 30 - July 1')
# ax1.set_xlabel('November 20 - November 21 - November 22 - November 23 - November 24')
# #ax1.set_xticks(arange(1, 365*24*4, step=24*31*4), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])  # Set label locations. 
# #ax1.set_xticks(arange(1, 365*24*4, step=24*31*4))
# #ax1.set_xticks([])
# ax1.set_ylabel('Global irradiance, G, [W/m$^{2}$]') # , color='b'  # we already handled the x-label with ax1
# ax1.plot([i/3600 for i in G[T_start*96:(T_end+1)*96]], 'teal', label='$G$')
# ax1.set_ylim([0, 1000])
# #ax1.set_yticks(arange(0, 1001, step=30))
# #ax1.tick_params(axis='y', labelcolor='b')

# ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

# ax2.set_ylabel('Temperature, [°C]') # , color='r'
# ax2.plot([i-273 for i in T_amb[T_start*96:(T_end+1)*96]], 'red', label='$T_{amb}$')
# ax2.plot([i for i in fluid_registry[0][T_start*96:(T_end+1)*96]], 'olive', label='$T_{glass}$')
# ax2.plot([i for i in fluid_registry[1][T_start*96:(T_end+1)*96]], 'blue', label='$T_{PV}$')
# ax2.plot([i for i in fluid_registry[2][T_start*96:(T_end+1)*96]], 'navy', label='$T_{a}$')
# ax2.plot([i for i in fluid_registry[4][T_start*96:(T_end+1)*96]], 'magenta', label='$T_{net}$')
# ax2.plot([i for i in fluid_registry[3][T_start*96:(T_end+1)*96]], 'yellow', label='$T_{f}$')
# ax2.plot(T_tank_PVT_registry[T_start*96:(T_end+1)*96], 'black', label='$T_{PVT}$') # T_{tank, PVT}$
# #T_tank_PVT_registry
# ax2.set_xticks(arange(1, (T_end-T_start+1)*96, step=6*4))
# ax2.set_xlim([1,(T_end-T_start+1)*96])
# #ax2.set_ylim([0, 1001])
# #ax2.set_xticks(arange(0, int(t_final/2)+1, step=300))
# #ax2.set_yticks(arange(0, 1001, step=100))
# ax2.tick_params(axis='y', labelcolor='r')

# #f6.legend(loc='lower center', ncol = 8)

# f6.tight_layout()  # otherwise the right y-label is slightly clipped
# ax1.grid()
# plt.show()


# f7 = plt.figure()
# plt.plot(P_Grid, 'y', label='$P_{Grid}$')
# plt.plot(P_HP, 'r', label='$P_{HP}$')
# plt.plot(P_Load, 'c', label='$P_{Load}$')
# plt.plot(P_BESS, 'k', label='$P_{BESS}$')
# plt.plot(P_PVT, 'b', label='$P_{PVT}$')
# plt.xlim([0, end_day*24])
# plt.xticks(arange(1, 365*24*4, step=24*31*4), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])  # Set label locations. 
# plt.ylabel('Electric power, $P$, [kW]')
# plt.legend(loc='lower center', ncol = 5)
# plt.grid()
# plt.show()
# #
# #
# #f8 = plt.figure()
# #plt.plot([i*100 for i in SoC_BESS], label='$SoC_{BESS}$')
# #plt.xlim([0, end_day*24])
# #plt.xticks(arange(1, 365*24*4, step=24*31*4), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])  # Set label locations. 
# #plt.ylabel('State-of-Charge of the BESS, $SoC_{BESS}$, [%]')
# ##plt.legend(loc='lower center', ncol = 1)
# #plt.grid()
# #plt.show()
# #
# #f8 = plt.figure()
# #plt.plot(Qdot_TESS_SD, label='$\dot{Q}_{TESS}^{SD}$')
# #plt.xlim([0, end_day*24])
# #plt.xticks(arange(1, 365*24*4, step=24*31*4), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])  # Set label locations. 
# #plt.ylabel('TESS self-discharge themal power, $\dot{Q}_{TESS}^{SD}$, [W]')
# ##plt.legend(loc='lower center', ncol = 1)
# #plt.grid()
# #plt.show()
# #
