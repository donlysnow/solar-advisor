import pandas as pd

def simulate_battery(df, capacity_kWh, charge_rate_kW, discharge_rate_kW, initial_soc_pct=50):
    """
    Simulates the battery against a given load profile.
    df must contain: 'predicted_pv_kW' and 'total_load_kW'.
    
    Returns a new dataframe with:
    - battery_soc_kWh (state of charge)
    - battery_discharge_kW (power supplied by battery to the house)
    - battery_charge_kW (power sent from solar to battery)
    - grid_import_kW (power needed from the grid)
    - solar_used_kW (solar power directly used by the house)
    """
    df = df.copy()
    dt_hours = 0.25
    soc = capacity_kWh * (initial_soc_pct / 100.0)
    
    soc_list = []
    discharge_list = []
    charge_list = []
    grid_list = []
    solar_used_list = []
    
    pv = df["predicted_pv_kW"].values
    load = df["total_load_kW"].values
    
    for i in range(len(df)):
        pv_now = pv[i]
        load_now = load[i]
        
        # 1. Solar feeds the house first
        solar_to_house = min(pv_now, load_now)
        remaining_load = load_now - solar_to_house
        remaining_pv = pv_now - solar_to_house
        
        # 2. Excess solar charges the battery
        charge_kW = 0.0
        if remaining_pv > 0:
            charge_kW = min(remaining_pv, charge_rate_kW, (capacity_kWh - soc) / dt_hours)
            soc += charge_kW * dt_hours
            
        # 3. If house still needs power, battery discharges
        discharge_kW = 0.0
        if remaining_load > 0:
            discharge_kW = min(remaining_load, discharge_rate_kW, soc / dt_hours)
            soc -= discharge_kW * dt_hours
            remaining_load -= discharge_kW
            
        # 4. Grid covers the rest
        grid_kW = remaining_load
        
        soc_list.append(soc)
        discharge_list.append(discharge_kW)
        charge_list.append(charge_kW)
        grid_list.append(grid_kW)
        solar_used_list.append(solar_to_house)
        
    df["battery_soc_kWh"] = soc_list
    df["battery_discharge_kW"] = discharge_list
    df["battery_charge_kW"] = charge_list
    df["grid_import_kW"] = grid_list
    df["solar_used_kW"] = solar_used_list
    
    # We overwrite surplus_kW to mean "excess solar sent to the grid after everything" for backward compatibility in charts
    df["surplus_kW"] = pv - load - df["battery_charge_kW"] + df["battery_discharge_kW"]
    return df