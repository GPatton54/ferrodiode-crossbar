# Kim, et al. (2024)
kim_2024 = {
    'FE' : 'AlScN',
    'Interlayer' : ['Al2O3', 'HfO2'],
    'Top Electrode' : ['Ti', 'Cr'],
    'Bottom Electrode' : 'Al',
    'FE Thickness' : [5, 10, 20], # [nm]
    'Interlayer Thickness' : [0, 2, 4], # [nm]
    'ON/OFF' : [1650, 3170],
    'V_read' : 8.0, # [V]
    'V_write' : [8.0, 11.0], # [V]
    'LRS' : 0.04, # [GOhm]
    'HRS' : 600, # [GOhm]
    'V_range' : [5, 8, 10, 12, 16], # [±V]
    'I_range' : [1e-12, 1e-2] # [A]
}

# Han & Chen, et al. (2025)
han_chen_2025 = {
    'FE' : 'AlScN',
    'Interlayer' : ['Al2O3', 'HfO2'],
    'Top Electrode' : 'Cr',
    'Bottom Electrode' : 'Pt',
    'FE Thickness' : 45, # [nm]
    'Interlayer Thickness' : 10, # [nm]
    'ON/OFF' : 9.91,
    'V_read' : 8.0, # [V]
    'V_write' : 21.0, # [V]
    'LRS' : [75, 92], # [GOhm]
    'HRS' : [135, 177], # [GOhm]
    'V_range' : 15, # [±V]
    'I_range' : [1e-13, 1e-5] # [A]
}

# Liu, et al. (2022)
liu_2022 = {
    'FE' : 'AlScN',
    'Interlayer' : 'None',
    'Top Electrode' : 'Al',
    'Bottom Electrode' : 'Al',
    'FE Thickness' : 45, # [nm]
    'Interlayer Thickness' : 0, # [nm]
    'ON/OFF' : 100,
    'V_read' : 8.0, # [V]
    'V_write' : 15.0, # [V]
    'LRS' : 0.004, # [GOhm]
    'HRS' : [0.04, 0.5], # [GOhm]
    'V_range' : 15, # [±V]
    'I_range' : [1e-10, 1e-4] # [A]
}

# Hu, et al. (2025)
hu_2025 = {
    'FE' : 'AlScN',
    'Interlayer' : 'HfO2',
    'Top Electrode' : 'Ti',
    'Bottom Electrode' : 'Al',
    'FE Thickness' : 20, # [nm]
    'Interlayer Thickness' : 4, # [nm]
    'ON/OFF' : [60, 100],
    'V_read' : 5.0, # [V]
    'V_write' : [7.0, 9.0], # [V]
    'LRS' : 1, # [GOhm]
    'HRS' : 15, # [GOhm]
    'V_range' : [12, 14], # [±V]
    'I_range' : [1e-14, 1e-4] # [A]
}