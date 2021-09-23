# DISCLAIMER TO CONTEST TEAMS :  DO NOT MAKE CHANGES IN THIS FILE.
classes = {
    "Tasit": 0,
    "Insan": 1,
    "UAP": 2,
    "UAI": 3,
}
landing_statuses = {
    "Inilebilir": "1",
    "Inilemez": "0",
    "Inis Alani Degil": "-1"
}

switcher = {
    0: {"classes": "1", "landing_statuses": -1},
    1: {"classes": "1", "landing_statuses": -1},
    2: {"classes": "0", "landing_statuses": -1},
    3: {"classes": "0", "landing_statuses": -1},
    4: {"classes": "0", "landing_statuses": -1},
    5: {"classes": "0", "landing_statuses": -1},
    6: {"classes": "0", "landing_statuses": -1},
    7: {"classes": "0", "landing_statuses": -1},
    8: {"classes": "0", "landing_statuses": -1},
    9: {"classes": "0", "landing_statuses": -1},
    10: {"classes": "0", "landing_statuses": -1},
    11: {"classes": "0", "landing_statuses": -1},  # This may cause problem. What does "others" define?
    12: {"classes": "2", "landing_statuses": 0},  # UAP inilemez
    13: {"classes": "2", "landing_statuses": 1},  # UAP inilebilir
    14: {"classes": "3", "landing_statuses": 0},  # UAI inilemez
    15: {"classes": "0", "landing_statuses": -1},  # is makinesi
    16: {"classes": "3", "landing_statuses": 1},  # UAI inilebilir
}
