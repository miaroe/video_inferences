def get_stations_config(station_config_nr):
    if station_config_nr == 1:  # binary classification
        return {
            '4L': 0,
            '7R': 1,
        }
    elif station_config_nr == 2:  # multiclass classification with almost balanced classes
        return {
            '4L': 0,
            '4R': 1,
            '7L': 2,
            '7R': 3,
            '10R': 4,
            '11R': 5
        }

    elif station_config_nr == 3:  # multiclass classification with unbalanced classes, deleted 7
        return {
            '4L': 0,
            '4R': 1,
            '7L': 2,
            '7R': 3,
            '10L': 4,
            '10R': 5,
            '11L': 6,
            '11R': 7
        }
    elif station_config_nr == 4:  # multiclass classification with unbalanced classes, deleted 7 and 10L
        return {
            '4L': 0,
            '4R': 1,
            '7L': 2,
            '7R': 3,
            '10R': 4,
            '11L': 5,
            '11R': 6
        }
    elif station_config_nr == 5:  # multiclass classification with unbalanced classes, combined 7, 7R and 7L
        return {
            '4L': 0,
            '4R': 1,
            '7': 2,
            '10L': 3,
            '10R': 4,
            '11L': 5,
            '11R': 6
        }
    elif station_config_nr == 6:  # multiclass classification with unbalanced classes, combined 7, 7R and 7L, removed 10L
        return {
            '4L': 0,
            '4R': 1,
            '7': 2,
            '10R': 3,
            '11L': 4,
            '11R': 5
        }
    elif station_config_nr == 7:  # multiclass classification with station 0
        return {
            '0': 0,
            '4L': 1,
            '4R': 2,
            '7L': 3,
            '7R': 4,
            '10L': 5,
            '10R': 6,
            '11L': 7,
            '11R': 8
        }

    elif station_config_nr == 8:  # multiclass classification with unbalanced classes, deleted 7
        return {
            '4L': 1,
            '4R': 2,
            '7L': 3,
            '7R': 4,
            '10L': 5,
            '10R': 6,
            '11L': 7,
            '11R': 8
        }
