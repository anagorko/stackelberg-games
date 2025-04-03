import appdirs


user_data_dir = appdirs.user_data_dir("stackelberg_games.patrolling", "anagorko")
tile_cache_dir = f"{user_data_dir}/tiles/"
plots_dir = f"{user_data_dir}/plots/"
results_dir = f"{user_data_dir}/results/"
