from download_datasets import download_all
from process_dem import process_dem
from calculate_slope import calculate_slope_aspect
from calculate_twi import calculate_twi
from calculate_distance_to_water import calculate_distance_to_water
from process_rainfall import process_rainfall
from build_feature_grid import build_feature_grid
from build_training_dataset import build_training_dataset


def main() -> None:
    download_all()
    process_dem()
    calculate_slope_aspect()
    calculate_twi()
    process_rainfall()
    calculate_distance_to_water()
    build_feature_grid()
    build_training_dataset()


if __name__ == "__main__":
    main()
