from calibrateLights import calibrate_light
from downcaledTargetNormals import generate_normal_map


def run():
    directory = "./2-10-25 Images/preprocessedImages/"
    # Calibrate the lights
    light_directions = calibrate_light(directory, 4)
    # Generate the normal maps
    generate_normal_map(directory)



    # Generate depth map


if __name__ == "__main__":
    run()