from calibrateLights import calibrate_light


def run():
    # Calibrate the lights
    light_directions = calibrate_light("./2-10-25 Images/preprocessedImages/", 4)
    print("Light Directions")
    print(light_directions)

    # Generate the normal maps


    # Generate depth map


if __name__ == "__main__":
    run()