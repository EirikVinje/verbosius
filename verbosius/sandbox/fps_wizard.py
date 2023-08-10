import numpy as np

nerf_dart_weight_gram = 1.2 # in grams
nerf_dart_weight_kg = nerf_dart_weight_gram / 1000 # in kg




def calculate_momentum(distance_box, u):

    mass_box = 24.5 / 1000 # in kg
    g = 9.81
    mass_dart = 0.001 

    v = np.sqrt(2*u*g*distance_box) 

    momentum_dart_ms = v * (mass_dart + mass_box) / mass_dart
    momentum_dart_fps = momentum_dart_ms * 3.28084

    return momentum_dart_fps


def find_u(mass_dart, distance_box, momentum_dart_fps):

    mass_box = 24.5 / 1000 # in kg
    g = 9.81

    v = momentum_dart_fps / 3.28084

    u = ((v*mass_dart) / (mass_dart + mass_box))**2 * (1/(2*g*distance_box))

    return u


def data():

    fps_ratings_half = [153, 137, 161, 73, 153, 89, 66]
    distance_box_half = [0.63, 0.74, 1.28, 0.19, 0.12, 0.48, 0.13]

    us = []
    for i in range(len(fps_ratings_half)):
        us.append(find_u(0.001, distance_box_half[i], fps_ratings_half[i]))

    # sort us
    us = sorted(us)[0:-1]

    print("us: ", us)

    return np.mean(us)


if __name__ == "__main__":

    
    distance_box = float(input("Distance box in meters: "))

    u = data()

    print("u: ", u)

    momentum_dart_fps = calculate_momentum(distance_box, u)

    print("Momentum of dart in fps: ", momentum_dart_fps)

